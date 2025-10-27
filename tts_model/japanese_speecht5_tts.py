# -*- coding: utf-8 -*-
"""
esnya/japanese_speecht5_tts をローカル保存して日本語TTSを行うラッパ。

想定環境:
- Python 3.11
- transformers>=4.46.0
- torch (GPU, CUDA 12.4 / pytorch 2.6.*)
- huggingface_hub
- soundfile (pysoundfile)
- datasets（話者 xvector の既定取得に使用）

参考（SpeechT5の使用方針 本モデル準拠）:
- SpeechT5OpenjtalkTokenizer + SpeechT5FeatureExtractor -> SpeechT5Processor
- SpeechT5ForTextToSpeech / SpeechT5HifiGan
- 話者埋め込みは 16 次元ベクトル（[-1, 1] 一様分布）を想定（PDFの使用例に準拠）

機能:
- 最初に huggingface_hub.snapshot_download で TTS本体 / Vocoder をローカル保存（存在時はスキップ）
- SpeechT5 の generate（generate_speech/generate）で推論
- 日本語テキストをそのまま渡せるよう trust_remote_code=True を使用
- GPU を index（0/1/..）で指定可能
- wav へ保存（必要に応じてサンプリングレート変更、保存は堅牢化: soundfile→torchaudio→wave 直書き）

注意:
- 既定では "Matthijs/cmu-arctic-xvectors" の index=7306 を話者埋め込みとして使用（成人女性の例）
  ・必要なら speaker_index で上書き可能
  ・独自xvector(.npy)を指定したい場合は xvector_npy を使用（shape: (512,) or (1,512)）
- Vocoder は "microsoft/speecht5_hifigan" を使用
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

# soundfile は conda-forge の pysoundfile パッケージ名
try:
    import soundfile as sf
except Exception as e:
    raise RuntimeError("soundfile(pysoundfile) が見つかりません。環境の依存関係を確認してください。") from e

# SciPy があれば高品質リサンプルに使用（無ければスキップして元のSRで保存）
try:
    from scipy.signal import resample_poly  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ---- 軽量ユーティリティ ----

def _get_hf_token(enable_transfer: bool = True) -> Optional[str]:
    token = os.getenv("HF_TOKEN", None)
    if token and enable_transfer:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return token


def _safe_model_dirname(model_id: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)


def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _build_device(gpu: int) -> str:
    if torch.cuda.is_available() and gpu is not None and gpu >= 0:
        return f"cuda:{gpu}"
    return "cpu"


def _maybe_resample(y: np.ndarray, src_sr: int, dst_sr: Optional[int]) -> Tuple[np.ndarray, int]:
    if dst_sr is None or dst_sr == src_sr:
        return y, src_sr
    if not HAS_SCIPY:
        return y, src_sr
    import math
    g = math.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    y_rs = resample_poly(y, up, down, axis=0)
    return y_rs.astype(np.float32), dst_sr


def _finalize_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    最高品質化のための最終処理:
    - DCオフセット除去（平均値を引く）
    - -1.0 dBFS 目標のピーク正規化（クリッピング防止）
    - クリックノイズ防止のフェードイン/フェードアウト（各10ms）
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
        x = x.T
    # DC offset remove
    x = x - np.mean(x, axis=0, keepdims=False)
    # Peak normalize to -1 dBFS
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    target = 10 ** (-1.0 / 20.0)  # -1 dBFS
    if peak > 0:
        gain = min(target / peak, 1.0)
        x = x * gain
    # 10ms fade in/out
    nfade = max(int(sr * 0.01), 1)
    if x.ndim == 1 and x.size >= nfade * 2:
        ramp = np.linspace(0.0, 1.0, nfade, dtype=np.float32)
        x[:nfade] *= ramp
        x[-nfade:] *= ramp[::-1]
    elif x.ndim == 2 and x.shape[0] >= nfade * 2:
        ramp = np.linspace(0.0, 1.0, nfade, dtype=np.float32)[:, None]
        x[:nfade, :] *= ramp
        x[-nfade:, :] *= ramp[::-1]
    # safety clamp
    x = np.clip(x, -1.0, 1.0)
    return x


def _save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """
    WAV 保存の堅牢化:
    - まず soundfile で保存（WAV/PCM16 明示）
    - 失敗時は torchaudio.save（(C,T) 形状）へフォールバック
    - 最後に wave 直書き（PCM16）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        sf.write(path, audio, int(sr), subtype="PCM_24", format="WAV")
        return
    except Exception:
        pass
    try:
        import torchaudio  # type: ignore
        t = torch.from_numpy(audio.T if audio.ndim == 2 else audio[np.newaxis, ...])
        torchaudio.save(path, t, int(sr), format="wav")
        return
    except Exception:
        pass
    import wave
    x = np.clip(audio, -1.0, 1.0)
    x16 = (x * 32767.0).astype(np.int16)
    if x16.ndim == 1:
        frames = x16.tobytes()
        nch = 1
    else:
        if x16.shape[0] <= 8 and x16.shape[1] > x16.shape[0]:
            x16 = x16.T
        nch = x16.shape[1]
        frames = x16.reshape(-1, nch).tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(frames)


def _ensure_local_model(model_id: str, local_dir: Optional[str] = None) -> str:
    """
    モデル/Vocoder をローカルへ snapshot_download（存在すればそのまま使用）。
    """
    if local_dir is None:
        local_dir = os.path.join("models", _safe_model_dirname(model_id))
    os.makedirs(local_dir, exist_ok=True)
    # 簡易判定: 何かしらの構成 or 重みがあればOK
    has_any = any(fn.endswith((".safetensors", ".bin", "config.json", "preprocessor_config.json")) for fn in os.listdir(local_dir))
    if has_any:
        return local_dir
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=_get_hf_token(enable_transfer=True),
    )
    return local_dir


def _load_default_xvector(
    device: str,
    speaker_index: int = 7306,
    split: str = "validation",
):
    """
    既定の話者埋め込みを datasets から取得（Matthijs/cmu-arctic-xvectors）。
    - 戻り値: torch.FloatTensor shape (1, 512) on device
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets が見つかりません。`pip/conda`で datasets をインストールしてください。") from e

    try:
        ds = load_dataset("Matthijs/cmu-arctic-xvectors", split=split)
        sp = ds[int(speaker_index)]["xvector"]  # list[float] 長さ512
        emb = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)  # (1,512)
        return emb.to(device)
    except Exception as e:
        raise RuntimeError(f"xvector のロードに失敗しました: {e}")


def _patch_tokenizer_compat(tokenizer) -> None:
    """
    互換性パッチ:
    - 一部 transformers バージョンで tokenizer.split_special_tokens が参照されるが、
      外部実装の OpenJTalk トークナイザに未定義の場合がある。
      その場合に限り no-op の split_special_tokens を付与してエラーを防ぐ。
    """
    try:
        import types
        if not hasattr(tokenizer, "split_special_tokens"):
            def _split_special_tokens(self, text):
                return text
            try:
                tokenizer.split_special_tokens = types.MethodType(_split_special_tokens, tokenizer)
            except Exception:
                pass
            try:
                tokenizer.__class__.split_special_tokens = _split_special_tokens  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        # パッチはベストエフォート（失敗しても処理継続）
        pass

def _ensure_tokenizer_class_compat(TokenCls) -> None:
    """
    インスタンス生成前にクラス側へ最低限の互換APIを付与するパッチ。
    - split_special_tokens: 存在しない場合に no-op を定義
    - _in_target_context_manager: 既定 False
    - as_target_tokenizer: 簡易コンテキストマネージャ
    - is_fast: 既定 False
    """
    try:
        from contextlib import contextmanager

        # split_special_tokens
        if not hasattr(TokenCls, "split_special_tokens"):
            def _split_special_tokens(self, text):
                # list/tuple が来るケースも無変換でそのまま返す
                return text
            setattr(TokenCls, "split_special_tokens", _split_special_tokens)

        # 既定フラグ
        if not hasattr(TokenCls, "_in_target_context_manager"):
            try:
                setattr(TokenCls, "_in_target_context_manager", False)
            except Exception:
                pass

        # as_target_tokenizer
        if not hasattr(TokenCls, "as_target_tokenizer"):
            @contextmanager
            def _as_target_tokenizer(self):
                prev = getattr(self, "_in_target_context_manager", False)
                try:
                    setattr(self, "_in_target_context_manager", True)
                except Exception:
                    pass
                try:
                    yield self
                finally:
                    try:
                        setattr(self, "_in_target_context_manager", prev)
                    except Exception:
                        pass
            try:
                setattr(TokenCls, "as_target_tokenizer", _as_target_tokenizer)
            except Exception:
                pass

        # is_fast
        if not hasattr(TokenCls, "is_fast"):
            try:
                setattr(TokenCls, "is_fast", False)
            except Exception:
                pass

    except Exception:
        # ベストエフォート
        pass

def _force_patch_tokenizer_module() -> None:
    """
    モジュール側のクラスオブジェクトへ互換APIを強制付与（from_pretrained 実行前に実施）。
    - _ensure_tokenizer_class_compat をモジュール内の SpeechT5OpenjtalkTokenizer へ適用
    """
    try:
        import speecht5_openjtalk_tokenizer as _tokmod  # type: ignore
        TokenCls = getattr(_tokmod, "SpeechT5OpenjtalkTokenizer", None)
        if TokenCls is not None:
            _ensure_tokenizer_class_compat(TokenCls)
    except Exception:
        # ベストエフォート
        pass

# ---- メインAPI ----

DEFAULT_MODEL_ID = "esnya/japanese_speecht5_tts"
DEFAULT_VOCODER_ID = "microsoft/speecht5_hifigan"
DEFAULT_SPEAKER_INDEX = 7306  # よく使われる女性話者例
DEFAULT_SPEAKER_SPLIT = "validation"


def synthesize_to_wav(
    text: str,
    out_wav_path: str,
    gpu: int = 0,
    model_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    vocoder_id: Optional[str] = None,
    vocoder_local_dir: Optional[str] = None,
    speaker_index: int = DEFAULT_SPEAKER_INDEX,
    speaker_split: str = DEFAULT_SPEAKER_SPLIT,
    xvector_npy: Optional[str] = None,
    seed: Optional[int] = None,
    sample_rate: Optional[int] = None,
) -> str:
    """
    SpeechT5 (esnya/japanese_speecht5_tts) で text を音声化して out_wav_path に保存。

    引数:
    - text: 話させたい日本語テキスト
    - out_wav_path: 保存先 .wav パス（親ディレクトリは自動作成）
    - gpu: 使用GPUのインデックス（0,1,...）。GPUがなければCPU。
    - model_id: 使用するHFモデルID（未指定は DEFAULT_MODEL_ID）
    - local_dir: 上記モデルのローカル保存先（未指定は ./models/...）
    - vocoder_id: 使用するVocoder ID（未指定は DEFAULT_VOCODER_ID）
    - vocoder_local_dir: Vocoder のローカル保存先（未指定は ./models/...）
    - speaker_index: 既定の xvector の取得に使用する index（cmu-arctic-xvectors）
    - speaker_split: xvector 取得に用いる split（train/validation/test など）
    - xvector_npy: 独自xvectorファイル (.npy; shape=(512,) or (1,512)) を使用する場合に指定
    - seed: 乱数シード（再現性が必要なとき）
    - sample_rate: 望ましいサンプリングレート（未指定は vocoder/model の既定）

    戻り値: 保存した wav の絶対パス
    """
    assert isinstance(text, str) and text.strip(), "text が空です"
    model_id = model_id or DEFAULT_MODEL_ID
    vocoder_id = vocoder_id or DEFAULT_VOCODER_ID

    device = _build_device(gpu)
    dtype = _select_dtype()

    local_model = _ensure_local_model(model_id, local_dir)
    local_vocoder = _ensure_local_model(vocoder_id, vocoder_local_dir)

    # 生成再現のための generator（seed 指定時）
    generator = None
    if seed is not None:
        try:
            device_for_gen = device if device.startswith("cuda") else "cpu"
        except Exception:
            device_for_gen = "cpu"
        generator = torch.Generator(device=device_for_gen)
        generator.manual_seed(int(seed))

    # Transformers: Processor / Model / Vocoder
    processor = None
    model = None
    vocoder = None
    sr = None

    # 1) Vocoder
    try:
        from transformers import SpeechT5HifiGan  # type: ignore
        vocoder = SpeechT5HifiGan.from_pretrained(local_vocoder, trust_remote_code=True)
        # SR 推定
        try:
            sr = getattr(vocoder.config, "sampling_rate", None) or getattr(vocoder, "sampling_rate", None)
        except Exception:
            sr = None
    except Exception as e:
        raise RuntimeError(f"SpeechT5HifiGan のロードに失敗しました: {e}")

    # 2) Processor（Open JTalk トークナイザ + FeatureExtractor -> Processor）
    try:
        from speecht5_openjtalk_tokenizer import SpeechT5OpenjtalkTokenizer  # type: ignore
        from transformers import SpeechT5FeatureExtractor, SpeechT5Processor  # type: ignore
        _force_patch_tokenizer_module()

        # 高品質優先: pyopenjtalk が利用可能ならオリジナルG2Pを使用。
        # 利用不可（ABI不整合等）の場合のみ、安全フォールバック実装に切替。
        try:
            import pyopenjtalk as _ojt  # type: ignore
        except Exception:
            try:
                import speecht5_openjtalk_tokenizer as _tokmod  # type: ignore
                import re as _re
                def _safe_g2p_with_np(_text: str, _np_list: str):
                    _pat = _re.compile(f"([{_re.escape(_np_list)}])")
                    _out = []
                    for _seg in _pat.split(_text):
                        if not _seg:
                            continue
                        if _seg in _np_list:
                            _out.append(_seg)
                        else:
                            _out.extend(list(_seg))
                    return _out
                setattr(_tokmod, "_g2p_with_np", _safe_g2p_with_np)
            except Exception:
                pass

        _ensure_tokenizer_class_compat(SpeechT5OpenjtalkTokenizer)
        tokenizer = SpeechT5OpenjtalkTokenizer.from_pretrained(local_model)
        _patch_tokenizer_compat(tokenizer)
        feature_extractor = SpeechT5FeatureExtractor.from_pretrained(local_model)
        processor = SpeechT5Processor(feature_extractor, tokenizer)
    except Exception as e:
        # speecht5_openjtalk_tokenizer が見つからない場合は、モデルリポジトリから動的取得を試みる
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            import importlib.util, types, sys as _sys

            repo_id = model_id or DEFAULT_MODEL_ID
            py_path = hf_hub_download(
                repo_id=repo_id,
                filename="speecht5_openjtalk_tokenizer.py",
                local_dir_use_symlinks=False,
            )
            spec = importlib.util.spec_from_file_location("speecht5_openjtalk_tokenizer", py_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("speecht5_openjtalk_tokenizer の import spec を構築できませんでした。")
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            _sys.modules["speecht5_openjtalk_tokenizer"] = mod

            # 一部 transformers バージョンでは外部トークナイザが
            # transformers.models.speecht5.tokenization_speecht5 から
            # PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES を import しようとして失敗する。
            # 足りない場合は空辞書でパッチして互換性を確保する。
            try:
                from transformers.models.speecht5 import tokenization_speecht5 as _t5tok  # type: ignore
                if not hasattr(_t5tok, "PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES"):
                    _t5tok.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}
            except Exception:
                pass

            spec.loader.exec_module(mod)  # type: ignore[attr-defined]

            # 高品質優先: pyopenjtalk が使えるときは元のG2Pを使用。使えない場合のみフォールバック。
            try:
                import pyopenjtalk as _ojt  # type: ignore
            except Exception:
                try:
                    import re as _re
                    def _safe_g2p_with_np(_text: str, _np_list: str):
                        _pat = _re.compile(f"([{_re.escape(_np_list)}])")
                        _out = []
                        for _seg in _pat.split(_text):
                            if not _seg:
                                continue
                            if _seg in _np_list:
                                _out.append(_seg)
                            else:
                                _out.extend(list(_seg))
                        return _out
                    setattr(mod, "_g2p_with_np", _safe_g2p_with_np)
                except Exception:
                    pass

            SpeechT5OpenjtalkTokenizer = getattr(mod, "SpeechT5OpenjtalkTokenizer")

            from transformers import SpeechT5FeatureExtractor, SpeechT5Processor  # type: ignore
            _force_patch_tokenizer_module()
            _ensure_tokenizer_class_compat(SpeechT5OpenjtalkTokenizer)
            tokenizer = SpeechT5OpenjtalkTokenizer.from_pretrained(local_model)
            _patch_tokenizer_compat(tokenizer)
            feature_extractor = SpeechT5FeatureExtractor.from_pretrained(local_model)
            processor = SpeechT5Processor(feature_extractor, tokenizer)
        except Exception as e2:
            raise RuntimeError(
                "SpeechT5 Processor/tokenizer のロードに失敗しました。"
                " speecht5_openjtalk_tokenizer が未導入の可能性があります。"
                " 対応策: (1) pyopenjtalk/pyopenjtalk-prebuilt を有効化、(2) "
                "モデルリポの speecht5_openjtalk_tokenizer.py を取得できる状態にする（本実装は自動取得を試行済み）。"
                f" 原因1={e}; 原因2={e2}"
            )

    # 3) Model
    try:
        from transformers import SpeechT5ForTextToSpeech  # type: ignore
        model = SpeechT5ForTextToSpeech.from_pretrained(
            local_model, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device).eval()
    except Exception:
        # AutoModel フォールバック（trust_remote_code）
        try:
            from transformers import AutoModel  # type: ignore
            model = AutoModel.from_pretrained(
                local_model, trust_remote_code=True, torch_dtype=torch.float32
            ).to(device).eval()
        except Exception as e:
            raise RuntimeError(f"SpeechT5 モデルのロードに失敗しました: {e}")

    # 入力エンコード
    inputs = processor(text=text, return_tensors="pt")
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # 話者埋め込み（本モデルは 16 次元埋め込みを想定）
    if xvector_npy:
        arr = np.load(xvector_npy)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, ...]
        if arr.shape[-1] != 16:
            raise RuntimeError(f"提供された話者埋め込みの次元が16ではありません: shape={arr.shape}")
        emb = torch.tensor(arr, dtype=getattr(model, "dtype", torch.float32), device=getattr(model, "device", device))
    else:
        rng = np.random.default_rng(seed if seed is not None else None)
        arr = rng.uniform(-1.0, 1.0, size=(1, 16)).astype(np.float32)
        emb = torch.tensor(arr, dtype=getattr(model, "dtype", torch.float32), device=getattr(model, "device", device))

    # 生成（API差異に備えて generate_speech / generate を順に試す）
    speech = None
    with torch.inference_mode():
        gen_kwargs = {}
        if generator is not None:
            gen_kwargs["generator"] = generator
        # generate_speech 優先
        if hasattr(model, "generate_speech"):
            try:
                speech = model.generate_speech(
                    inputs["input_ids"], speaker_embeddings=emb, vocoder=vocoder, **gen_kwargs
                )
            except Exception:
                speech = None
        if speech is None:
            # generate フォールバック（モデル実装に依存）
            try:
                speech = model.generate(
                    **inputs, speaker_embeddings=emb, vocoder=vocoder, **gen_kwargs
                )
            except Exception as e:
                raise RuntimeError(f"SpeechT5 生成に失敗しました: {e}")

    # 出力抽出
    audio = None
    if torch.is_tensor(speech):
        audio = speech.detach().cpu().float().numpy()
    elif isinstance(speech, dict) and "audio" in speech:
        audio = speech["audio"]
    else:
        # (list/tuple) or モデル独自出力
        try:
            audio = np.asarray(speech, dtype=np.float32)
        except Exception:
            pass

    if audio is None:
        raise RuntimeError("SpeechT5 の出力から音声波形を取得できませんでした。")

    audio = np.asarray(audio, dtype=np.float32)
    # 形状正規化: (T,) or (T, C) へ
    if audio.ndim > 2:
        T = audio.shape[0]
        audio = audio.reshape(T, -1)
    elif audio.ndim == 2:
        # (C, T) を (T, C) へ
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T

    # サンプリングレート
    if sr is None:
        try:
            sr = getattr(getattr(model, "config", object()), "sampling_rate", None)
        except Exception:
            sr = None
    if sr is None:
        sr = 16000  # SpeechT5 の一般的既定

    # リサンプル
    audio, final_sr = _maybe_resample(audio, int(sr), sample_rate)

    # 最高品質の最終処理
    audio = _finalize_audio(audio, final_sr)

    # 保存
    out_abs = os.path.abspath(out_wav_path)
    _save_wav(out_abs, audio, final_sr)
    return out_abs


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SpeechT5 (Japanese) wrapper: esnya/japanese_speecht5_tts")
    parser.add_argument("--text", type=str, required=True, help="話させたい日本語テキスト")
    parser.add_argument("--out", type=str, required=True, help="出力wavパス")
    parser.add_argument("--gpu", type=int, default=0, help="使用GPU index（既定: 0）")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="HFモデルID（TTS本体）")
    parser.add_argument("--local-dir", type=str, default=None, help="TTS本体のローカル保存先（未指定は ./models/...）")
    parser.add_argument("--vocoder-id", type=str, default=DEFAULT_VOCODER_ID, help="Vocoder のHFモデルID")
    parser.add_argument("--vocoder-local-dir", type=str, default=None, help="Vocoder のローカル保存先（未指定は ./models/...）")
    parser.add_argument("--speaker-index", type=int, default=DEFAULT_SPEAKER_INDEX, help="既定xvectorのindex（Matthijs/cmu-arctic-xvectors）")
    parser.add_argument("--speaker-split", type=str, default=DEFAULT_SPEAKER_SPLIT, help="既定xvectorのsplit（train/validation/testなど）")
    parser.add_argument("--xvector-npy", type=str, default=None, help="独自xvector(.npy)を使用する場合のパス")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument("--sr", type=int, default=None, help="保存時のサンプリングレート（未指定: モデル/Vocoder既定）")
    args = parser.parse_args()

    path = synthesize_to_wav(
        text=args.text,
        out_wav_path=args.out,
        gpu=args.gpu,
        model_id=args.model_id,
        local_dir=args.local_dir,
        vocoder_id=args.vocoder_id,
        vocoder_local_dir=args.vocoder_local_dir,
        speaker_index=args.speaker_index,
        speaker_split=args.speaker_split,
        xvector_npy=args.xvector_npy,
        seed=args.seed,
        sample_rate=args.sr,
    )
    print(f"[speecht5] saved: {path}")


if __name__ == "__main__":
    _cli()
