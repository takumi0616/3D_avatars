# -*- coding: utf-8 -*-
"""
2121-8/japanese-parler-tts-large-bate をローカル保存して日本語TTSを行うラッパ。

参考PDFの使用例に忠実:
- ParlerTTSForConditionalGeneration（parler-tts）と AutoTokenizer（Transformers）
- 「description」を音色/話者記述としてトークナイズ（input_ids）
- 「text」（話させたい日本語）をトークナイズ（prompt_input_ids）
- model.generate(input_ids=..., prompt_input_ids=...) で音声生成
- RubyInserter の add_ruby() を使える場合は text へ適用（日本語向けの読みの付与）

想定環境:
- Python 3.11
- transformers>=4.46.0
- torch (GPU, CUDA 12.4 / pytorch 2.6.*)
- huggingface_hub
- soundfile (pysoundfile)
- 可能なら: parler-tts, rubyinserter

注意:
- large-bate は独自 tokenizer を採用。AutoTokenizer は本モデルIDからロードする。
- 生成は不安定な場合があるため、パラメータは必要に応じて調整（do_sample, temperature 等）。
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

# RubyInserter（任意）
def _try_add_ruby(text: str) -> str:
    try:
        from rubyinserter import add_ruby  # type: ignore
        return add_ruby(text)
    except Exception:
        return text


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


def _ensure_local_model(model_id: str, local_dir: Optional[str]) -> str:
    """
    モデルをローカルへ snapshot_download（存在すればそのまま使用）。
    """
    if local_dir is None:
        local_dir = os.path.join("models", _safe_model_dirname(model_id))
    os.makedirs(local_dir, exist_ok=True)
    # 簡易: 主要ファイルが何かしらあればスキップ
    has_any = any(
        fn.endswith((".safetensors", ".bin", "config.json", "tokenizer.json", "preprocessor_config.json"))
        for fn in os.listdir(local_dir)
    )
    if has_any:
        return local_dir
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=_get_hf_token(enable_transfer=True),
    )
    return local_dir


# ---- メインAPI ----

DEFAULT_MODEL_ID = "2121-8/japanese-parler-tts-large-bate"

# 日本語向け 既定の話者記述（description）。必要に応じて上書き可能。
DEFAULT_DESCRIPTION = "自然な成人女性の声。やや高めのピッチで、明瞭な発音、丁寧で落ち着いたトーン。適度な抑揚で読み上げる。"


def synthesize_to_wav(
    text: str,
    out_wav_path: str,
    gpu: int = 0,
    model_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    description: Optional[str] = None,
    # 後方互換: run_tts.py の --parler-prompt を description に流用
    prompt: Optional[str] = None,
    use_ruby: bool = True,
    seed: Optional[int] = None,
    sample_rate: Optional[int] = None,
) -> str:
    """
    japanese-parler-tts-large-bate で text を音声化して out_wav_path に保存。
    - text: 話させたい日本語テキスト
    - description: 話者/スタイル記述（英日どちらでも可）。未指定なら DEFAULT_DESCRIPTION。
    - prompt: 互換目的（--parler-prompt）。指定があれば description として扱う。
    - use_ruby: True なら text に add_ruby() を適用（利用可能な環境のみ）
    """
    assert isinstance(text, str) and text.strip(), "text が空です"
    model_id = model_id or DEFAULT_MODEL_ID
    desc = (description or prompt or DEFAULT_DESCRIPTION).strip()

    device = _build_device(gpu)
    dtype = _select_dtype()

    local_path = _ensure_local_model(model_id, local_dir)

    # 生成再現のための generator（seed 指定時）
    generator = None
    if seed is not None:
        try:
            device_for_gen = device if device.startswith("cuda") else "cpu"
        except Exception:
            device_for_gen = "cpu"
        generator = torch.Generator(device=device_for_gen)
        generator.manual_seed(int(seed))

    # 1) 参考実装に忠実: parler-tts ライブラリ + AutoTokenizer
    try:
        try:
            from parler_tts import ParlerTTSForConditionalGeneration  # type: ignore
        except Exception:
            # 一部環境では transformers 側にある場合もあるため試行
            from transformers import ParlerTTSForConditionalGeneration  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(local_path)
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            local_path, torch_dtype=torch.float32
        ).to(device).eval()

        # description と text をそれぞれトークナイズ（attention_mask も取得）
        desc_tok = tok(desc, return_tensors="pt")
        input_ids = desc_tok.input_ids.to(device)
        desc_attn = getattr(desc_tok, "attention_mask", None)
        if desc_attn is not None:
            desc_attn = desc_attn.to(device)

        text_in = _try_add_ruby(text) if use_ruby else text
        prompt_tok = tok(text_in, return_tensors="pt")
        prompt_input_ids = prompt_tok.input_ids.to(device)

        gen_kwargs = {}
        if generator is not None:
            gen_kwargs["generator"] = generator
        # attention_mask を指定できる場合は付与（信頼性向上）
        if 'desc_attn' in locals() and desc_attn is not None:
            gen_kwargs["attention_mask"] = desc_attn

        with torch.inference_mode():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                **gen_kwargs,
            )

        # 出力の抽出
        if torch.is_tensor(generation):
            audio = generation.detach().cpu().float().numpy()
        elif isinstance(generation, dict) and "audio_values" in generation:
            audio = np.asarray(generation["audio_values"], dtype=np.float32)
        else:
            try:
                audio = np.asarray(generation, dtype=np.float32)
            except Exception:
                audio = None
        if audio is None:
            raise RuntimeError("Parler-TTS generate の出力から音声を取得できませんでした。")

        # 形状正規化: (T,) or (T,C)
        if audio.ndim > 2:
            T = audio.shape[0]
            audio = audio.reshape(T, -1)
        elif audio.ndim == 2 and (audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]):
            audio = audio.T

        sr = None
        try:
            sr = getattr(getattr(model, "config", object()), "sampling_rate", None) or getattr(model, "sampling_rate", None)
        except Exception:
            sr = None
        if sr is None:
            sr = 24000

        audio, final_sr = _maybe_resample(audio, int(sr), sample_rate)
        audio = _finalize_audio(audio, final_sr)

        out_abs = os.path.abspath(out_wav_path)
        _save_wav(out_abs, audio, final_sr)
        return out_abs
    except Exception as e_primary:
        last_err = e_primary

    # 2) フォールバック: AutoModel + AutoTokenizer（trust_remote_code=True）
    try:
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        use_dtype = torch.float32
        model = AutoModel.from_pretrained(
            local_path, trust_remote_code=True, torch_dtype=use_dtype
        ).to(device).eval()

        # description と text をそれぞれトークナイズ（attention_mask も取得）
        desc_tok = tok(desc, return_tensors="pt")
        input_ids = desc_tok.input_ids.to(device)
        desc_attn = getattr(desc_tok, "attention_mask", None)
        if desc_attn is not None:
            desc_attn = desc_attn.to(device)

        text_in = _try_add_ruby(text) if use_ruby else text
        prompt_tok = tok(text_in, return_tensors="pt")
        prompt_input_ids = prompt_tok.input_ids.to(device)

        gen_kwargs = {}
        if generator is not None:
            gen_kwargs["generator"] = generator
        if 'desc_attn' in locals() and desc_attn is not None:
            gen_kwargs["attention_mask"] = desc_attn

        with torch.inference_mode():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                **gen_kwargs,
            )

        # 出力の抽出
        if torch.is_tensor(generation):
            audio = generation.detach().cpu().float().numpy()
        elif isinstance(generation, dict) and "audio_values" in generation:
            audio = np.asarray(generation["audio_values"], dtype=np.float32)
        else:
            try:
                audio = np.asarray(generation, dtype=np.float32)
            except Exception:
                audio = None
        if audio is None:
            raise RuntimeError("Parler AutoModel.generate の出力から音声を取得できませんでした。")

        # 形状正規化
        if audio.ndim > 2:
            T = audio.shape[0]
            audio = audio.reshape(T, -1)
        elif audio.ndim == 2 and (audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]):
            audio = audio.T

        # サンプリングレート取得
        sr = (
            getattr(getattr(model, "generation_config", object()), "sample_rate", None)
            or getattr(getattr(model, "config", object()), "sampling_rate", None)
            or getattr(model, "sampling_rate", None)
            or 24000
        )

        audio, final_sr = _maybe_resample(audio, int(sr), sample_rate)
        audio = _finalize_audio(audio, final_sr)
        out_abs = os.path.abspath(out_wav_path)
        _save_wav(out_abs, audio, final_sr)
        return out_abs
    except Exception as e_auto:
        last_err = e_auto

    # 3) フォールバック: Transformers pipeline（text-to-speech / text-to-audio）
    try:
        from transformers import pipeline  # type: ignore
        # pipeline は text を1本で受けるため、desc を先頭に付与して擬似的にスタイルを伝える
        text_in = _try_add_ruby(text) if use_ruby else text
        merged = f"{desc}\n{text_in}" if desc else text_in

        pipe = None
        for task in ("text-to-speech", "text-to-audio"):
            try:
                try:
                    pipe = pipeline(
                        task,
                        model=local_path,
                        device=(gpu if (torch.cuda.is_available() and gpu is not None and gpu >= 0) else -1),
                        torch_dtype=(dtype if device.startswith("cuda") else torch.float32),
                        trust_remote_code=True,
                    )
                except TypeError:
                    # 古い Transformers では torch_dtype 引数自体が未対応のため、dtype 指定なしで再試行
                    pipe = pipeline(
                        task,
                        model=local_path,
                        device=(gpu if (torch.cuda.is_available() and gpu is not None and gpu >= 0) else -1),
                        trust_remote_code=True,
                    )
                break
            except Exception as _:
                pipe = None
        if pipe is None:
            raise RuntimeError(f"Transformers pipeline 構築に失敗しました: {last_err}")

        out = pipe(merged)
        # 出力抽出（audio: dict/ndarray/tensor/一時WAV）
        audio = None
        sr = None

        def _unpack(obj):
            if obj is None:
                return None, None
            if isinstance(obj, dict):
                if "array" in obj and (obj.get("sampling_rate") or obj.get("sample_rate")):
                    arr = obj["array"]
                    rate = obj.get("sampling_rate") or obj.get("sample_rate")
                    return arr, rate
                if "path" in obj:
                    try:
                        arr, rate = sf.read(obj["path"], dtype="float32", always_2d=False)
                        return arr, rate
                    except Exception:
                        return None, None
                if "audio" in obj:
                    return _unpack(obj["audio"])
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                return _unpack(obj[0])
            if torch.is_tensor(obj):
                return obj.detach().cpu().float().numpy(), None
            try:
                arr = np.asarray(obj)
                if arr.dtype.kind in ("f", "i", "u"):
                    return arr.astype(np.float32, copy=False), None
            except Exception:
                pass
            return None, None

        if isinstance(out, dict):
            audio, sr = _unpack(out.get("audio"))
            if sr is None:
                sr = out.get("sampling_rate") or out.get("sample_rate")
        else:
            audio, sr = _unpack(out)

        if audio is None or sr is None:
            raise RuntimeError("pipeline 出力から音声/サンプリングレートを取得できませんでした。")

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 2:
            T = audio.shape[0]
            audio = audio.reshape(T, -1)
        elif audio.ndim == 2 and (audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]):
            audio = audio.T

        audio, final_sr = _maybe_resample(audio, int(sr), sample_rate)
        out_abs = os.path.abspath(out_wav_path)
        _save_wav(out_abs, audio, final_sr)
        return out_abs
    except Exception as e_fallback:
        raise RuntimeError(f"Parler large-bate 生成に失敗しました。primary={last_err}, fallback={e_fallback}")


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Japanese Parler-TTS Large (β) wrapper")
    parser.add_argument("--text", type=str, required=True, help="話させたい日本語テキスト")
    parser.add_argument("--out", type=str, required=True, help="出力wavパス")
    parser.add_argument("--gpu", type=int, default=0, help="使用GPU index（既定: 0）")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="HFモデルID")
    parser.add_argument("--local-dir", type=str, default=None, help="ローカル保存先（未指定は ./models/...）")
    parser.add_argument("--description", type=str, default=None, help="話者/スタイル記述（未指定は既定）")
    parser.add_argument("--prompt", type=str, default=None, help="後方互換: --description と同義")
    parser.add_argument("--no-ruby", action="store_true", help="RubyInserter の add_ruby を無効化")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument("--sr", type=int, default=None, help="保存時のサンプリングレート（未指定: モデル既定）")
    args = parser.parse_args()

    path = synthesize_to_wav(
        text=args.text,
        out_wav_path=args.out,
        gpu=args.gpu,
        model_id=args.model_id,
        local_dir=args.local_dir,
        description=args.description,
        prompt=args.prompt,
        use_ruby=(not args.no_ruby),
        seed=args.seed,
        sample_rate=args.sr,
    )
    print(f"[parler-large-bate] saved: {path}")


if __name__ == "__main__":
    _cli()
