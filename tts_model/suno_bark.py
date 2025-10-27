# -*- coding: utf-8 -*-
"""
Suno Bark 系（Hugging Face Transformers）の日本語TTSラッパ。

想定環境:
- Python 3.11
- transformers>=4.46.0
- torch (GPU, CUDA 12.4 / pytorch 2.6.*)
- huggingface_hub
- encodec
- soundfile (pysoundfile)

機能:
- huggingface_hub.snapshot_download でモデルをローカル保存（存在時はスキップ）
- Transformers の pipeline('text-to-speech') 経由で推論（Barkは text-to-audio でも動作）
- 日本語前提の voice_preset を既定で使用（上書き可能）
- GPU を index（0/1/..）で指定可能
- wav へ保存（必要に応じてサンプリングレート変更）

注意:
- Bark は多言語対応で、日本語テキストをそのまま与えると日本語で合成されます。
- voice_preset により話者スタイルを切り替えられます。既定は日本語向けの "v2/ja_speaker_1"。
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


def _ensure_local_model(model_id: str, local_dir: Optional[str] = None) -> str:
    """
    モデルをローカルへ snapshot_download（存在すればそのまま使用）。
    """
    if local_dir is None:
        local_dir = os.path.join("models", _safe_model_dirname(model_id))
    os.makedirs(local_dir, exist_ok=True)

    # 既に main weight があればスキップ（簡易判定）
    has_weight = any(
        fn.endswith((".safetensors", ".bin")) for fn in os.listdir(local_dir)
    )
    if has_weight:
        return local_dir

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=_get_hf_token(enable_transfer=True),
    )
    return local_dir


def _build_device(gpu: int) -> str:
    if torch.cuda.is_available() and gpu is not None and gpu >= 0:
        return f"cuda:{gpu}"
    return "cpu"

def _pipeline_device_arg(gpu: int) -> int:
    """
    transformers.pipeline の device 引数は int（GPU index）または -1(CPU) を期待する版がある。
    互換性のため、ここで適切な値へ変換する。
    """
    if torch.cuda.is_available() and gpu is not None and gpu >= 0:
        try:
            return int(gpu)
        except Exception:
            return 0
    return -1


def _maybe_resample(y: np.ndarray, src_sr: int, dst_sr: Optional[int]) -> Tuple[np.ndarray, int]:
    if dst_sr is None or dst_sr == src_sr:
        return y, src_sr
    if not HAS_SCIPY:
        # SciPy が無い場合は元のSRで保存
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
    # shape to (frames, channels)
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

# ---- メインAPI ----

DEFAULT_MODEL_ID = "suno/bark"
DEFAULT_JA_VOICE = "v2/ja_speaker_1"  # Transformers が提供する日本語話者プリセット


def synthesize_to_wav(
    text: str,
    out_wav_path: str,
    gpu: int = 0,
    model_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    voice_preset: Optional[str] = None,
    seed: Optional[int] = None,
    sample_rate: Optional[int] = None,
) -> str:
    """
    Bark で text を音声化して out_wav_path に保存。

    引数:
    - text: 話させたい日本語テキスト
    - out_wav_path: 保存先 .wav パス（親ディレクトリは自動作成）
    - gpu: 使用GPUのインデックス（0,1,...）。GPUがなければCPU。
    - model_id: 使用するHFモデルID（未指定は DEFAULT_MODEL_ID）
    - local_dir: ローカル保存先（未指定は ./models/{safe_id}）
    - voice_preset: 話者スタイルのプリセット（未指定は DEFAULT_JA_VOICE）
    - seed: 乱数シード（再現性が必要なとき）
    - sample_rate: 望ましいサンプリングレート（未指定はモデル出力に従う）

    戻り値: 保存した wav の絶対パス
    """
    assert isinstance(text, str) and text.strip(), "text が空です"
    model_id = model_id or DEFAULT_MODEL_ID
    voice_preset = voice_preset or DEFAULT_JA_VOICE

    device = _build_device(gpu)
    dtype = _select_dtype()

    local_path = _ensure_local_model(model_id, local_dir)

    from transformers import pipeline

    generator = None
    if seed is not None:
        try:
            device_for_gen = f"cuda:{gpu}" if (torch.cuda.is_available() and gpu is not None and gpu >= 0) else "cpu"
        except Exception:
            device_for_gen = "cpu"
        generator = torch.Generator(device=device_for_gen)
        generator.manual_seed(int(seed))

    pipe = None
    last_err = None
    for task in ("text-to-speech", "text-to-audio"):
        # device 引数の互換確保: "cuda:0" -> int(gpu) -> -1 の順に試行
        for dev_arg in (device, _pipeline_device_arg(gpu), -1):
            try:
                # CPU で動作する場合は float32 を強制（float16/bfloat16 は避ける）
                use_dtype = dtype
                if (isinstance(dev_arg, int) and dev_arg == -1) or (isinstance(dev_arg, str) and dev_arg.startswith("cpu")):
                    use_dtype = torch.float32
                try:
                    pipe = pipeline(
                        task,
                        model=local_path,
                        device=dev_arg,
                        torch_dtype=use_dtype,
                        trust_remote_code=True,
                    )
                except TypeError:
                    # 古い Transformers では torch_dtype 引数自体が未対応のため、dtype指定なしで再試行
                    pipe = pipeline(
                        task,
                        model=local_path,
                        device=dev_arg,
                        trust_remote_code=True,
                    )
                # 可能ならGPUへ転送
                try:
                    if hasattr(pipe, "to") and isinstance(device, str):
                        pipe.to(device)
                except Exception:
                    pass
                break
            except Exception as e:
                last_err = e
                pipe = None
        if pipe is not None:
            break
    if pipe is None:
        raise RuntimeError(f"pipeline 構築に失敗しました（{model_id}）。最後の例外: {last_err}")

    gen_kwargs = {}
    if generator is not None:
        gen_kwargs["generator"] = generator

    # Bark 向けには voice_preset を渡せる場合がある。互換性のため例外時は外す。
    try:
        out = pipe(text, voice_preset=voice_preset, **gen_kwargs)
    except TypeError:
        out = pipe(text, **gen_kwargs)

    # 出力抽出（Transformers の版差に対応: audio が dict/ndarray/tensor/パスのいずれでも可）
    audio = None
    sr = None

    def _unpack_audio_payload(obj):
        if obj is None:
            return None, None
        # payload が dict の場合
        if isinstance(obj, dict):
            # 新仕様: {"array": np.ndarray, "sampling_rate": int}
            if "array" in obj and (obj.get("sampling_rate") or obj.get("sample_rate")):
                arr = obj["array"]
                rate = obj.get("sampling_rate") or obj.get("sample_rate")
                return arr, rate
            # 旧仕様/一時ファイル: {"path": "/tmp/xxx.wav"}
            if "path" in obj:
                path = obj["path"]
                try:
                    arr, rate = sf.read(path, dtype="float32", always_2d=False)
                    return arr, rate
                except Exception:
                    return None, None
            # さらにネストされた dict 形式
            if "audio" in obj:
                return _unpack_audio_payload(obj["audio"])
        # list/tuple で先頭要素が dict or payload
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            a0, r0 = _unpack_audio_payload(obj[0])
            if a0 is not None:
                return a0, r0
        # torch.Tensor / np.ndarray
        if torch.is_tensor(obj):
            return obj.detach().cpu().float().numpy(), None
        try:
            arr = np.asarray(obj)
            if arr.dtype.kind in ("f", "i", "u"):
                return arr.astype(np.float32, copy=False), None
        except Exception:
            pass
        return None, None

    # pipeline の返り値から抽出
    if isinstance(out, dict):
        audio, sr = _unpack_audio_payload(out.get("audio"))
        if sr is None:
            sr = out.get("sampling_rate") or out.get("sample_rate")
    else:
        audio, sr = _unpack_audio_payload(out)

    if audio is None or sr is None:
        raise RuntimeError("audio 波形または sampling_rate を解釈できませんでした。Transformers の出力仕様をご確認ください。")

    # 波形の形状整理（(T,) または (T, C)）
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 2:
        T = audio.shape[0]
        audio = audio.reshape(T, -1)
    elif audio.ndim == 2:
        # (C, T) 形式で返る実装への対処（C は小数、T は大数）
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T

    # サンプリングレート調整
    audio, final_sr = _maybe_resample(audio, int(sr), sample_rate)
    audio = _finalize_audio(audio, final_sr)

    out_abs = os.path.abspath(out_wav_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    _save_wav(out_abs, audio, final_sr)
    return out_abs


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Suno Bark (Japanese) wrapper")
    parser.add_argument("--text", type=str, required=True, help="話させたい日本語テキスト")
    parser.add_argument("--out", type=str, required=True, help="出力wavパス")
    parser.add_argument("--gpu", type=int, default=0, help="使用GPU index（既定: 0）")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="HFモデルID")
    parser.add_argument("--local-dir", type=str, default=None, help="ローカル保存先（未指定は ./models/...）")
    parser.add_argument("--voice-preset", type=str, default=DEFAULT_JA_VOICE, help="Bark の話者プリセット（例: v2/ja_speaker_1）")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument("--sr", type=int, default=None, help="保存時のサンプリングレート（未指定: モデル出力に従う）")
    args = parser.parse_args()

    path = synthesize_to_wav(
        text=args.text,
        out_wav_path=args.out,
        gpu=args.gpu,
        model_id=args.model_id,
        local_dir=args.local_dir,
        voice_preset=args.voice_preset,
        seed=args.seed,
        sample_rate=args.sr,
    )
    print(f"[bark] saved: {path}")


if __name__ == "__main__":
    _cli()
