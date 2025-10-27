# -*- coding: utf-8 -*-
# =============================================================================
# 使い方（GUIなしLinuxサーバー＋Docker/Conda前提・通信制限を考慮、GPU選択対応）
# -----------------------------------------------------------------------------
# 概要:
#   - このスクリプトは、音声(.wav)から「ARKit風ブレンドシェイプ係数（例: jawOpen等）」の
#     時系列を推定（ダミー実装）し、その係数に応じた簡易レンダリング（2D）を行い、
#     連番フレームからMP4動画を生成します。
#   - オフライン/GUIなし環境で確実に動作する簡易パイプラインです。
#   - GPU使用の有無を選択可能（--device {auto,cpu,cuda}）。GPUが有効な場合は、ブレンドシェイプ
#     推定（信号処理）をPyTorch CUDA上で実行します（レンダリング自体はCPU/PIL/cv2）。
#
# 実行例:
#   $ python run_Audio2Face.py
#     → サンプル音声の自動探索 or 合成音声生成、outへPNG/MP4出力（デフォルトはGPU自動判定）
#
#   オプション例:
#   - GPUを使う（CUDA優先、なければCPUにフォールバック）:
#       python run_Audio2Face.py --device auto
#   - CPU固定:
#       python run_Audio2Face.py --device cpu
#   - CUDAを明示（未利用ならCPUへフォールバック）:
#       python run_Audio2Face.py --device cuda
#   - 入力/出力/処理長:
#       python run_Audio2Face.py --audio src/3D_avatars/Audio2Face-3D-Samples/example_audio/example.wav --fps 25 --seconds 5 --out ./output
#
# 主なオプション:
#   --audio PATH       入力wavのパス。未指定時は自動探索→合成音声にフォールバック
#   --out   DIR        出力ディレクトリ（デフォルト: ./output）
#   --fps   N          動画FPS（デフォルト: 25）
#   --seconds S        冒頭S秒のみ解析（0なら全体、デフォルト: 0）
#   --width W          フレーム幅（デフォルト: 640）
#   --height H         フレーム高さ（デフォルト: 640）
#   --seed N           乱数シード（スタイライズに使用、デフォルト: 42）
#   --device {auto,cpu,cuda}  計算デバイス選択（デフォルト: auto）
#   --use-nim          将来のNIM接続用フラグ（現時点はダミー動作）
#   --skip-video       動画化をスキップ（連番PNGのみ欲しい場合）
#
# 出力:
#   - {out}/blendshapes.json  : フレームごとのブレンドシェイプ係数
#   - {out}/blendshapes.csv   : 同上のCSV
#   - {out}/frames/frame_XXXX.png : レンダリング済みフレーム（画像）
#   - {out}/output.mp4        : MP4動画（可能であれば作成）
#
# 依存関係（自動インストールは行いません／オフラインでも動く工夫をしています）:
#   - あると嬉しい: numpy, pillow(PIL), opencv-python(cv2), imageio, imageio-ffmpeg
#                    soundfile または scipy（音声読み込み）
#   - GPU利用時: PyTorch（torch）+ CUDAが必要（--device cuda/autoでCUDA使用時）
#     優先度: (1) cv2 があれば描画＋動画化まで一気通貫で行います
#            (2) cv2が無い場合は PIL でPNG生成 → imageio(+imageio-ffmpeg)でmp4化
#            (3) どちらも無い場合は、PNG/MP4は生成できず JSON/CSVのみ出力
#
# GUIなしのヘッドレス環境について:
#   - 本スクリプトは2Dレンダリング（cv2またはPIL）でフレームを生成します。X/Waylandは不要です。
#   - フル3DのEGL/OSMesa構築は不要ですが、将来Unreal/Omniverseなどへ置換可能なように
#     ブレンドシェイプJSONを出力します。
#
# NVIDIA A2F-3D NIM（本番構成）についての要点（参考）:
#   - 本来はNGCからA2F-3D NIMコンテナを取得し、GPU上でマイクロサービスとして起動。
#   - gRPC/RESTで音声→ARKitブレンドシェイプの推定結果（感情付加含む）を取得します。
#   - 推奨GPU: Ampere世代以降（例: A100, A10, L4, RTX A6000, H100等）、CUDA 12.x/13.x系に整合。
#   - 今回は通信制限があるため、この手順は行っていません（ダミー推定で代替）。
#
# Macでの結果確認:
#   - 出力のMP4をホスト側で確認できます（compose.ymlで ./src:/app/src がマウント）。
#     例: ローカルで ./output/output.mp4 を再生してください。
#
# 注意:
#   - ブレンドシェイプ推定のGPU化は「信号処理部分（包絡抽出、平滑、補間）」で有効です。
#     レンダリング/エンコードはCPUで実施します。
#   - CUDA 12.6（例: NVIDIA A2 + Driver 560.35.05）等で、torchのCUDA対応ホイールと整合する必要があります。
# =============================================================================

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# 依存パッケージの遅延インポート
# ---------------------------
def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


np = _try_import("numpy")
cv2 = _try_import("cv2")
PIL = _try_import("PIL")
if PIL is not None:
    from PIL import Image, ImageDraw
imageio = _try_import("imageio")
imageio_ffmpeg = _try_import("imageio_ffmpeg")  # noqa: F401
torch = _try_import("torch")
if torch is not None:
    import torch.nn.functional as F  # type: ignore


# ---------------------------
# ユーティリティ
# ---------------------------
def log(msg: str):
    print(f"[run_Audio2Face] {msg}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def select_device(req: str) -> str:
    """
    req: 'auto'|'cpu'|'cuda'
    戻り: 実際に使用する 'cpu' or 'cuda'
    """
    r = (req or "auto").lower()
    if r == "cpu":
        return "cpu"
    if r == "cuda":
        if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
        log("CUDA指定ですが、torch.cuda.is_available()=False のため CPU へフォールバックします。")
        return "cpu"
    # auto
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------
# 音声のロード（オフライン前提）
# ---------------------------
def find_default_audio() -> Optional[Path]:
    candidates_root = Path("src/3D_avatars/Audio2Face-3D-Samples/example_audio")
    if candidates_root.exists():
        wavs = sorted(candidates_root.glob("*.wav"))
        if wavs:
            return wavs[0]
    return None


def load_wav_any(audio_path: Optional[Path], target_sr: int = 16000, seconds: float = 0.0) -> Tuple[List[float], int]:
    """
    可能なら soundfile/scipy を使う。
    無ければ wave(標準ライブラリ)でPCM WAVのみ読み込み。
    返り値は mono の float[-1~1] リストとサンプリングレート。
    """
    if audio_path is None or not audio_path.exists():
        log("入力音声が見つからないため、合成音声を内部生成します。")
        return synth_tone(duration=3.0 if seconds <= 0 else seconds, sr=target_sr), target_sr

    # 優先1: soundfile
    sf = _try_import("soundfile")
    if sf is not None:
        try:
            data, sr = sf.read(str(audio_path), always_2d=False)
            # モノラル化
            if len(getattr(data, "shape", [])) > 1:
                if np is not None:
                    data = data.mean(axis=1)
                else:
                    mono = []
                    for frame in data:
                        if isinstance(frame, (list, tuple)):
                            mono.append(sum(frame) / float(len(frame)))
                        else:
                            mono.append(float(frame))
                    data = mono
            if seconds > 0:
                n = int(sr * seconds)
                data = data[:n]
            data = list(map(lambda x: float(max(-1.0, min(1.0, x))), data))
            if sr != target_sr:
                data = resample_1d(data, sr, target_sr)
                sr = target_sr
            return data, sr
        except Exception as e:
            log(f"soundfileでの読み込みに失敗: {e}")

    # 優先2: scipy
    sp_io = _try_import("scipy.io")
    if sp_io is not None and hasattr(sp_io, "wavfile"):
        try:
            sr, data = sp_io.wavfile.read(str(audio_path))
            data = to_float_mono(data)
            if seconds > 0:
                n = int(sr * seconds)
                data = data[:n]
            if sr != target_sr:
                data = resample_1d(data, sr, target_sr)
                sr = target_sr
            return data, sr
        except Exception as e:
            log(f"scipy.io.wavfileでの読み込みに失敗: {e}")

    # フォールバック: wave(標準ライブラリ)。PCMのみ対応。
    import wave
    try:
        with wave.open(str(audio_path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            if seconds > 0:
                n_frames = min(n_frames, int(sr * seconds))
            raw = wf.readframes(n_frames)

        # 量子化ビット数で分岐
        import struct
        if sampwidth == 2:
            fmt = "<" + "h" * (len(raw) // 2)
            ints = struct.unpack(fmt, raw)
            frames = []
            if n_channels == 1:
                frames = ints
            else:
                for i in range(0, len(ints), n_channels):
                    s = 0
                    for c in range(n_channels):
                        s += ints[i + c]
                    frames.append(s / n_channels)
            data = [float(x) / 32768.0 for x in frames]
        elif sampwidth == 1:
            import array
            arr = array.array("B", raw)
            frames = []
            if n_channels == 1:
                frames = arr
            else:
                for i in range(0, len(arr), n_channels):
                    s = 0
                    for c in range(n_channels):
                        s += arr[i + c]
                    frames.append(s / n_channels)
            data = [((float(x) - 128.0) / 128.0) for x in frames]
        else:
            log(f"未対応のサンプル幅: {sampwidth} bytes")
            return synth_tone(duration=3.0 if seconds <= 0 else seconds, sr=target_sr), target_sr

        if sr != target_sr:
            data = resample_1d(data, sr, target_sr)
            sr = target_sr
        return data, sr

    except Exception as e:
        log(f"wave(標準ライブラリ)での読み込みに失敗: {e}")
        return synth_tone(duration=3.0 if seconds <= 0 else seconds, sr=target_sr), target_sr


def to_float_mono(data: Any) -> List[float]:
    if np is not None and isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            data = data.astype(np.float32) / max(abs(info.min), info.max)
        else:
            data = data.astype(np.float32)
        return data.tolist()
    else:
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
            mono = []
            for frame in data:
                mono.append(sum(frame) / float(len(frame)))
            data = mono
        out = []
        for x in data:
            if isinstance(x, int):
                out.append(float(x) / 32768.0)
            else:
                out.append(float(x))
        return out


def resample_1d(data: List[float], src_sr: int, dst_sr: int) -> List[float]:
    if src_sr == dst_sr:
        return data
    if np is not None:
        arr = np.asarray(data, dtype=np.float32)
        t_src = np.linspace(0, 1, len(arr), endpoint=False)
        t_dst = np.linspace(0, 1, int(len(arr) * dst_sr / src_sr), endpoint=False)
        out = np.interp(t_dst, t_src, arr).astype(np.float32)
        return out.tolist()
    out = []
    n_out = int(len(data) * dst_sr / src_sr)
    for i in range(n_out):
        pos = i * (len(data) - 1) / max(1, n_out - 1)
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, len(data) - 1)
        alpha = pos - i0
        out.append((1 - alpha) * data[i0] + alpha * data[i1])
    return out


def synth_tone(duration: float = 3.0, sr: int = 16000) -> List[float]:
    T = int(duration * sr)
    out = []
    for n in range(T):
        env = 0.5 * (1.0 + math.sin(2 * math.pi * 1.5 * n / sr))
        sig = env * math.sin(2 * math.pi * 220 * n / sr)
        out.append(sig)
    return out


# ---------------------------
# ダミー A2F 推定（CPU版 / GPU版）
# ---------------------------
def moving_average(x: List[float], win: int) -> List[float]:
    if win <= 1:
        return x[:]
    out = []
    s = 0.0
    q: List[float] = []
    for v in x:
        q.append(v)
        s += v
        if len(q) > win:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def audio_to_blendshapes_cpu(
    audio: List[float],
    sr: int,
    fps: int = 25,
    smooth_ms: float = 80.0,
) -> List[Dict[str, float]]:
    abs_sig = [abs(x) for x in audio]
    smooth_win = max(1, int((smooth_ms / 1000.0) * sr))
    env = moving_average(abs_sig, smooth_win)

    peak = max(1e-6, max(env))
    env = [min(1.0, v / peak) for v in env]

    duration = len(audio) / float(sr)
    n_frames = max(1, int(round(duration * fps)))
    env_frame = resample_1d(env, sr, fps)

    bs_series: List[Dict[str, float]] = []
    for i in range(n_frames):
        e = env_frame[min(i, len(env_frame) - 1)]
        jawOpen = clamp01(e)
        mouthFunnel = clamp01(math.sqrt(e))
        mouthPucker = clamp01((1.0 - e) * 0.4)
        cheekPuff = clamp01((e ** 0.3) * 0.3)
        blink = 1.0 if (i % int(max(5, fps))) == 0 and e < 0.2 else 0.0

        bs = {
            "jawOpen": jawOpen,
            "mouthFunnel": mouthFunnel,
            "mouthPucker": mouthPucker,
            "cheekPuff": cheekPuff,
            "eyeBlinkLeft": blink,
            "eyeBlinkRight": blink,
        }
        bs_series.append(bs)
    return bs_series


def audio_to_blendshapes_torch(
    audio: List[float],
    sr: int,
    fps: int = 25,
    smooth_ms: float = 80.0,
    device: str = "cuda",
) -> List[Dict[str, float]]:
    if torch is None:
        return audio_to_blendshapes_cpu(audio, sr, fps, smooth_ms)

    x = torch.tensor(audio, dtype=torch.float32, device=device).abs()  # |audio|
    win = max(1, int((smooth_ms / 1000.0) * sr))
    if win > 1:
        kernel = torch.ones(1, 1, win, device=device, dtype=torch.float32) / float(win)
        x_pad = F.pad(x.view(1, 1, -1), (win - 1, 0), mode="replicate")
        env = F.conv1d(x_pad, kernel).view(-1)
    else:
        env = x

    peak = torch.clamp(env.max(), min=1e-6)
    env = torch.clamp(env / peak, 0.0, 1.0)

    n_frames = max(1, int(round(len(audio) * fps / sr)))
    T = env.shape[0]
    pos = torch.linspace(0, T - 1, n_frames, device=device)
    i0 = pos.floor().long()
    i1 = torch.clamp(i0 + 1, max=T - 1)
    alpha = (pos - i0.float())
    e0 = env.index_select(0, i0)
    e1 = env.index_select(0, i1)
    env_frame = (1.0 - alpha) * e0 + alpha * e1  # (n_frames,)

    jawOpen = torch.clamp(env_frame, 0.0, 1.0)
    mouthFunnel = torch.clamp(torch.sqrt(env_frame), 0.0, 1.0)
    mouthPucker = torch.clamp((1.0 - env_frame) * 0.4, 0.0, 1.0)
    cheekPuff = torch.clamp(torch.pow(env_frame, 0.3) * 0.3, 0.0, 1.0)

    idx = torch.arange(n_frames, device=device)
    period = max(5, fps)
    blink_mask = ((idx % period) == 0) & (env_frame < 0.2)
    blink = blink_mask.float()

    vals = {
        "jawOpen": jawOpen,
        "mouthFunnel": mouthFunnel,
        "mouthPucker": mouthPucker,
        "cheekPuff": cheekPuff,
        "eyeBlinkLeft": blink,
        "eyeBlinkRight": blink,
    }
    vals_cpu = {k: v.detach().cpu().tolist() for k, v in vals.items()}

    bs_series: List[Dict[str, float]] = []
    for i in range(n_frames):
        bs_series.append({k: float(vals_cpu[k][i]) for k in vals_cpu})
    return bs_series


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ---------------------------
# レンダリング: cv2優先 → PILフォールバック（CPU）
# ---------------------------
def render_frame_cv2(bs: Dict[str, float], w: int, h: int):
    if np is None or cv2 is None:
        return None
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    center = (w // 2, h // 2)
    radius = min(w, h) // 3
    cv2.circle(canvas, center, radius, (220, 220, 220), thickness=-1)

    jaw = bs.get("jawOpen", 0.0)
    funnel = bs.get("mouthFunnel", 0.0)
    mx, my = center[0], center[1] + int(radius * 0.3)
    mouth_w = int(radius * (0.8 - 0.4 * funnel))
    mouth_h = int(max(2, radius * (0.1 + 0.35 * jaw)))

    cv2.ellipse(
        canvas,
        (mx, my),
        (mouth_w, mouth_h),
        0,
        0,
        360,
        (60, 60, 60),
        thickness=-1,
    )

    blink = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
    eye_h = int(max(2, radius * (0.10 * (1.0 - blink) + 0.02)))
    eye_w = int(radius * 0.18)
    ex_off = int(radius * 0.45)
    ey = center[1] - int(radius * 0.35)
    cv2.ellipse(canvas, (center[0] - ex_off, ey), (eye_w, eye_h), 0, 0, 360, (50, 50, 50), thickness=-1)
    cv2.ellipse(canvas, (center[0] + ex_off, ey), (eye_w, eye_h), 0, 0, 360, (50, 50, 50), thickness=-1)

    return canvas


def render_frame_pil(bs: Dict[str, float], w: int, h: int):
    if PIL is None:
        return None
    img = Image.new("RGB", (w, h), (255, 255, 255))
    drw = ImageDraw.Draw(img)

    cx, cy = w // 2, h // 2
    radius = min(w, h) // 3
    drw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(220, 220, 220))

    jaw = bs.get("jawOpen", 0.0)
    funnel = bs.get("mouthFunnel", 0.0)
    mx, my = cx, cy + int(radius * 0.3)
    mouth_w = int(radius * (0.8 - 0.4 * funnel))
    mouth_h = int(max(2, radius * (0.1 + 0.35 * jaw)))
    drw.ellipse((mx - mouth_w, my - mouth_h, mx + mouth_w, my + mouth_h), fill=(60, 60, 60))

    blink = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
    eye_h = int(max(2, radius * (0.10 * (1.0 - blink) + 0.02)))
    eye_w = int(radius * 0.18)
    ex_off = int(radius * 0.45)
    ey = cy - int(radius * 0.35)
    drw.ellipse((cx - ex_off - eye_w, ey - eye_h, cx - ex_off + eye_w, ey + eye_h), fill=(50, 50, 50))
    drw.ellipse((cx + ex_off - eye_w, ey - eye_h, cx + ex_off + eye_w, ey + eye_h), fill=(50, 50, 50))

    return img


def write_video_cv2(frames: List[Any], out_mp4: Path, fps: int) -> bool:
    if cv2 is None or np is None or len(frames) == 0:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not vw.isOpened():
        log("cv2.VideoWriter が開けませんでした。")
        return False
    ok_count = 0
    for fr in frames:
        if fr is None:
            continue
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h))
        vw.write(fr)
        ok_count += 1
    vw.release()
    log(f"cv2で {ok_count} フレームを書き込みました: {out_mp4}")
    return True


def write_video_imageio(frames: List[Any], out_mp4: Path, fps: int) -> bool:
    if imageio is None or len(frames) == 0 or np is None:
        return False
    try:
        with imageio.get_writer(str(out_mp4), fps=fps) as writer:
            for fr in frames:
                if fr is None:
                    continue
                if PIL is not None and isinstance(fr, Image.Image):
                    arr = np.asarray(fr)
                    writer.append_data(arr)
                elif isinstance(fr, np.ndarray):
                    if fr.ndim == 3 and fr.shape[2] == 3:
                        # BGR(cv2) -> RGB
                        writer.append_data(fr[:, :, ::-1])
                    else:
                        writer.append_data(fr)
        log(f"imageioで動画を書き出しました: {out_mp4}")
        return True
    except Exception as e:
        log(f"imageioでの動画書き出しに失敗: {e}")
        return False


# ---------------------------
# NIM接続用の将来拡張（スタブ）
# ---------------------------
class NimClientStub:
    """
    将来、NVIDIA A2F-3D NIMへgRPC/RESTで接続する場合はここを実装します。
    現在はダミー（ローカル推定）を返します。
    """
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port

    def infer_blendshapes(self, audio: List[float], sr: int, fps: int) -> List[Dict[str, float]]:
        # TODO: proto/ に基づいて実装。現状はローカルのダミー推定を返す。
        return audio_to_blendshapes_cpu(audio, sr, fps)


# ---------------------------
# メイン処理
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Audio2Face-like dummy pipeline (GPU selectable): WAV -> Blendshapes -> MP4")
    parser.add_argument("--audio", type=str, default="", help="入力wavのパス。未指定なら自動探索→合成音にフォールバック")
    parser.add_argument("--out", type=str, default="./output", help="出力ディレクトリ")
    parser.add_argument("--fps", type=int, default=25, help="動画FPS")
    parser.add_argument("--seconds", type=float, default=0.0, help="冒頭S秒のみ使用(0なら全体)")
    parser.add_argument("--width", type=int, default=640, help="フレーム幅")
    parser.add_argument("--height", type=int, default=640, help="フレーム高さ")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（将来拡張用）")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="計算デバイス選択（デフォルト: auto）")
    parser.add_argument("--use-nim", action="store_true", help="将来のNIM接続用（現状はダミーと同じ動作）")
    parser.add_argument("--skip-video", action="store_true", help="動画化をスキップ（PNGのみ出力）")
    args = parser.parse_args()

    device = select_device(args.device)
    if device == "cuda":
        cuda_ver = getattr(torch, "version", None)
        cuda_ver = getattr(cuda_ver, "cuda", "unknown") if cuda_ver is not None else "unknown"
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        log(f"計算デバイス: CUDA（GPU: {gpu_name}, torch CUDA={cuda_ver}）")
    else:
        log("計算デバイス: CPU")

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    ensure_dir(out_dir)
    ensure_dir(frames_dir)

    # 音声の決定
    audio_path = Path(args.audio) if args.audio else find_default_audio()
    if audio_path:
        log(f"入力音声: {audio_path}")
    else:
        log("入力音声: 合成音声（内部生成）")

    # 音声読み込み/生成
    audio, sr = load_wav_any(audio_path, target_sr=16000, seconds=args.seconds)
    duration_sec = len(audio) / float(sr)
    log(f"音声長: {duration_sec:.2f} sec, SR={sr}")

    # ブレンドシェイプ推定
    if args.use_nim:
        log("NIM接続フラグが指定されましたが、現在はダミー推定で置き換えます。")
        client = NimClientStub()
        bs_series = client.infer_blendshapes(audio, sr, args.fps)
    else:
        if device == "cuda":
            bs_series = audio_to_blendshapes_torch(audio, sr, args.fps, device=device)
        else:
            bs_series = audio_to_blendshapes_cpu(audio, sr, args.fps)

    # 保存（JSON/CSV）
    bs_json = out_dir / "blendshapes.json"
    with bs_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fps": args.fps,
                "num_frames": len(bs_series),
                "blendshapes": bs_series,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log(f"ブレンドシェイプを保存: {bs_json}")

    bs_csv = out_dir / "blendshapes.csv"
    save_blendshapes_csv(bs_series, bs_csv)
    log(f"ブレンドシェイプCSVを保存: {bs_csv}")

    # レンダリング（cv2優先 → PILフォールバック）
    frames_any: List[Any] = []
    w, h = int(args.width), int(args.height)
    t0 = time.time()
    for i, bs in enumerate(bs_series):
        img = None
        if cv2 is not None and np is not None:
            img = render_frame_cv2(bs, w, h)  # np.ndarray (BGR)
        if img is None and PIL is not None:
            img = render_frame_pil(bs, w, h)  # PIL.Image

        # ファイル保存（PNG）
        frame_path = frames_dir / f"frame_{i:04d}.png"
        if np is not None and cv2 is not None and isinstance(img, np.ndarray):
            cv2.imwrite(str(frame_path), img)  # BGRで保存
        elif PIL is not None and isinstance(img, Image.Image):
            img.save(str(frame_path))
        else:
            # 画像生成不可（cv2/PIL無し）→ PNGは出せない
            pass
        frames_any.append(img)
    dt = time.time() - t0
    log(f"フレーム生成: {len(frames_any)}枚, {dt:.2f}s")

    # 動画化
    out_mp4 = out_dir / "output.mp4"
    wrote = False
    if not args.skip_video:
        # 1) cv2（BGR配列）で試す
        if not wrote and cv2 is not None and np is not None:
            frames_bgr = []
            for fr in frames_any:
                if fr is None:
                    continue
                if isinstance(fr, np.ndarray):
                    frames_bgr.append(fr)
                elif PIL is not None and isinstance(fr, Image.Image):
                    arr = np.asarray(fr)  # RGB
                    frames_bgr.append(arr[:, :, ::-1].copy())  # to BGR
            if frames_bgr:
                wrote = write_video_cv2(frames_bgr, out_mp4, args.fps)

        # 2) imageio(+imageio-ffmpeg)（RGB想定）で試す
        if not wrote and imageio is not None and np is not None:
            frames_rgb = []
            for fr in frames_any:
                if fr is None:
                    continue
                if PIL is not None and isinstance(fr, Image.Image):
                    arr = np.asarray(fr)
                    frames_rgb.append(arr)
                elif isinstance(fr, np.ndarray):
                    frames_rgb.append(fr[:, :, ::-1])  # BGR->RGB
            if frames_rgb:
                wrote = write_video_imageio(frames_rgb, out_mp4, args.fps)

        if wrote:
            log(f"MP4を書き出しました: {out_mp4}")
        else:
            log("MP4の書き出しに失敗しました。cv2 または imageio(+imageio-ffmpeg) の導入をご検討ください。")

    # 完了メッセージ
    log("処理が終了しました。")
    log(f"出力先: {out_dir}")
    if out_mp4.exists():
        log(f"動画: {out_mp4}")
    else:
        log("動画が生成されていない場合は、PNGフレームやJSONを確認してください。")


def save_blendshapes_csv(bs_series: List[Dict[str, float]], csv_path: Path):
    keys = sorted({k for bs in bs_series for k in bs.keys()})
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame," + ",".join(keys) + "\n")
        for i, bs in enumerate(bs_series):
            row = [f"{i}"] + [f"{float(bs.get(k, 0.0)):.6f}" for k in keys]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    main()
