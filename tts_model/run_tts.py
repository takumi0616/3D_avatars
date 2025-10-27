# -*- coding: utf-8 -*-
"""
日本語TTSの統括実行スクリプト。

- Parler-TTS large-bate / Suno Bark / SpeechT5 を簡単に実行
- GPU指定（0/1/...）、乱数シード、保存先、各モデル固有設定をCLIで指定可能
- 既定では実行ファイル（本スクリプト）から見て ./result にモデルごとの wav を保存
- 各モデルの実装は同ディレクトリの japanese_parler_tts_large_bate.py / suno_bark.py / japanese_speecht5_tts.py を使用
  ・いずれも Hugging Face の snapshot_download でローカル保存（存在時は再DLしない）

使用例:
nohup python run_tts.py --text "きわめ、きわめ、きわめ、きわめ、がんばれ！" --gpu 0 > run_tts.log 2>&1 &

モデルを個別指定して実行:
python run_tts.py --text "こんにちは" --models parler
python run_tts.py --text "こんにちは" --models bark
python run_tts.py --text "こんにちは" --models speecht5

保存先ディレクトリ、話者/プロンプトなどの上書き:
python run_tts.py --text "きわめ、きわめ、きわめ、きわめ、がんばれ！" --out-dir ./result \
  --parler-prompt "落ち着いた成人女性、ゆっくり、丁寧な日本語" \
  --bark-voice-preset "v2/ja_speaker_1"

注意:
- 実行には GPU 環境（CUDA 12.4 + torch 2.6.*）を想定。GPUが使えるものは基本GPU 0番を利用。
- 依存: transformers>=4.46.0, huggingface_hub, encodec, soundfile, numpy, scipy(任意)
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import List, Optional

# 実装モジュール（同ディレクトリ）を確実に import 可能にする
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# モデル別ラッパ
import japanese_parler_tts_large_bate as parler_mod  # noqa: E402
import suno_bark as bark_mod  # noqa: E402
import japanese_speecht5_tts as speecht5_mod  # noqa: E402


def _default_results_dir() -> str:
    """
    実行ファイル（本スクリプト）から見て ./result を既定にする。
    """
    return os.path.abspath(os.path.join(_THIS_DIR, "result"))


def _resolve_out_dir(arg: Optional[str]) -> str:
    """
    --out-dir の解決:
    - None: 既定の ./result（スクリプト相対）
    - 絶対パス: そのまま使用
    - 相対パス: スクリプト相対に解決
    """
    if arg is None:
        return _default_results_dir()
    if os.path.isabs(arg):
        return arg
    return os.path.abspath(os.path.join(_THIS_DIR, arg))

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Japanese TTS runner (Parler-TTS / Bark)")
    parser.add_argument("--text", type=str, required=True, help="喋らせたい日本語テキスト")
    parser.add_argument("--gpu", type=int, default=0, help="使用GPU index（既定: 0）")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="実行するモデル（all|parler|bark|カンマ区切りで複数指定: 例 'parler,bark'）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="出力ディレクトリ（スクリプト相対可。既定: ./result）",
    )
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（再現性が必要なとき）")
    parser.add_argument("--sr", type=int, default=None, help="保存時サンプリングレート（未指定はモデル出力）")
    parser.add_argument("--prefix", type=str, default=None, help="生成ファイル名の先頭プレフィックス（任意）")

    # Parler固有
    parser.add_argument("--parler-model-id", type=str, default=None, help="Parler の HF モデルID（未指定は内部既定）")
    parser.add_argument("--parler-local-dir", type=str, default=None, help="Parler のローカル保存先（未指定は ./models/...）")
    parser.add_argument("--parler-prompt", type=str, default=None, help="Parler の日本語スタイル・プロンプト")

    # Bark固有
    parser.add_argument("--bark-model-id", type=str, default=None, help="Bark の HF モデルID（未指定は内部既定）")
    parser.add_argument("--bark-local-dir", type=str, default=None, help="Bark のローカル保存先（未指定は ./models/...）")
    parser.add_argument("--bark-voice-preset", type=str, default=None, help="Bark の話者プリセット（例: v2/ja_speaker_1）")

    # SpeechT5 固有
    parser.add_argument("--speecht5-model-id", type=str, default=None, help="SpeechT5 の HF モデルID（未指定は内部既定）")
    parser.add_argument("--speecht5-local-dir", type=str, default=None, help="SpeechT5 のローカル保存先（未指定は ./models/...）")
    parser.add_argument("--speecht5-vocoder-id", type=str, default=None, help="SpeechT5 の Vocoder モデルID（未指定は内部既定）")
    parser.add_argument("--speecht5-vocoder-local-dir", type=str, default=None, help="SpeechT5 Vocoder のローカル保存先（未指定は ./models/...）")
    parser.add_argument("--speecht5-speaker-index", type=int, default=None, help="SpeechT5 既定話者の xvector index（cmu-arctic-xvectors）")
    parser.add_argument("--speecht5-speaker-split", type=str, default=None, help="xvector 取得 split（train/validation/test など）")
    parser.add_argument("--speecht5-xvector-npy", type=str, default=None, help="独自 xvector(.npy; shape=(512,) or (1,512)) を使用する場合のパス")

    return parser.parse_args()


def _select_models(spec: str) -> List[str]:
    spec = (spec or "").strip().lower()
    if spec in ("", "all", "both"):
        return ["parler", "bark", "speecht5"]
    # カンマ区切り
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    valid = []
    for p in parts:
        if p in ("parler", "bark", "speecht5"):
            valid.append(p)
    return valid or ["parler", "bark"]


def main() -> None:
    args = parse_args()

    out_dir = _resolve_out_dir(args.out_dir)
    _ensure_dir(out_dir)
    ts = _timestamp()

    # モデル選択
    models = _select_models(args.models)

    saved_paths: List[str] = []

    if "parler" in models:
        # ファイル名: {prefix_}parler_{ts}.wav
        fname = f"parler_{ts}.wav" if not args.prefix else f"{args.prefix}_parler_{ts}.wav"
        out_path = os.path.join(out_dir, fname)
        try:
            path = parler_mod.synthesize_to_wav(
                text=args.text,
                out_wav_path=out_path,
                gpu=args.gpu,
                model_id=args.parler_model_id,      # None -> モジュール既定
                local_dir=args.parler_local_dir,    # None -> ./models/...
                prompt=args.parler_prompt,          # None -> 日本語既定
                seed=args.seed,
                sample_rate=args.sr,
            )
            print(f"[run_tts] Parler saved: {path}")
            saved_paths.append(path)
        except Exception as e:
            # Parler は Transformers のバージョン互換で失敗することがあるため、Bark を継続実行可能にする
            print(f"[run_tts] Parler 失敗のためスキップ: {e}")

    if "bark" in models:
        fname = f"bark_{ts}.wav" if not args.prefix else f"{args.prefix}_bark_{ts}.wav"
        out_path = os.path.join(out_dir, fname)
        try:
            path = bark_mod.synthesize_to_wav(
                text=args.text,
                out_wav_path=out_path,
                gpu=args.gpu,
                model_id=args.bark_model_id,        # None -> モジュール既定
                local_dir=args.bark_local_dir,      # None -> ./models/...
                voice_preset=args.bark_voice_preset,  # None -> 日本語既定
                seed=args.seed,
                sample_rate=args.sr,
            )
            print(f"[run_tts] Bark saved: {path}")
            saved_paths.append(path)
        except Exception as e:
            print(f"[run_tts] Bark 失敗のためスキップ: {e}")

    if "speecht5" in models:
        fname = f"speecht5_{ts}.wav" if not args.prefix else f"{args.prefix}_speecht5_{ts}.wav"
        out_path = os.path.join(out_dir, fname)
        try:
            path = speecht5_mod.synthesize_to_wav(
                text=args.text,
                out_wav_path=out_path,
                gpu=args.gpu,
                model_id=args.speecht5_model_id,             # None -> モジュール既定
                local_dir=args.speecht5_local_dir,           # None -> ./models/...
                vocoder_id=args.speecht5_vocoder_id,         # None -> モジュール既定
                vocoder_local_dir=args.speecht5_vocoder_local_dir,  # None -> ./models/...
                speaker_index=(args.speecht5_speaker_index if args.speecht5_speaker_index is not None else speecht5_mod.DEFAULT_SPEAKER_INDEX),
                speaker_split=(args.speecht5_speaker_split if args.speecht5_speaker_split is not None else speecht5_mod.DEFAULT_SPEAKER_SPLIT),
                xvector_npy=args.speecht5_xvector_npy,
                seed=args.seed,
                sample_rate=args.sr,
            )
            print(f"[run_tts] SpeechT5 saved: {path}")
            saved_paths.append(path)
        except Exception as e:
            print(f"[run_tts] SpeechT5 失敗のためスキップ: {e}")

    # まとめ出力
    print("===== Completed TTS generation =====")
    if not saved_paths:
        print("[run_tts] 生成に失敗しました。Transformers のバージョンやモデルID、GPU利用可否をご確認ください。")
        print("[run_tts] Parler 側で失敗する場合は、--models bark でBarkのみ実行、または transformers を新しめに更新してください。")
    else:
        for p in saved_paths:
            print(p)


if __name__ == "__main__":
    main()
