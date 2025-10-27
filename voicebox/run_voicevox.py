#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOICEVOX ENGINE を Python のみで起動・利用して、テキストから wav を生成するスクリプト。

機能概要:
- src/3D_avatars/voicebox/voicevox_engine/run.py をサブプロセスとして起動（Docker 不要）
- エンジンの HTTP API (/audio_query, /synthesis) をたたいて wav を ./result に保存
- 既にエンジンが(指定ホスト・ポートで)起動していれば再起動せずにそのまま利用
- GPU 利用: --use-gpu を指定すればエンジンに --use_gpu を渡す（GPU 対応の VOICEVOX CORE が導入済みの場合に有効）
- コア未導入時は --enable-mock でモック合成（製品版の音声ではなく簡易音）にフォールバック可能

注意:
- OSS 版 ENGINE はコア(音声ライブラリ)が同梱されません。実音声で合成するには製品版 VOICEVOX または VOICEVOX CORE を導入し、
  --voicevox-dir または --voicelib-dir/--runtime-dir を指定してください。未導入の場合は --enable-mock で動かしてください。
- サンプリングレートはエンジン既定(24000Hz)です。

python run_voicevox.py --text "冬型の気圧配置緩む,初めは西高東低の冬型気圧配置だったが、高気圧が黄海から本州方面へ進み、全国的に天気回復。朝方冷えて鹿児島県枕崎で初霜、長崎県福江で初氷。"

"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def http_get_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.ok
    except requests.RequestException:
        return False


class VoiceVoxEngineRunner:
    def __init__(
        self,
        engine_dir: Path,
        host: str = "127.0.0.1",
        port: int = 50021,
        use_gpu: bool = False,
        enable_mock: bool = False,
        voicevox_dir: Optional[Path] = None,
        voicelib_dirs: Optional[list[Path]] = None,
        runtime_dirs: Optional[list[Path]] = None,
        output_log_utf8: bool = True,
        cpu_num_threads: Optional[int] = None,
    ) -> None:
        self.engine_dir = engine_dir
        self.host = host
        self.port = port
        self.use_gpu = use_gpu
        self.enable_mock = enable_mock
        self.voicevox_dir = voicevox_dir
        self.voicelib_dirs = voicelib_dirs or []
        self.runtime_dirs = runtime_dirs or []
        self.output_log_utf8 = output_log_utf8
        self.cpu_num_threads = cpu_num_threads

        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def already_running(self) -> bool:
        if not is_port_open(self.host, self.port):
            return False
        # /version が返れば VOICEVOX ENGINE が居る可能性が高い
        return http_get_ok(f"{self.base_url}/version")

    def start(self) -> None:
        if self.already_running():
            print(f"[info] VOICEVOX ENGINE is already running at {self.base_url}")
            return

        run_py = self.engine_dir / "run.py"
        if not run_py.is_file():
            raise FileNotFoundError(f"run.py not found: {run_py}")

        # 起動コマンドを構築
        cmd: list[str] = [
            sys.executable,
            str(run_py),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.use_gpu:
            cmd.append("--use_gpu")
        if self.enable_mock:
            cmd.append("--enable_mock")
        if self.voicevox_dir is not None:
            cmd.extend(["--voicevox_dir", str(self.voicevox_dir)])
        for p in self.voicelib_dirs:
            cmd.extend(["--voicelib_dir", str(p)])
        for p in self.runtime_dirs:
            cmd.extend(["--runtime_dir", str(p)])
        if self.cpu_num_threads is not None:
            cmd.extend(["--cpu_num_threads", str(self.cpu_num_threads)])
        if self.output_log_utf8:
            cmd.append("--output_log_utf8")

        env = os.environ.copy()
        if self.output_log_utf8:
            env["VV_OUTPUT_LOG_UTF8"] = "1"

        # エンジンは engine_dir をカレントにして起動するのが安全（パッケージ相対参照が解決される）
        print(f"[info] starting VOICEVOX ENGINE: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.engine_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # 終了時にクリーンアップ
        atexit.register(self.stop)

    def wait_ready(self, timeout: float = 60.0, ping_interval: float = 0.5) -> None:
        deadline = time.time() + timeout
        url = f"{self.base_url}/version"
        # すでに居るなら早期 return
        if self.already_running():
            print(f"[info] engine is ready at {url}")
            return

        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=1.0)
                if r.ok:
                    print(f"[info] engine is ready: {r.text}")
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            # プロセスが落ちていないかチェック
            if self._proc is not None and self._proc.poll() is not None:
                stdout = ""
                stderr = ""
                try:
                    stdout, stderr = self._proc.communicate(timeout=1)
                except Exception:
                    pass
                raise RuntimeError(
                    f"VOICEVOX ENGINE exited prematurely (code={self._proc.returncode}).\n"
                    f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
                )
            time.sleep(ping_interval)
        raise TimeoutError(f"VOICEVOX ENGINE not ready in {timeout} sec. last_err={last_err}")

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            return
        try:
            # 優雅に終了を試みる
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        finally:
            self._proc = None
            print("[info] engine process stopped")


def pick_speaker_id(base_url: str, preferred: Optional[int]) -> int:
    if preferred is not None:
        return int(preferred)
    # /speakers から最初のスタイルIDを１つ選ぶ
    url = f"{base_url}/speakers"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        # data は list[ { name, styles: [ { id, ... }, ... ] }, ... ]
        for sp in data:
            styles = sp.get("styles") or []
            for st in styles:
                sid = st.get("id")
                if isinstance(sid, int):
                    return sid
    except Exception:
        pass
    # 互換用に 1 を返す（環境によっては存在しない可能性もある）
    return 1


def synthesize_wav(
    base_url: str,
    text: str,
    out_path: Path,
    speaker_id: Optional[int] = None,
    speed_scale: Optional[float] = None,
    pitch_scale: Optional[float] = None,
    intonation_scale: Optional[float] = None,
    volume_scale: Optional[float] = None,
    pre_phoneme_length: Optional[float] = None,
    post_phoneme_length: Optional[float] = None,
    enable_katakana_english: Optional[bool] = None,
) -> Path:
    speaker = pick_speaker_id(base_url, speaker_id)

    # 1) audio_query
    q_params = {"speaker": speaker, "text": text}
    q_url = f"{base_url}/audio_query"
    r = requests.post(q_url, params=q_params, timeout=30)
    r.raise_for_status()
    query: dict[str, Any] = r.json()

    # 任意パラメータを上書き
    def set_if(name: str, value: Optional[Any]) -> None:
        if value is not None:
            query[name] = value

    set_if("speedScale", speed_scale)
    set_if("pitchScale", pitch_scale)
    set_if("intonationScale", intonation_scale)
    set_if("volumeScale", volume_scale)
    set_if("prePhonemeLength", pre_phoneme_length)
    set_if("postPhonemeLength", post_phoneme_length)
    if enable_katakana_english is not None:
        # audio_query のパラメータとしてはクエリ側よりURL引数が正式だが、
        # ここでは簡易にクエリ JSON にフラグを混ぜても ENGINE 側で無視されるため、
        # 必要なら audio_query 呼出しの params に持たせる実装へ拡張してください。
        query["enableKatakanaEnglish"] = bool(enable_katakana_english)

    # 2) synthesis
    s_url = f"{base_url}/synthesis"
    headers = {"Content-Type": "application/json"}
    r = requests.post(s_url, params={"speaker": speaker}, headers=headers, data=json.dumps(query), timeout=120)
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VOICEVOX ENGINE: テキストから wav を生成するユーティリティ")

    # 入出力
    p.add_argument("--text", type=str, required=True, help="読み上げるテキスト(UTF-8)")
    p.add_argument("--outfile", type=Path, default=None, help="保存先パス。未指定なら ./result/voicevox_YYYYmmdd_HHMMSS.wav")

    # エンジン接続
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=50021)
    p.add_argument("--timeout", type=float, default=60.0, help="エンジンの起動待ちタイムアウト(秒)")
    p.add_argument("--no-start", action="store_true", help="エンジンの新規起動を行わず、既存インスタンスに接続のみ行う")

    # エンジン起動オプション
    p.add_argument("--use-gpu", dest="use_gpu", action="store_true", help="GPU を利用 (--use_gpu を ENGINE に渡す)")
    p.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="GPU を利用しない")
    p.set_defaults(use_gpu=True)

    p.add_argument("--enable-mock", action="store_true", help="コア未導入でもモック合成で動かす")

    p.add_argument("--voicevox-dir", type=Path, default=None, help="製品版 VOICEVOX の ENGINE ディレクトリ")
    p.add_argument("--voicelib-dir", type=Path, action="append", default=None, help="VOICEVOX CORE のディレクトリ(複数可)")
    p.add_argument("--runtime-dir", type=Path, action="append", default=None, help="ONNXRuntime / libtorch 等のランタイムディレクトリ(複数可)")
    p.add_argument("--cpu-num-threads", type=int, default=None, help="CPU スレッド数(省略時は ENGINE の既定)")

    # 音声調整
    p.add_argument("--speaker", type=int, default=None, help="speaker(style_id)。未指定なら /speakers の先頭を選択")
    p.add_argument("--speed-scale", type=float, default=None)
    p.add_argument("--pitch-scale", type=float, default=None)
    p.add_argument("--intonation-scale", type=float, default=None)
    p.add_argument("--volume-scale", type=float, default=None)
    p.add_argument("--pre-phoneme-length", type=float, default=None)
    p.add_argument("--post-phoneme-length", type=float, default=None)
    p.add_argument("--enable-katakana-english", action="store_true", default=None)

    return p.parse_args()


def default_outfile() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "result" / f"voicevox_{ts}.wav"


def main() -> None:
    args = parse_args()

    # エンジンのディレクトリ（このファイルと同階層の voicevox_engine）
    engine_dir = Path(__file__).resolve().parent / "voicevox_engine"
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"voicevox_engine directory not found: {engine_dir}")

    runner = VoiceVoxEngineRunner(
        engine_dir=engine_dir,
        host=args.host,
        port=args.port,
        use_gpu=bool(args.use_gpu),
        enable_mock=bool(args.enable_mock),
        voicevox_dir=args.voicevox_dir,
        voicelib_dirs=list(args.voicelib_dir) if args.voicelib_dir else [],
        runtime_dirs=list(args.runtime_dir) if args.runtime_dir else [],
        output_log_utf8=True,
        cpu_num_threads=args.cpu_num_threads,
    )

    # コアのパス指定が無い場合は最初からモックで起動してエラーを抑止
    core_specified = (args.voicevox_dir is not None) or (args.voicelib_dir is not None and len(args.voicelib_dir) > 0)
    if not core_specified and not runner.enable_mock:
        print("[info] VOICEVOX CORE のパス指定が無いため、モック(--enable_mock)で起動します。"
              "実音声で合成するには --voicevox-dir または --voicelib-dir/--runtime-dir を指定してください。")
        runner.enable_mock = True
        runner.use_gpu = False

    # 既存接続 or 起動
    if not runner.already_running():
        if args.no_start:
            raise RuntimeError(
                f"VOICEVOX ENGINE not running at {runner.base_url}. "
                "Use without --no-start or start the engine manually."
            )
        # GPU で起動 → ダメなら CPU/モックにフォールバックする戦略
        try:
            runner.start()
            runner.wait_ready(timeout=args.timeout)
        except Exception as e_gpu:
            if args.use_gpu:
                if "コアが見つかりません" in str(e_gpu) or "core" in str(e_gpu).lower():
                    print("[error] VOICEVOX CORE が見つかりません。--voicevox-dir または --voicelib-dir/--runtime-dir を指定してください。"
                          "未指定の場合はモック(--enable_mock)での起動を推奨します。")
                print(f"[warn] failed to start engine with GPU: {e_gpu}")
                print("[warn] retrying without --use_gpu ...")
                # GPU を切って再試行
                runner.stop()
                runner.use_gpu = False
                try:
                    runner.start()
                    runner.wait_ready(timeout=args.timeout)
                except Exception as e_cpu:
                    print(f"[warn] failed to start engine on CPU: {e_cpu}")
                    print("[warn] retrying with --enable_mock ...")
                    runner.stop()
                    runner.enable_mock = True
                    try:
                        runner.start()
                        runner.wait_ready(timeout=args.timeout)
                    except Exception as e_mock:
                        runner.stop()
                        raise RuntimeError(
                            f"failed to start engine: CPU and mock fallback also failed: {e_mock}"
                        ) from e_mock
            else:
                # CPU 指定時に失敗した場合 → モックへフォールバックを試行
                print(f"[warn] failed to start engine: {e_gpu}")
                if not runner.enable_mock:
                    print("[warn] retrying with --enable_mock ...")
                    runner.stop()
                    runner.enable_mock = True
                    try:
                        runner.start()
                        runner.wait_ready(timeout=args.timeout)
                    except Exception as e_mock_only:
                        runner.stop()
                        raise
                else:
                    raise
    else:
        print(f"[info] reuse running engine at {runner.base_url}")

    # 出力パス
    out_path = args.outfile if args.outfile is not None else default_outfile()

    # 合成
    out_file = synthesize_wav(
        base_url=runner.base_url,
        text=args.text,
        out_path=out_path,
        speaker_id=args.speaker,
        speed_scale=args.speed_scale,
        pitch_scale=args.pitch_scale,
        intonation_scale=args.intonation_scale,
        volume_scale=args.volume_scale,
        pre_phoneme_length=args.pre_phoneme_length,
        post_phoneme_length=args.post_phoneme_length,
        enable_katakana_english=args.enable_katakana_english,
    )
    print(f"[done] saved: {out_file}")

    # 自動起動した場合はプロセスを残さず終了（既存接続のみなら stop() は何もしない）
    runner.stop()


if __name__ == "__main__":
    main()
