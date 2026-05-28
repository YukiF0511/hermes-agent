"""
動画シーケンス — セグメント計画・メディアユーティリティ
====================================================

長尺動画の連続生成を行うコアモジュール。

提供機能:
  - VideoSequenceRequest / VideoSegmentPlan データクラス
  - plan_segments() — プロバイダーの上限を尊重しながら合計尺をセグメント分割
  - materialize_video() — 動画をローカルシーケンスキャッシュにダウンロード/コピー
  - extract_last_frame() — ffmpeg で動画の最終フレームを取得
  - image_file_to_data_url() — 画像ファイルを base64 データ URL にエンコード
  - concat_videos() — ffmpeg の再エンコードで mp4 を結合

キャッシュ配置: $HERMES_HOME/cache/videos/sequences/<sequence_id>/
"""

from __future__ import annotations

import base64
import math
import mimetypes
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _sequence_cache_dir(sequence_id: str) -> Path:
    """$HERMES_HOME/cache/videos/sequences/<sequence_id>/ を返す（親ディレクトリも作成する）。"""
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "videos" / "sequences" / sequence_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VideoSegmentPlan:
    """シーケンス内の単一の計画済み動画セグメント。"""

    index: int
    start_seconds: float
    duration: int  # 秒数。プロバイダーの [min_duration, max_duration] にクランプ済み
    prompt_hint: str = ""


@dataclass
class VideoSequenceRequest:
    """複数セグメントから構成される長尺動画の最上位リクエスト。"""

    total_duration: int  # 目標合計秒数
    segment_duration: int  # セグメントあたりの希望秒数
    prompt: str
    sequence_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    provider_name: str = ""

    # plan_segments() によって設定される
    segments: List[VideoSegmentPlan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Segment planning
# ---------------------------------------------------------------------------


def plan_segments(
    total_duration: int,
    segment_duration: int,
    provider_caps: Dict[str, Any],
) -> List[VideoSegmentPlan]:
    """*total_duration* をプロバイダーの上限を尊重しながらセグメントに分割する。

    Args:
        total_duration: 目標の動画合計尺（秒）。
        segment_duration: セグメントあたりの希望尺（秒）。
        provider_caps: ``VideoGenProvider.capabilities()`` が返す辞書。
            ``min_duration``（デフォルト 1）と ``max_duration``（デフォルト 10）を参照する。

    Returns:
        全尺をカバーする :class:`VideoSegmentPlan` の順序付きリスト。
        合計が割り切れない場合、最後のセグメントは ``segment_duration`` より短くなることがあるが、
        ``min_duration`` を下回ることはない。
    """
    min_dur: int = int(provider_caps.get("min_duration", 1))
    max_dur: int = int(provider_caps.get("max_duration", 10))

    # 希望セグメント尺をプロバイダーの上限にクランプする。
    clamped = max(min_dur, min(segment_duration, max_dur))

    n_segments = math.ceil(total_duration / clamped)
    plans: List[VideoSegmentPlan] = []

    remaining = total_duration
    for i in range(n_segments):
        dur = min(clamped, remaining)
        # プロバイダーの最小値を下回るセグメントは生成しない。
        if dur < min_dur:
            dur = min_dur
        plans.append(
            VideoSegmentPlan(
                index=i,
                start_seconds=i * clamped,
                duration=dur,
            )
        )
        remaining -= dur
        if remaining <= 0:
            break

    return plans


# ---------------------------------------------------------------------------
# Media utilities
# ---------------------------------------------------------------------------


def materialize_video(
    url_or_path: str,
    *,
    sequence_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """動画をローカルシーケンスキャッシュにダウンロードまたはコピーする。

    Args:
        url_or_path: HTTP(S) URL またはファイルシステムの絶対/相対パス。
        sequence_id: キャッシュのサブディレクトリキー。省略時はランダムな UUID が生成される。
        filename: 保存するファイル名の上書き指定。省略時はソースから導出される。

    Returns:
        キャッシュ済み動画ファイルへの絶対 :class:`Path`。
    """
    sid = sequence_id or uuid.uuid4().hex
    cache_dir = _sequence_cache_dir(sid)

    if url_or_path.startswith(("http://", "https://")):
        if not filename:
            url_path = url_or_path.split("?")[0].rstrip("/")
            filename = url_path.split("/")[-1] or f"video_{uuid.uuid4().hex[:8]}.mp4"
        dest = cache_dir / filename
        urllib.request.urlretrieve(url_or_path, dest)
    else:
        src = Path(url_or_path)
        dest = cache_dir / (filename or src.name)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    return dest


def extract_last_frame(video_path: Path) -> Path:
    """ffmpeg を使って *video_path* の最終フレームを PNG として抽出する。

    出力はソース動画と同じディレクトリに ``<stem>_last_frame.png`` という名前で書き出される。

    Raises:
        FileNotFoundError: ffmpeg が PATH 上に見つからない場合。
        subprocess.CalledProcessError: ffmpeg がゼロ以外のステータスで終了した場合。
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH — install it to use extract_last_frame()"
        )

    video_path = Path(video_path)
    out_path = video_path.parent / f"{video_path.stem}_last_frame.png"

    # 終端近くにシークできるようにストリームの再生時間をプローブする。
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    seek_to: Optional[float] = None
    for line in probe_result.stdout.splitlines():
        line = line.strip()
        try:
            dur = float(line)
            seek_to = max(0.0, dur - 0.1)
            break
        except ValueError:
            continue

    if seek_to is not None:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_to),
            "-i", str(video_path),
            "-vframes", "1",
            str(out_path),
        ]
    else:
        # 再生時間のプローブに失敗した場合のフォールバック: sseof で最終フレームを取得する。
        cmd = [
            "ffmpeg", "-y",
            "-sseof", "-0.1",
            "-i", str(video_path),
            "-vframes", "1",
            "-update", "1",
            str(out_path),
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def image_file_to_data_url(path: Path) -> str:
    """画像ファイルを読み込み、base64 データ URL として返す。

    Args:
        path: 画像ファイルのパス（PNG、JPEG、WebP など）。

    Returns:
        ``data:<mime>;base64,<encoded>`` 形式の文字列。
    """
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"  # ffmpeg で抽出したフレームのデフォルト値として安全
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def concat_videos(
    paths: List[Path],
    output_path: Path,
) -> Path:
    """mp4 ファイルのリストを ffmpeg の再エンコードで一つの出力ファイルに結合する。

    一時ファイルリストを使った ``concat`` デマルチプレクサを使用し、
    シーク可能なクリーンな出力を得るために H.264/AAC で再エンコードする。

    Args:
        paths: ソース動画パスの順序付きリスト。
        output_path: 結合後の mp4 の出力先パス。

    Returns:
        出力ファイルへの絶対 :class:`Path`。

    Raises:
        FileNotFoundError: ffmpeg が PATH 上に見つからない場合。
        ValueError: *paths* が空の場合。
        subprocess.CalledProcessError: ffmpeg がゼロ以外のステータスで終了した場合。
    """
    if not paths:
        raise ValueError("concat_videos() requires at least one input path")

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH — install it to use concat_videos()"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as flist:
        for p in paths:
            safe = str(Path(p).resolve()).replace("'", "'\\''")
            flist.write(f"file '{safe}'\n")
        flist_path = flist.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", flist_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        Path(flist_path).unlink(missing_ok=True)

    return output_path.resolve()
