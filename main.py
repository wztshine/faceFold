"""将含有相同人脸的视频归入同一文件夹，支持断点续跑与增量处理。

处理流程：
1. 对每个视频抽帧，用 insightface 检测人脸并提取 512 维 embedding
2. 质量过滤：低置信度（det_score）/过小的人脸丢弃
3. 可选性别过滤（--gender）：只保留指定性别的人脸，max_faces/跳段都基于过滤后数量
4. 视频内聚合：同一人多帧向量按质量排序后聚簇、取平均，得到一个代表向量
5. 与画廊中各分组的"主体簇"比对（余弦相似度 + top-k 平均 + 可选 margin）
6. 相似度达到阈值 -> 归入已有分组；否则生成新分组
7. 每个分组达到 min_videos 个视频才创建文件夹并移动；不足的留在原地（记入 pending）
8. 断点续跑：已处理视频记入 progress.txt；画廊/缓存/待定文件按需保存

增量处理：新增视频放入输入目录即可，旧视频（已移动的）不在输入目录、未移动的走缓存快速跳过。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import onnxruntime
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import insightface
import numpy as np

from face_common import (
    dedup_against_gallery,
    dominant_vectors,
    find_group_folder,
    gid_of,
    load_gallery,
    load_person_names,
    person_folder_name,
    save_gallery,
    save_person_names,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".m4v", ".webm", ".mpg", ".mpeg"}

log = logging.getLogger("face_group")


def setup_logging(log_file: Path | None = None) -> None:
    """配置日志：控制台只显示 INFO 以上（干净），日志文件记录全部 DEBUG 细节。"""
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)
    log.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)
    log.setLevel(logging.DEBUG)
    log.propagate = False


def get_video_files(src_dir: Path) -> list[Path]:
    return sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def emb_cache_key(video_path: Path) -> str:
    """以 路径+大小+修改时间 生成唯一键，视频内容变了会自动失效。"""
    st = video_path.stat()
    raw = f"{video_path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_progress(progress_path: Path) -> set[str]:
    """读取已处理视频的唯一键集合（断点续跑）。"""
    if not progress_path.exists():
        return set()
    keys = set()
    try:
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.add(line)
    except Exception as e:
        log.warning("[进度] 读取失败，从空开始: %s", e)
    return keys


def save_progress(progress_path: Path, keys: set[str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with progress_path.open("w", encoding="utf-8") as f:
            for k in sorted(keys):
                f.write(k + "\n")
    except Exception as e:
        log.warning("[进度] 写入失败: %s", e)


def cached_extract_embeddings(
    app: insightface.app.FaceAnalysis,
    video_path: Path,
    frame_interval: int,
    cache_dir: Path | None,
    max_faces: int = 0,
    min_det_score: float = 0.5,
    min_face_size: int = 32,
    dedup_threshold: float = 0.5,
    no_face_window: int = 5,
    skip_seconds: float = 10.0,
    cluster_threshold: float = 0.5,
    gender_filter: str = "all",
) -> np.ndarray:
    """带缓存的特征提取：缓存命中直接读 npy，否则计算后写入缓存。
    缓存键包含特征提取参数，参数变化时自动失效。"""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = (
            f"{emb_cache_key(video_path)}_{min_det_score}_{min_face_size}_{dedup_threshold}"
            f"_{frame_interval}_{max_faces}_{no_face_window}_{skip_seconds}_{cluster_threshold}_{gender_filter}"
        )
        cache_file = cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embs = np.load(cache_file)
                log.debug("  [缓存] 命中 %s（%d 个向量）", video_path.name, len(embs))
                return embs
            except Exception as e:
                log.warning("[缓存] 读取失败 %s，重新计算: %s", cache_file.name, e)

    embs = extract_embeddings(
        app, video_path, frame_interval, max_faces, min_det_score, min_face_size,
        dedup_threshold, no_face_window, skip_seconds, cluster_threshold, gender_filter,
    )

    if cache_dir is not None and len(embs) > 0:
        try:
            np.save(cache_dir / f"{cache_key}.npy", embs)
        except Exception as e:
            log.warning("  [缓存] 写入失败: %s", e)
    return embs


def extract_embeddings(
    app: insightface.app.FaceAnalysis,
    video_path: Path,
    frame_interval: int,
    max_faces: int = 0,
    min_det_score: float = 0.5,
    min_face_size: int = 32,
    dedup_threshold: float = 0.5,
    no_face_window: int = 5,
    skip_seconds: float = 10.0,
    cluster_threshold: float = 0.5,
    gender_filter: str = "all",
) -> np.ndarray:
    """抽取视频中所有人脸的代表 embedding，返回形状 (N, 512) 的归一化向量。

    通过 seek 跳帧解码。max_faces > 0 时累计检出到该帧数提前停止。
    - min_det_score：过滤低置信度检测（模糊/误检）
    - min_face_size：过滤过小的人脸（bbox 短边小于该值），小脸 embedding 不可靠
    - no_face_window：连续多少次采样无人脸后，认为进入无人脸段并跳段
    - skip_seconds：无人脸段一次性向后跳过的秒数
    - cluster_threshold：视频主体簇提取阈值（过滤镜子/背景/次要角度的假向量）
    - gender_filter：性别过滤，all=不过滤 / female=只保留女性 / male=只保留男性
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("[跳过] 无法打开: %s", video_path.name)
        cap.release()
        return np.empty((0, 512), dtype=np.float32)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_step = max(1, int(round(fps * frame_interval)))

    # 按秒逐级降级采样间隔：若当前间隔下采样帧数不足以达到 max_faces，
    # 依次降为 1/2 间隔（如 2s -> 1s -> 0.5s ...），直到采样数达标或达到下限。
    # 例如 10 秒视频按 2 秒间隔只有 5 帧采样，会自动降到 1 秒取 10 帧。
    if max_faces > 0 and total_frames > 0:
        interval = frame_interval
        min_interval = 0.5  # 下限：最多每秒 2 帧，太密无意义
        while frame_interval > min_interval:
            step = max(1, int(round(fps * interval)))
            if total_frames // step >= max_faces:
                break
            interval = interval / 2.0
        frame_step = max(1, int(round(fps * interval)))
        if abs(interval - frame_interval) > 1e-9:
            log.debug("  [适配] %s 采样帧不足，采样间隔降为 %.2f 秒", video_path.name, interval)

    # 每个被保留的人脸记录 (embedding, det_score)
    raw: list[tuple[np.ndarray, float]] = []
    frame_failures = 0
    face_count = 0
    no_face_streak = 0
    last_progress = 0.0
    start_time = time.time()
    # 性别目标值：insightface 的 face.gender 为 0=女 / 1=男；all 不过滤
    target_gender = {"female": 0, "male": 1, "all": None}[gender_filter]

    # 目标采样帧位置（从 0 开始每隔 frame_step 帧）
    target = 0
    while True:
        if total_frames > 0 and target >= total_frames:
            break

        # 跳到目标帧并读取
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            # 读不到就再尝试向后挪几帧
            ok2, frame2 = cap.read()
            if not ok2:
                break
            frame = frame2

        try:
            faces = app.get(frame)
        except Exception:
            frame_failures += 1
            if frame_failures > 10:
                log.warning("[跳过] %s 连续推理失败，停止该视频", video_path.name)
                cap.release()
                return np.empty((0, 512), dtype=np.float32)
            target += frame_step
            continue

        n_accepted = 0
        for face in faces:
            # 质量过滤：低置信度 / 小脸直接丢弃（P1）
            det_score = float(getattr(face, "det_score", 0.0))
            if det_score < min_det_score:
                continue
            bbox = getattr(face, "bbox", None)
            if bbox is not None:
                w = float(bbox[2] - bbox[0])
                h = float(bbox[3] - bbox[1])
                if min(w, h) < min_face_size:
                    continue
            # 性别过滤：只保留目标性别的人脸，其他性别不参与计数（max_faces/跳段都基于过滤后数量）
            if target_gender is not None:
                if int(getattr(face, "gender", -1)) != target_gender:
                    continue
            raw.append((face.normed_embedding, det_score))
            n_accepted += 1
        face_count += n_accepted

        if max_faces > 0 and face_count >= max_faces:
            log.debug("  [早停] %s 已检出 %d 帧人脸（上限 %d），提前结束", video_path.name, face_count, max_faces)
            break

        # 连续无人脸段跳过：连续 no_face_window 次采样都没通过质量过滤的脸，
        # 认为进入无人脸段，一次性向后跳过 skip_seconds 秒再继续采样
        if n_accepted > 0:
            no_face_streak = 0
        else:
            no_face_streak += 1
            if no_face_streak >= no_face_window:
                jump = int(round(fps * skip_seconds))
                log.debug("  [跳段] %s 连续 %d 帧无人脸，跳过 %.0f 秒", video_path.name, no_face_streak, skip_seconds)
                no_face_streak = 0
                target += jump
                continue

        target += frame_step
        if total_frames > 0 and target / total_frames - last_progress >= 0.1:
            last_progress = target / total_frames
            elapsed = time.time() - start_time
            log.debug("  [进度] %s %.0f%% 采样帧 %d 检出人脸 %d 耗时 %.1fs",
                      video_path.name, target / total_frames * 100, target // frame_step, face_count, elapsed)

    cap.release()

    if frame_failures:
        log.warning("[提示] %s 有 %d 帧推理失败被跳过", video_path.name, frame_failures)
    if not raw:
        return np.empty((0, 512), dtype=np.float32)

    embs = np.asarray([r[0] for r in raw], dtype=np.float32)
    scores = np.asarray([r[1] for r in raw], dtype=np.float32)
    agg = aggregate_embeddings(embs, scores, dedup_threshold)
    # 视频主体簇提取：单人视频可能因镜子/背景/多角度拆出多个代表向量，
    # 只保留最主要的那个（出现最多、彼此相似的那组），过滤假向量后再去匹配画廊
    return dominant_vectors(agg, cluster_threshold)


def aggregate_embeddings(embs: np.ndarray, scores: np.ndarray, dedup_threshold: float = 0.5) -> np.ndarray:
    """把视频内同一人的多个向量聚合成一个代表向量。

    改进（P2/P3）：
    1. 按 det_score 从高到低排序，让高质量脸先成为簇种子（避免第一张模糊脸定代表）
    2. 贪心聚类：与已有簇代表相似度 > 阈值则归入该簇
    3. 每簇取向量平均并 L2 归一化，作为该人的代表（比单帧稳定）
    """
    if len(embs) == 0:
        return np.empty((0, 512), dtype=np.float32)

    # 按质量排序，先处理可靠的脸（P3）
    order = np.argsort(-scores)
    embs = embs[order]

    clusters: list[list[np.ndarray]] = []
    for emb in embs:
        if not clusters:
            clusters.append([emb])
            continue
        # 与所有簇的代表（质心）比较；质心须 L2 归一化，否则模长<1会系统性低估相似度
        best_idx, best_sim = -1, -1.0
        for ci, cl in enumerate(clusters):
            centroid = np.mean(cl, axis=0)
            cn = np.linalg.norm(centroid)
            if cn == 0:
                continue
            centroid = centroid / cn
            sim = float(centroid @ emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = ci
        if best_sim >= dedup_threshold:
            clusters[best_idx].append(emb)
        else:
            clusters.append([emb])

    reps = []
    for cl in clusters:
        centroid = np.mean(cl, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            reps.append(centroid / norm)
    return np.asarray(reps, dtype=np.float32)


def load_pending(pending_path: Path) -> dict[str, str]:
    """加载待移动视频映射 {绝对路径: gid}。

    格式：每行 "路径<TAB>gid"。旧格式（只有路径无 gid）兼容处理，gid 留空。
    """
    if not pending_path.exists():
        return {}
    result: dict[str, str] = {}
    try:
        with pending_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    p_str, gid = line.split("\t", 1)
                else:
                    p_str, gid = line, ""
                p = Path(p_str)
                if p.exists():
                    result[str(p.resolve())] = gid
    except Exception as e:
        log.warning("[待定] 读取失败: %s", e)
    return result


def save_pending(pending_path: Path, pending: dict[str, str]) -> None:
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pending_path.open("w", encoding="utf-8") as f:
            for p_str, gid in sorted(pending.items()):
                f.write(f"{p_str}\t{gid}\n")
    except Exception as e:
        log.warning("[待定] 写入失败: %s", e)


def match_video_to_gallery(
    video_embs: np.ndarray,
    dominants: dict[str, np.ndarray],
    threshold: float,
    video_match_ratio: float = 0.5,
    top_k: int = 3,
    margin: float = 0.0,
) -> tuple[str | None, float]:
    """在画廊中（用各组的主体簇）找到与视频最相似的组。

    只与每组的主体簇比较，避免被组内混入的次要人脸/背景人脸误导：
    要求视频中至少有 video_match_ratio 比例的向量能匹配上该组主体簇（相似度 >= threshold）。
    匹配打分用 top-k 人脸的平均相似度（比单点 max 更稳，P7），
    并支持 margin：第一候选须明显优于第二候选，否则不归组（减少边界误并）。
    返回 (组名, 最高平均相似度)。"""
    scored: list[tuple[str, float, float]] = []  # (gname, topk_score, best_sim)
    for gname, dom in dominants.items():
        if len(dom) == 0:
            continue
        sims = dom @ video_embs.T  # (主体簇向量, 视频向量)
        max_sim_per_video_vec = sims.max(axis=0)  # 每个视频向量与主体簇最像的距离
        matched_ratio = float((max_sim_per_video_vec >= threshold).mean())
        if matched_ratio < video_match_ratio:
            continue
        # top-k 均值：取与主体簇最像的若干个视频向量的平均相似度
        k = min(top_k, len(max_sim_per_video_vec))
        topk_score = float(np.sort(max_sim_per_video_vec)[-k:].mean())
        scored.append((gname, topk_score, float(max_sim_per_video_vec.max())))

    if not scored:
        return None, -1.0

    scored.sort(key=lambda x: -x[1])
    first, second = scored[0], (scored[1] if len(scored) > 1 else None)
    if second is not None and (first[1] - second[1]) < margin:
        # 第一候选与第二候选差距不足，属于边界情况，不归组
        return None, first[1]
    return first[0], first[1]


def next_group_index(gallery: dict[str, np.ndarray], output_dir: Path) -> int:
    """根据画廊和输出目录中的现有分组，计算下一个可用编号。
    兼容任意前缀命名（如 张三_001）和不定长数字。"""
    idxs: set[int] = set()
    for key in gallery:
        gid = gid_of(key)
        if gid is not None:
            idxs.add(int(gid))
    if output_dir.is_dir():
        for d in output_dir.iterdir():
            if d.is_dir():
                gid = gid_of(d.name)
                if gid is not None:
                    idxs.add(int(gid))
    return (max(idxs) + 1) if idxs else 0


def make_model(device: int) -> insightface.app.FaceAnalysis:
    # 只加载 detection + recognition + genderage：
    # - 跳过 landmark_3d_68 / landmark_2d_106（识别用检测的 5 点关键点对齐，不依赖它们）
    # - genderage 用于可选的按性别过滤（--gender）
    app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=[*onnxruntime.get_available_providers()],
        provider_options=[{} for _ in onnxruntime.get_available_providers()],
        allowed_modules=["detection", "recognition", "genderage"],
    )
    app.prepare(ctx_id=device, det_size=(640, 640))
    return app


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
            stream.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="将含相同人脸的视频归入同一文件夹（流式处理 + 断点续跑）")
    parser.add_argument("input_dir", type=Path, help="视频所在目录")
    parser.add_argument("--output_dir", type=Path, default=None, help="输出目录，默认 input_dir 下新建 groups")
    parser.add_argument("--frame_interval", type=float, default=2.0, help="隔多少秒抽一帧检测人脸，默认 2.0")
    parser.add_argument("--sim_threshold", type=float, default=0.5,
                        help="两张人脸相似度达到多少才算同一个人（越大越严格，默认 0.5）")
    parser.add_argument("--video_match_ratio", type=float, default=0.5,
                        help="一个视频要归入某组，至少多大比例的人脸要能匹配上（默认 0.5）")
    parser.add_argument("--cluster_threshold", type=float, default=0.5,
                        help="组内两张人脸相似度超过多少就算同一个人，用来找出每组最主要的人（默认 0.5）")
    parser.add_argument("--dedup_threshold", type=float, default=0.5,
                        help="视频内聚合阈值：同一视频里相似度超过该值视为同一人并取平均（默认 0.5）")
    parser.add_argument("--min_det_score", type=float, default=0.5,
                        help="人脸检测置信度下限，低于该值的模糊/误检脸丢弃（默认 0.5）")
    parser.add_argument("--min_face_size", type=int, default=32,
                        help="人脸 bbox 短边最小像素数，小于该值的小脸丢弃（默认 32）")
    parser.add_argument("--top_k", type=int, default=3,
                        help="匹配打分取最像的多少张人脸求平均（默认 3）")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="匹配时第一候选与第二候选的最小得分差，不足则不归组（默认 0=关闭）")
    parser.add_argument("--max_faces", type=int, default=15,
                        help="一个视频检出多少帧符合过滤条件的人脸就提前结束，0=不限（默认 15）")
    parser.add_argument("--gender", type=str, default="all", choices=["all", "male", "female"],
                        help="按性别过滤人脸：all=不过滤 / male=只保留男性 / female=只保留女性（默认 all）")
    parser.add_argument("--no_face_window", type=int, default=2,
                        help="连续多少次采样无人脸就判定进入无人脸段，然后跳过一个采样间隔（默认 2）")
    parser.add_argument("--skip_seconds", type=float, default=2.0,
                        help="无人脸段一次向后跳过的秒数，建议等于 frame_interval（默认 2）")
    parser.add_argument("--min_videos", type=int, default=2, help="一组至少几个视频才建文件夹并移动，不够先留在原地（默认 2）")
    parser.add_argument("--device", type=int, default=0, help="onnxruntime 设备 id，默认 0 (GPU)")
    parser.add_argument("--report", type=Path, default=None, help="分组报告 CSV 路径，默认 output_dir/group_report.csv")
    parser.add_argument("--log", type=Path, default=None, help="日志文件路径，默认不写文件")
    parser.add_argument("--cache_dir", type=Path, default=None, help="特征缓存目录，默认 output_dir/emb_cache")
    parser.add_argument("--gallery", type=Path, default=None,
                        help="画廊文件，默认 ~/.face_video_group/face_gallery.npz（跨目录/跨盘共享同一画廊）")
    parser.add_argument("--progress", type=Path, default=None, help="断点续跑进度文件，默认 output_dir/progress.txt")
    parser.add_argument("--pending", type=Path, default=None, help="待移动视频列表，默认 output_dir/pending.txt")
    parser.add_argument("--names", type=Path, default=None,
                        help="人物名映射文件，默认 ~/.face_video_group/person_names.json（不存在时静默）")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.input_dir / "groups")
    report_path = args.report or (output_dir / "group_report.csv")
    cache_dir = args.cache_dir or (output_dir / "emb_cache")
    default_gallery = Path.home() / ".face_video_group" / "face_gallery.npz"
    gallery_path = args.gallery or default_gallery
    names_path = args.names or (Path.home() / ".face_video_group" / "person_names.json")
    progress_path = args.progress or (output_dir / "progress.txt")
    pending_path = args.pending or (output_dir / "pending.txt")
    setup_logging(args.log)

    if not args.input_dir.is_dir():
        log.error("输入目录不存在: %s", args.input_dir)
        sys.exit(1)

    videos = get_video_files(args.input_dir)
    log.info("共发现 %d 个视频文件", len(videos))

    gallery = load_gallery(gallery_path)
    if gallery:
        log.info("已加载画廊：%d 个分组", len(gallery))
    else:
        log.info("无画廊，本次将生成初始分组")

    person_names = load_person_names(names_path)
    if person_names:
        log.info("已加载人物名映射：%d 个人物命名", len(person_names))

    progress_keys = load_progress(progress_path)
    if progress_keys:
        log.info("已加载进度：%d 个视频已处理", len(progress_keys))

    pending = load_pending(pending_path)
    if pending:
        log.info("已加载待移动：%d 个视频", len(pending))

    # 需要处理的视频 = 输入目录中的 + 待移动的（去重）
    all_videos: list[Path] = list(videos)
    seen = {str(v.resolve()) for v in videos}
    pending_paths: list[Path] = []
    for p_str in pending:
        p = Path(p_str)
        if str(p.resolve()) not in seen:
            all_videos.append(p)
            seen.add(str(p.resolve()))
        pending_paths.append(p)
    # pending 视频已有稳定 gid 的沿用（不重新匹配，避免自我匹配到上次建的组）；
    # gid 为空的是旧格式（只有路径无 gid），需要重新匹配
    pending_stable: dict[str, str] = {k: v for k, v in pending.items() if v}
    pending_rematch: set[str] = {str(p.resolve()) for p in pending_paths if pending.get(str(p.resolve())) == ""}

    log.info("加载 insightface 模型（buffalo_l）...")
    app = make_model(args.device)
    log.info("模型加载完成（%s）", ", ".join(onnxruntime.get_available_providers()))

    output_dir.mkdir(parents=True, exist_ok=True)

    # 方案 A：对齐重命名。有人名的分组文件夹，若名字与映射不符则重命名；
    # 无人名映射的保持原样。person_names.json 不存在时此列表为空，不报错。
    if output_dir.is_dir():
        for d in output_dir.iterdir():
            if not d.is_dir():
                continue
            gid = gid_of(d.name)
            if gid is None or gid not in person_names:
                continue
            target_name = person_folder_name(gid, person_names)
            if d.name != target_name:
                dest = d.parent / target_name
                if not dest.exists():
                    try:
                        d.rename(dest)
                        log.info("[重命名] %s -> %s", d.name, target_name)
                    except OSError as e:
                        log.warning("[重命名] 失败 %s -> %s: %s", d.name, target_name, e)

    rows = []
    grouped = 0
    untouched = 0
    skipped = 0

    # ---- 阶段 1：分组（只记录归属，不移动） ----
    video_assign: list[tuple[Path, str] | None] = [None] * len(all_videos)

    # 主体簇缓存：只重算被更新的组（其他组保持不变，避免每视频全量重算）
    dominants: dict[str, np.ndarray] = {
        g: dominant_vectors(v, args.cluster_threshold) for g, v in gallery.items()
    }

    for i, video in enumerate(all_videos):
        key = emb_cache_key(video)
        vres = str(video.resolve())
        # 已稳定分组的 pending 视频：沿用上次 gid，跳过重新匹配（避免自我匹配）
        if vres in pending_stable:
            gid = pending_stable[vres]
            video_assign[i] = (video, gid)
            log.info("[%d/%d] 沿用 %s -> group_%s（等待补移）", i + 1, len(all_videos), video.name, gid)
            continue
        # 非 pending 且命中 progress：已处理，跳过
        if key in progress_keys and vres not in pending_rematch:
            skipped += 1
            log.info("[%d/%d] 跳过 %s（已处理）", i + 1, len(all_videos), video.name)
            continue

        log.info("[%d/%d] 处理 %s", i + 1, len(all_videos), video.name)
        try:
            embs = cached_extract_embeddings(
                app, video, args.frame_interval, cache_dir, args.max_faces,
                args.min_det_score, args.min_face_size, args.dedup_threshold,
                args.no_face_window, args.skip_seconds, args.cluster_threshold, args.gender,
            )
        except Exception as e:
            log.warning("[跳过] %s 处理异常: %s: %s", video.name, type(e).__name__, e)
            log.info("[恢复] 重新加载模型后继续")
            try:
                app = make_model(args.device)
                embs = cached_extract_embeddings(
                    app, video, args.frame_interval, cache_dir, args.max_faces,
                    args.min_det_score, args.min_face_size, args.dedup_threshold,
                    args.no_face_window, args.skip_seconds, args.cluster_threshold, args.gender,
                )
            except Exception as e2:
                log.warning("[跳过] %s 重建后仍失败: %s: %s", video.name, type(e2).__name__, e2)
                embs = np.empty((0, 512), dtype=np.float32)

        # 无人脸：留在原地，但记录进度避免重复处理
        if len(embs) == 0:
            log.info("[%d/%d] 无人脸 %s，留在原地", i + 1, len(all_videos), video.name)
            rows.append({"video": video.name, "group": "", "moved_to": "", "note": "无人脸，未移动"})
            untouched += 1
            progress_keys.add(key)
            save_progress(progress_path, progress_keys)
            continue

        # 匹配画廊：只与各分组的主体簇比较，避免被污染向量误导（P2）
        gid, sim = match_video_to_gallery(
            embs, dominants, args.sim_threshold, args.video_match_ratio, args.top_k, args.margin
        )
        if gid is not None and sim >= args.sim_threshold:
            # 只并入与组主体簇达标的向量，防止次要人脸/背景人脸混入组（P1）
            group_vecs = gallery[gid]
            sims = dominants[gid] @ embs.T
            matched = embs[sims.max(axis=0) >= args.sim_threshold]
            # 剔除与组内已有向量高度相似的（防止 pending 视频跨轮次重复灌入）
            matched = dedup_against_gallery(matched, group_vecs)
            if len(matched) > 0:
                gallery[gid] = np.vstack([group_vecs, matched])
                # 该组向量更新了，重算它的主体簇
                dominants[gid] = dominant_vectors(gallery[gid], args.cluster_threshold)
            log.info("[%d/%d] 归组 %s -> group_%s（相似度 %.3f，并入 %d 个向量）",
                     i + 1, len(all_videos), video.name, gid, sim, len(matched))
        else:
            # 进入这里：要么无候选组（sim=-1），要么被 margin 拒绝
            if sim < 0:
                reason = "无匹配候选"
            elif sim < args.sim_threshold:
                reason = f"相似度 {sim:.3f} 低于阈值"
            else:
                reason = f"候选太接近（margin={args.margin}）"
            gid = str(next_group_index(gallery, output_dir))
            # 新组也只存该视频的主体簇，避免"出生即带污染"（背景人脸混入）
            gallery[gid] = dominant_vectors(embs, args.cluster_threshold)
            dominants[gid] = gallery[gid]
            log.info("[%d/%d] 新建 %s -> group_%s（%s）", i + 1, len(all_videos), video.name, gid, reason)

        video_assign[i] = (video, gid)
        save_gallery(gallery, gallery_path)

    # ---- 阶段 2：移动决策 ----
    # 统计每个 gid 本次分到的视频数（含沿用的 pending 视频）
    gid_count: Counter[str] = Counter()
    for assign in video_assign:
        if assign is not None:
            gid_count[assign[1]] += 1

    # 待移动的：本轮凑够 min_videos 就移动，否则留在原地并记录 gid
    remaining_pending: dict[str, str] = {}
    for assign in video_assign:
        if assign is None:
            continue
        video, gid = assign
        key = emb_cache_key(video)

        # 组文件夹已存在（历史批次已建立）：无条件移入，min_videos 只用于决定何时新建文件夹
        existing_dir = find_group_folder(output_dir, gid)
        if existing_dir is None and gid_count[gid] < args.min_videos:
            # 文件夹还不存在且该组视频数不足：留在原地，记入待移动列表（含 gid）
            remaining_pending[str(video.resolve())] = gid
            rows.append({"video": video.name, "group": "", "moved_to": "", "note": f"仅本组 {gid_count[gid]} 个视频，未移动"})
            log.info("[待定] %s -> group_%s（仅 %d 个视频，未达 min_videos=%d）",
                     video.name, gid, gid_count[gid], args.min_videos)
            untouched += 1
            continue

        # 移动视频：优先用已存在的（可能是重命名过的）文件夹，否则用 人名_gid 或默认 group_{gid}
        group_dir = existing_dir or (output_dir / person_folder_name(gid, person_names))
        group_dir.mkdir(parents=True, exist_ok=True)
        dest = group_dir / video.name
        if video.exists():
            shutil.move(str(video), str(dest))
        rows.append({"video": video.name, "group": group_dir.name, "moved_to": str(dest), "note": ""})
        log.info("[移动] %s -> %s/", video.name, group_dir.name)
        grouped += 1

        # 移动成功后补记进度，避免下次重复处理
        progress_keys.add(key)
        save_progress(progress_path, progress_keys)

    save_pending(pending_path, remaining_pending)
    if remaining_pending:
        log.info("另有 %d 个视频因分组视频数不足留在原地，已记入待移动列表", len(remaining_pending))

    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "group", "moved_to", "note"])
        writer.writeheader()
        writer.writerows(rows)

    save_gallery(gallery, gallery_path)
    log.info("完成：%d 个视频已移动，%d 个留在原地，%d 个已处理跳过。画廊 %d 个分组，报告：%s",
             grouped, untouched, skipped, len(gallery), report_path)


if __name__ == "__main__":
    main()
