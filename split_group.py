"""分组后处理工具：拆分被误并的分组（一个文件夹里其实是两个人）。

与 merge_groups 相反。画廊向量是逐视频累积、无法按视频切开的，因此本工具
重新提取该组每个视频的人脸向量（复用 emb_cache），做视频级聚类，确认后：
1. 最大的簇保留原分组文件夹；其余簇各新建分组文件夹并移动视频
2. 按拆分后的归属**重建**画廊向量（原 gid 只保留其簇的向量，去掉污染）

用法:
    uv run python split_group.py "D:/videos/groups" --gid 12
    uv run python split_group.py "D:/videos/groups" --gid 12 --dry_run
    uv run python split_group.py "D:/videos/groups" --gid 12 --sim_threshold 0.5
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

from face_common import (
    default_gallery_path,
    default_names_path,
    find_group_folder,
    load_gallery,
    load_person_names,
    person_folder_name,
    save_gallery,
)
from main import (
    VIDEO_SUFFIXES,
    cached_extract_embeddings,
    make_model,
    next_group_index,
    redirect_stderr_to_devnull,
    setup_logging,
)

log = logging.getLogger("split_group")


def cluster_videos(reps: list[np.ndarray], sim_threshold: float) -> list[list[int]]:
    """视频级连接聚类：两个视频任一对代表向量相似度 >= 阈值即同簇。

    返回簇（视频下标列表），按成员数降序。无人脸视频（空代表向量）单独成簇。
    """
    n = len(reps)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if len(reps[i]) == 0:
            continue
        for j in range(i + 1, n):
            if len(reps[j]) == 0:
                continue
            if float((reps[i] @ reps[j].T).max()) >= sim_threshold:
                parent[find(i)] = find(j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=len, reverse=True)


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("请输入 y 或 n")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
            stream.reconfigure(line_buffering=True)
    setup_logging()
    # setup_logging 只配置了 face_group/face_common 两个 logger；
    # 本工具用自己的 split_group logger，必须挂上 handler，否则 INFO 全部不可见
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    redirect_stderr_to_devnull()  # 与 main.py 一致：静音 ffmpeg 解码警告

    parser = argparse.ArgumentParser(description="拆分被误并的分组（一个文件夹里其实是两个人）")
    parser.add_argument("output_dir", type=Path, help="groups 输出目录")
    parser.add_argument("--gid", required=True, help="要拆分的分组 id（数字）")
    parser.add_argument("--gallery", type=Path, default=None,
                        help="画廊文件，默认 ~/.facefold/face_gallery.npz（旧版 ~/.face_video_group 自动兼容，与 main.py 一致）")
    parser.add_argument("--sim_threshold", type=float, default=0.5,
                        help="两个视频算同一个人所需的最大相似度（默认 0.5）")
    parser.add_argument("--cluster_threshold", type=float, default=0.5,
                        help="视频内聚簇阈值（默认 0.5）")
    parser.add_argument("--dedup_threshold", type=float, default=0.5,
                        help="视频内聚合阈值（默认 0.5）")
    parser.add_argument("--min_det_score", type=float, default=0.5, help="检测置信度下限（默认 0.5）")
    parser.add_argument("--min_face_size", type=int, default=32, help="人脸 bbox 短边下限（默认 32）")
    parser.add_argument("--max_faces", type=int, default=15, help="每视频最大采样人脸数（默认 15）")
    parser.add_argument("--gender", type=str, default="female", choices=["all", "male", "female"],
                        help="按性别过滤人脸（默认 female，与 main.py 一致）")
    parser.add_argument("--frame_interval", type=float, default=2.0, help="抽帧间隔秒（默认 2.0）")
    parser.add_argument("--device", type=int, default=0, help="onnxruntime 设备 id（默认 0 GPU）")
    parser.add_argument("--cache_dir", type=Path, default=None, help="特征缓存目录，默认 output_dir/emb_cache")
    parser.add_argument("--dry_run", action="store_true", help="只展示拆分方案，不移动视频/不写画廊")
    args = parser.parse_args()

    output_dir = args.output_dir
    gallery_path = args.gallery or default_gallery_path()
    cache_dir = args.cache_dir or (output_dir / "emb_cache")
    gid = args.gid

    gallery = load_gallery(gallery_path)
    person_names = load_person_names(default_names_path())

    folder = find_group_folder(output_dir, gid)
    if folder is None:
        log.error("找不到分组文件夹 group_%s 在 %s", gid, output_dir)
        sys.exit(1)
    videos = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        log.error("文件夹 %s 里没有视频", folder)
        sys.exit(1)
    log.info("分组 group_%s（%s）：%d 个视频", gid, folder.name, len(videos))

    log.info("加载 insightface 模型（buffalo_l）...")
    app = make_model(args.device)
    log.info("模型加载完成")

    # 重新提取每个视频的代表向量（复用缓存，参数与 main.py 一致）
    reps: list[np.ndarray] = []
    for i, v in enumerate(videos):
        try:
            embs = cached_extract_embeddings(
                app, v, args.frame_interval, cache_dir, args.max_faces,
                args.min_det_score, args.min_face_size, args.dedup_threshold,
                2, 2.0, args.cluster_threshold, args.gender,
            )
        except Exception as e:
            log.warning("[跳过] %s 提取失败: %s", v.name, e)
            embs = np.empty((0, 512), dtype=np.float32)
        reps.append(embs)
        log.info("[%d/%d] %s：%d 个代表向量", i + 1, len(videos), v.name, len(embs))

    no_face = [i for i, r in enumerate(reps) if len(r) == 0]
    if no_face:
        log.warning("以下 %d 个视频无人脸，无法归类，将留在原文件夹：%s",
                    len(no_face), [videos[i].name for i in no_face])

    clusters = [c for c in cluster_videos(reps, args.sim_threshold) if len(c) > 0]
    meaningful = [c for c in clusters if any(len(reps[i]) > 0 for i in c)]
    if len(meaningful) <= 1:
        log.info("未能拆分：全部视频只聚成一簇（可能是同一人，或相似度过高需调低 --sim_threshold）")
        sys.exit(0)

    # 展示方案：最大簇保留原分组，其余簇新建分组
    plan: list[tuple[str, list[int]]] = []
    for ci, cluster in enumerate(meaningful):
        label = gid if ci == 0 else f"新分组{ci}"
        plan.append((label, cluster))
        names = [videos[i].name for i in cluster]
        shown = ", ".join(names[:6]) + ("..." if len(names) > 6 else "")
        log.info("  [%s] %d 个视频：%s", label, len(cluster), shown)

    if args.dry_run:
        log.info("[dry_run] 未做任何实际改动。方案：原组 group_%s 保留 %d 个视频，另建 %d 个新分组（如上）",
                 gid, len(meaningful[0]), len(meaningful) - 1)
        sys.exit(0)

    if not ask_yes_no("按以上方案拆分（最大簇保留原分组）？"):
        log.info("已取消，未做任何改动")
        sys.exit(0)

    # 执行：移动视频 + 重建画廊向量
    for ci, cluster in enumerate(meaningful):
        if ci == 0:
            continue
        new_gid = str(next_group_index(gallery, output_dir))
        new_folder = output_dir / person_folder_name(new_gid, person_names)
        new_folder.mkdir(parents=True, exist_ok=True)
        for i in cluster:
            src = videos[i]
            dst = new_folder / src.name
            if src.exists() and not dst.exists():
                try:
                    shutil.move(str(src), str(dst))
                except OSError as e:
                    log.warning("[跳过] %s 移动失败: %s（留待下次）", src.name, e)
            else:
                log.warning("[跳过] %s 已存在或不存在，未移动", src.name)
        # 新组画廊向量：该簇内有人脸视频的代表向量
        new_vecs = [reps[i] for i in cluster if len(reps[i]) > 0]
        if new_vecs:
            gallery[new_gid] = np.vstack(new_vecs).astype(np.float32)
        log.info("已新建 group_%s（%s），移动 %d 个视频", new_gid, new_folder.name, len(cluster))

    # 原组画廊向量：只保留留在原组的视频
    keep_vecs = [reps[i] for i in meaningful[0] if len(reps[i]) > 0]
    if keep_vecs:
        gallery[gid] = np.vstack(keep_vecs).astype(np.float32)
    else:
        gallery.pop(gid, None)
        log.warning("原组 group_%s 已无带人脸视频，从画廊移除", gid)

    save_gallery(gallery, gallery_path)
    log.info("完成：画廊已重建并保存（%s，%d 个分组），视频已移动", gallery_path, len(gallery))


if __name__ == "__main__":
    main()
