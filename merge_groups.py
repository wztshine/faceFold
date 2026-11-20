"""分组后处理工具：合并被误拆为多个组的同一个人。

流程：
1. 读取画廊，把"文件夹已不存在"的残留分组从画廊中清理（避免重复提示已合并的分组）
2. 每个分组的向量只保留"主体簇"（组内互相最相似的最大连通分量）：
   画廊向量是逐视频累积的，可能混入次要人脸/噪声，直接用全部向量比较会被
   这类污染向量干扰（例如不同人的组因共享一个背景人脸而出现很高的最大相似度）。
   去掉非主体向量后再比较，结果才可靠。
3. 用主体簇两两比较，找出疑似同一人的组对：
   - 最大相似度（max_sim）达到强匹配阈值
   - 小分组主体被大分组主体覆盖的比例（coverage_ratio）达标
4. 逐对展示给用户确认（或 --auto 全自动）
5. 确认后合并：视频少的组移入保留组文件夹、合并画廊向量，每合并一对立即保存
6. 合并后重新计算候选，直到没有新的疑似组对（保证传递关系不漏）

用法：
    uv run python merge_groups.py "D:/videos/groups"
    uv run python merge_groups.py "D:/videos/groups" --dry_run
    uv run python merge_groups.py "D:/videos/groups" --auto
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

from face_common import (
    dominant_vectors,
    find_group_folder,
    load_gallery,
    load_pending,
    load_person_names,
    save_gallery,
    save_person_names,
    save_pending,
)

log = logging.getLogger("merge_groups")


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    # 挂接 face_common logger，否则其 WARNING（画廊保存失败等）落到 root 的
    # lastResort 处理器输出到 stderr，与其他日志风格不一致
    common = logging.getLogger("face_common")
    common.addHandler(h)
    common.setLevel(logging.INFO)
    common.propagate = False


def group_pair_similarity(ga: np.ndarray, gb: np.ndarray, threshold: float) -> tuple[float, float]:
    """返回 (最大余弦相似度, 小分组主体被大分组主体覆盖的比例)。

    小组合覆盖：较小一侧主体簇里，有多少比例的向量能在另一侧找到达标匹配。
    组大小不对称时（如 3 个向量 vs 88 个向量），全体占比会被稀释得很低，
    "小分组主体整体被大分组吸收"更能说明它们是同一人。
    """
    sims = ga @ gb.T
    max_sim = float(sims.max())
    match_a = float((sims.max(axis=1) >= threshold).mean())
    match_b = float((sims.max(axis=0) >= threshold).mean())
    coverage = match_a if len(ga) <= len(gb) else match_b
    return max_sim, coverage


def compute_candidates(
    gallery: dict[str, np.ndarray],
    sim_threshold: float,
    coverage_ratio: float,
    strong_match_threshold: float,
    cluster_threshold: float,
) -> list[tuple[float, float, str, str]]:
    """找出疑似同一人的组对，返回 (max_sim, coverage_ratio, a, b) 列表（按 max_sim 降序）。"""
    gids = sorted(gallery.keys(), key=int)
    dominants = {g: dominant_vectors(gallery[g], cluster_threshold) for g in gids}
    candidates = []
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            a, b = gids[i], gids[j]
            if len(dominants[a]) == 0 or len(dominants[b]) == 0:
                continue
            max_sim, small = group_pair_similarity(dominants[a], dominants[b], sim_threshold)
            if max_sim >= sim_threshold and (small >= coverage_ratio or max_sim >= strong_match_threshold):
                candidates.append((max_sim, small, a, b))
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    return candidates


def filter_mergeable_candidates(
    candidates: list[tuple[float, float, str, str]],
    output_dir: Path,
    rejected: set[frozenset[str]],
) -> list[tuple[float, float, str, str]]:
    """过滤出"可实际合并"的候选：两个组都必须有文件夹（无文件夹的纯 pending 组无法合并）。"""
    return [
        c for c in candidates
        if frozenset((c[2], c[3])) not in rejected
        and find_group_folder(output_dir, c[2]) is not None
        and find_group_folder(output_dir, c[3]) is not None
    ]


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("请输入 y 或 n")


def count_videos(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.iterdir() if _.is_file())


def merge_two(gallery: dict[str, np.ndarray], output_dir: Path, keep: str, drop: str,
              person_names: dict[str, str] | None = None) -> None:
    """把 drop 组的视频移入 keep 组，合并画廊向量。"""
    keep_folder = find_group_folder(output_dir, keep)
    drop_folder = find_group_folder(output_dir, drop)
    if keep_folder is None:
        keep_folder = output_dir / f"group_{keep}"
    if drop_folder is None:
        log.warning("[跳过] 找不到 drop 组文件夹 group_%s", drop)
        return

    keep_folder.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in drop_folder.iterdir():
        if f.is_file():
            dest = keep_folder / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                moved += 1

    # 合并画廊向量
    if drop in gallery:
        gallery[keep] = np.vstack([gallery[keep], gallery[drop]])
        del gallery[drop]

    # 移除 drop 组的人名映射（keep 组名字保留）
    if person_names is not None:
        person_names.pop(drop, None)

    # 移除空文件夹
    if moved > 0 or not any(drop_folder.iterdir()):
        try:
            drop_folder.rmdir()
        except OSError:
            log.warning("[提示] 文件夹 %s 非空，未删除", drop_folder.name)

    log.info("合并 group_%s -> group_%s（移动 %d 个文件）", drop, keep, moved)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
            stream.reconfigure(line_buffering=True)
    setup_logging()

    parser = argparse.ArgumentParser(description="合并被误拆的同一个人分组")
    parser.add_argument("output_dir", type=Path, help="groups 输出目录")
    parser.add_argument("--gallery", type=Path, default=None,
                        help="画廊文件，默认 ~/.face_video_group/face_gallery.npz（与 main.py 一致）")
    parser.add_argument("--pending", type=Path, default=None,
                        help="待移动视频列表，默认 output_dir/pending.txt（与 main.py 一致）")
    parser.add_argument("--sim_threshold", type=float, default=0.5,
                        help="两张人脸相似度达到多少算对得上（默认 0.5）")
    parser.add_argument("--coverage_ratio", type=float, default=0.5,
                        help="人少那组至少多大比例的人脸能在人多那组找到相似脸，才算疑似同一人（默认 0.5）")
    parser.add_argument("--strong_match_threshold", type=float, default=0.55,
                        help="兜底规则：即使覆盖比例不够，只要两组有某张脸相似度超过该值也提示合并（默认 0.55）")
    parser.add_argument("--cluster_threshold", type=float, default=0.5,
                        help="组内两张人脸相似度超过多少就算同一个人，用来找出每组最主要的人（默认 0.5）")
    parser.add_argument("--dry_run", action="store_true", help="只列出疑似组对，不执行合并")
    parser.add_argument("--auto", action="store_true", help="自动合并所有候选（不逐对确认）")
    parser.add_argument("--cleanup-orphans", action="store_true",
                        help="清理无文件夹的残留分组（默认关闭，避免误删纯 pending 分组）")
    args = parser.parse_args()

    output_dir = args.output_dir
    gallery_path = args.gallery or (Path.home() / ".face_video_group" / "face_gallery.npz")
    pending_path = args.pending or (output_dir / "pending.txt")

    gallery = load_gallery(gallery_path)
    if not gallery:
        log.error("画廊为空或不存在: %s", gallery_path)
        sys.exit(1)

    # 默认加载人物名映射（不存在时为空，不报错）
    names_path = Path.home() / ".face_video_group" / "person_names.json"
    person_names = load_person_names(names_path)

    # 清理残留：默认不自动删（纯 pending 分组没有文件夹，会被误删）。
    # 只有用户显式指定 --cleanup-orphans 才清理无文件夹的分组。
    if args.cleanup_orphans:
        orphans = [g for g in list(gallery) if find_group_folder(output_dir, g) is None]
        for g in orphans:
            log.warning("[清理] 分组 group_%s 的文件夹已不存在（疑似已合并），从画廊移除", g)
            del gallery[g]
        if orphans:
            log.info("[清理] 共移除 %d 个残留分组", len(orphans))
    if not gallery:
        log.error("画廊为空，无分组可合并。")
        sys.exit(1)
    log.info("共 %d 个分组", len(gallery))

    candidates = compute_candidates(gallery, args.sim_threshold, args.coverage_ratio,
                                    args.strong_match_threshold, args.cluster_threshold)
    candidates = filter_mergeable_candidates(candidates, output_dir, set())

    if args.dry_run:
        log.info("发现 %d 组可合并的疑似同一人组对：", len(candidates))
        for max_sim, small, a, b in candidates:
            fa, fb = find_group_folder(output_dir, a), find_group_folder(output_dir, b)
            log.info("  group_%s（视频 %d）<-> group_%s（视频 %d）  最大相似度 %.3f  小组合覆盖 %.2f",
                     a, count_videos(fa), b, count_videos(fb), max_sim, small)
        sys.exit(0)

    if not candidates:
        log.info("没有疑似同一人的组对，无需合并。")
        sys.exit(0)

    # 逐对确认并合并。每合并一对立即保存画廊，并重新计算候选，
    # 这样传递关系（A~B、B~C 但 A~C 相似度略低）也不会漏。
    pending = load_pending(pending_path)
    rejected: set[frozenset[str]] = set()
    round_no = 0
    merged_any = False
    while True:
        round_no += 1
        remaining = filter_mergeable_candidates(
            compute_candidates(gallery, args.sim_threshold, args.coverage_ratio,
                               args.strong_match_threshold, args.cluster_threshold),
            output_dir, rejected,
        )
        if not remaining:
            break
        max_sim, small, a, b = remaining[0]
        fa, fb = find_group_folder(output_dir, a), find_group_folder(output_dir, b)
        log.info("第 %d 轮：还有 %d 个疑似组对，最匹配：", round_no, len(remaining))
        log.info("  group_%s（视频 %d）<-> group_%s（视频 %d）  最大相似度 %.3f  小组合覆盖 %.2f",
                 a, count_videos(fa), b, count_videos(fb), max_sim, small)
        if not args.auto:
            if not ask_yes_no(f"确认合并 group_{a} 和 group_{b} 吗？"):
                rejected.add(frozenset((a, b)))
                continue
        # 保留视频较多的组，小的并入大的
        if count_videos(fa) >= count_videos(fb):
            keep, drop = a, b
        else:
            keep, drop = b, a
        merge_two(gallery, output_dir, keep, drop, person_names)
        merged_any = True
        try:
            save_gallery(gallery, gallery_path)
        except Exception as e:
            log.error("[画廊] 保存失败: %s: %s（视频已移动，画廊未落盘，建议修复后重跑）",
                       type(e).__name__, e)
        # pending.txt 中指向被合并（drop）组的记录重映射到保留组，避免留下死 gid
        remapped = sum(1 for g in pending.values() if g == drop)
        if remapped:
            pending = {p: (keep if g == drop else g) for p, g in pending.items()}
            save_pending(pending_path, pending)
            log.info("[待定] pending.txt 中 %d 条记录已从 group_%s 重映射到 group_%s",
                     remapped, drop, keep)

    try:
        save_gallery(gallery, gallery_path)
    except Exception as e:
        log.error("[画廊] 保存失败: %s: %s", type(e).__name__, e)
    # 发生过合并才写人名映射；即使映射被清空也要落盘（覆盖残留旧条目），
    # 但文件本就不存在且映射为空时不创建新文件
    if merged_any and (person_names or names_path.exists()):
        save_person_names(person_names, names_path)
    log.info("画廊已更新：%s（剩余 %d 个分组）", gallery_path, len(gallery))

    # 提示进度文件的影响
    log.info("提示：视频已移动，下次运行 main.py 时会跳过这些视频（progress.txt 中已记录）。")
    log.info("如需重新处理已合并的分组，请清空 progress.txt 后重跑 main.py。")


if __name__ == "__main__":
    main()
