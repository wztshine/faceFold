"""纯 numpy/标准库工具，供 main.py 与 merge_groups.py 共享。

不依赖 cv2 / insightface / onnxruntime，使 merge_groups 这类后处理工具
无需加载重型视觉库即可运行。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("face_common")


def dedup_against_gallery(embs: np.ndarray, group_vecs: np.ndarray, near_dup: float = 0.99) -> np.ndarray:
    """剔除与组内已有向量高度相似（>near_dup）的向量，防止 pending 视频重复灌入。"""
    if len(group_vecs) == 0:
        return embs
    sims = group_vecs @ embs.T
    keep = ~(sims.max(axis=0) >= near_dup)
    return embs[keep]


def dominant_vectors(embs: np.ndarray, link: float) -> np.ndarray:
    """提取分组的"主体簇"：组内向量按相似度 >= link 连成连通分量，取最大的分量。

    画廊向量逐视频累积，可能混入次要人脸/背景人脸等污染向量。
    用全部向量比较时，污染向量会制造虚假的"高相似"，导致误识别。
    主体簇代表该组最主要的（同一个人）人脸集合，用它比较才可靠。

    单链接 union-find 可能通过中间人把两个不同的人"链"进同一分量。
    因此在每个连通分量内再做**平均链接**子聚类：用与分量质心的相似度作种子，
    只保留与质心相似度 >= link 的成员，且质心随成员重新迭代计算。
    若迭代把主体簇抽空，退回 medoid（与成员最相似的点），保证不丢数据。
    """
    n = len(embs)
    if n <= 1:
        return embs
    sims = embs @ embs.T
    adj = sims >= link
    np.fill_diagonal(adj, False)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                parent[find(i)] = find(j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    best: np.ndarray | None = None
    for members in comps.values():
        if len(members) == 1:
            if best is None:
                best = embs[members]
            continue
        sub = embs[members]
        # 以质心为种子，迭代收敛到"与质心都像"的最大子簇（平均链接思想）
        keep = sub
        for _ in range(10):
            centroid = keep.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                break
            centroid = centroid / norm
            new_keep = keep[keep @ centroid >= link]
            if len(new_keep) == len(keep):
                break
            keep = new_keep
            if len(keep) == 0:
                break
        if len(keep) == 0:
            # 迭代把主体簇抽空了：退回 medoid（与成员最相似的点），保证不丢数据
            sims_sum = sub @ sub.T
            np.fill_diagonal(sims_sum, -1.0)
            medoid_idx = int(np.argmax(sims_sum.sum(axis=1)))
            keep = sub[medoid_idx : medoid_idx + 1]
        if len(keep) > 0 and (best is None or len(keep) > len(best)):
            best = keep

    if best is None:
        return np.empty((0, 512), dtype=np.float32)
    return best


def gid_of(name: str) -> str | None:
    """从文件夹名提取人物 ID。

    优先采用"id 在前"格式（如 '4_张三'、'5_Ella-Rose'）：取第一个 _ 之前的段。
    若该段不是纯数字（旧格式 '张三_4'），回退到遍历各段找唯一的数字段。
    """
    parts = name.split("_")
    if parts and parts[0].isdigit():
        return parts[0]
    digits = [p for p in parts if p.isdigit()]
    if len(digits) == 1:
        return digits[0]
    return None


def find_group_folder(output_dir: Path, gid: str) -> Path | None:
    """在输出目录中查找后缀数字为 gid 的文件夹（支持任意前缀命名）。"""
    if output_dir.is_dir():
        for d in output_dir.iterdir():
            if d.is_dir() and gid_of(d.name) == gid:
                return d
    return None


def load_gallery(gallery_path: Path) -> dict[str, np.ndarray]:
    """加载画廊：{gid: (N, 512) 向量矩阵}，键统一为数字后缀。不存在时返回空字典。"""
    if gallery_path.exists():
        try:
            data = np.load(gallery_path, allow_pickle=True)
            gallery: dict[str, np.ndarray] = {}
            for k in data.files:
                gid = gid_of(k)
                if gid is not None:
                    gallery[gid] = data[k].astype(np.float32)
            return gallery
        except Exception as e:
            log.warning("[画廊] 读取失败，按空画廊处理: %s", e)
    return {}


def save_gallery(gallery: dict[str, np.ndarray], gallery_path: Path) -> None:
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gallery_path, **gallery)


def load_person_names(names_path: Path) -> dict[str, str]:
    """加载人物名映射 {人物id: 人名}。文件不存在或损坏时返回空字典（不报错）。"""
    if names_path is None or not names_path.exists():
        return {}
    try:
        with names_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.warning("[人名] 格式错误，应为 JSON 对象，按空处理")
            return {}
        return {str(k): str(v) for k, v in data.items() if v}
    except Exception as e:
        log.warning("[人名] 读取失败 %s，按空处理: %s", names_path.name, e)
        return {}


def save_person_names(names: dict[str, str], names_path: Path) -> None:
    """保存人物名映射。文件目录不存在时创建。"""
    if names_path is None:
        return
    names_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with names_path.open("w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("[人名] 写入失败: %s", e)


def person_folder_name(gid: str, names: dict[str, str]) -> str:
    """根据人物名映射生成文件夹名：有人名用 gid_人名，否则用默认 group_gid。"""
    name = names.get(gid)
    if name:
        return f"{gid}_{name}"
    return f"group_{gid}"
