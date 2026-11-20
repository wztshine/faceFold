"""merge_groups / main 纯逻辑部分的单元测试（文件操作 + 匹配，不加载模型）。

运行（项目根目录）：uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

logging.getLogger("face_common").addHandler(logging.NullHandler())

import merge_groups as mg  # noqa: E402


def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _cluster(rng, n: int, scale: float = 0.02) -> np.ndarray:
    base = _norm(rng.normal(size=(1, 512)))
    return _norm(base + rng.normal(scale=scale, size=(n, 512))).astype(np.float32)


class TestComputeCandidates(unittest.TestCase):
    def test_finds_similar_pair(self):
        rng = np.random.default_rng(11)
        gallery = {
            "1": _cluster(rng, 5),
            "2": _cluster(rng, 4),
            "3": rng.normal(size=(3, 512)).astype(np.float32),
        }
        # 1 和 2 构造为同一人（共享基底）
        gallery["2"] = gallery["1"][:3] + 0.001
        gallery["2"] = _norm(gallery["2"])
        cands = mg.compute_candidates(gallery, 0.5, 0.5, 0.55, 0.5)
        self.assertTrue(any({a, b} == {"1", "2"} for _, _, a, b in cands))

    def test_different_people_not_matched(self):
        rng = np.random.default_rng(12)
        gallery = {"1": _cluster(rng, 5, 10.0), "2": _cluster(rng, 4, 10.0)}
        cands = mg.compute_candidates(gallery, 0.5, 0.5, 0.55, 0.5)
        self.assertEqual(cands, [])


class TestGroupPairSimilarity(unittest.TestCase):
    def test_asymmetric_coverage(self):
        rng = np.random.default_rng(13)
        small = _cluster(rng, 2)
        big = np.vstack([small, _cluster(rng, 10)])
        max_sim, coverage = mg.group_pair_similarity(big, small, 0.5)
        self.assertGreaterEqual(max_sim, 0.5)
        self.assertGreaterEqual(coverage, 0.99)  # 小组被大组完全覆盖


class TestMergeTwo(unittest.TestCase):
    def _setup(self, td: str):
        out = Path(td) / "groups"
        (out / "group_1").mkdir(parents=True)
        (out / "group_2").mkdir(parents=True)
        (out / "group_1" / "keep_a.mp4").write_bytes(b"a")
        (out / "group_2" / "drop_b.mp4").write_bytes(b"b")
        (out / "group_2" / "drop_c.mp4").write_bytes(b"c")
        rng = np.random.default_rng(21)
        gallery = {"1": _cluster(rng, 4), "2": _cluster(rng, 3)}
        return out, gallery

    def test_merge_moves_files_and_vectors(self):
        with tempfile.TemporaryDirectory() as td:
            out, gallery = self._setup(td)
            person_names = {"2": "李四"}
            mg.merge_two(gallery, out, "1", "2", person_names)
            # 视频移入保留组，空文件夹删除
            self.assertTrue((out / "group_1" / "drop_b.mp4").exists())
            self.assertTrue((out / "group_1" / "drop_c.mp4").exists())
            self.assertFalse((out / "group_2").exists())
            # 画廊向量合并
            self.assertNotIn("2", gallery)
            self.assertEqual(len(gallery["1"]), 7)
            # 人名映射清理
            self.assertNotIn("2", person_names)

    def test_name_conflict_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            out, gallery = self._setup(td)
            (out / "group_1" / "drop_b.mp4").write_bytes(b"conflict")  # 制造重名
            mg.merge_two(gallery, out, "1", "2", None)
            self.assertTrue((out / "group_2" / "drop_b.mp4").exists())  # 重名文件留在原组
            self.assertTrue((out / "group_1" / "drop_c.mp4").exists())  # 不重名的正常移动


class TestUniqueDestPath(unittest.TestCase):
    def test_no_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            from main import unique_dest_path
            self.assertEqual(unique_dest_path(d, "v.mp4"), d / "v.mp4")  # 目标不存在，原样返回

    def test_conflict_gets_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "v.mp4").write_bytes(b"x")
            from main import unique_dest_path
            dest = unique_dest_path(d, "v.mp4")
            self.assertEqual(dest.name, "v (1).mp4")
            (d / "v (1).mp4").write_bytes(b"x")
            dest = unique_dest_path(d, "v.mp4")
            self.assertEqual(dest.name, "v (2).mp4")


if __name__ == "__main__":
    unittest.main()
