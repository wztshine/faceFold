"""face_common 的单元测试：纯 numpy 逻辑，不加载模型、不依赖 GPU。

运行（项目根目录）：uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_common import (
    _connected_labels,
    dedup_against_gallery,
    dominant_vectors,
    gid_of,
    load_gallery,
    load_pending,
    load_person_names,
    save_gallery,
    save_pending,
)


def setUpModule() -> None:  # 静默 face_common 的 WARNING 输出
    logging.getLogger("face_common").addHandler(logging.NullHandler())


def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestGidOf(unittest.TestCase):
    def test_id_first(self):
        self.assertEqual(gid_of("4_张三"), "4")
        self.assertEqual(gid_of("5_Ella-Rose"), "5")

    def test_group_prefix(self):
        self.assertEqual(gid_of("group_4"), "4")
        self.assertEqual(gid_of("group_100"), "100")

    def test_leading_zero_canonicalized(self):
        self.assertEqual(gid_of("group_04"), "4")
        self.assertEqual(gid_of("007"), "7")

    def test_legacy_suffix(self):
        self.assertEqual(gid_of("张三_4"), "4")
        self.assertEqual(gid_of("张三_04"), "4")

    def test_no_gid(self):
        self.assertIsNone(gid_of("张三_李四"))
        self.assertIsNone(gid_of("misc"))


class TestConnectedLabels(unittest.TestCase):
    """scipy 路径（n>=128）与纯 Python 路径（n<128）都必须与 scipy 参照划分等价。"""

    def _assert_partition_equal(self, a: np.ndarray, b: np.ndarray):
        self.assertEqual(len(a), len(b))
        same = (a[:, None] == a[None, :]) == (b[:, None] == b[None, :])
        self.assertTrue(np.all(same), "划分不一致")

    def test_equivalence_sparse_and_dense(self):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        rng = np.random.default_rng(2024)
        for trial in range(6):
            n = int(rng.integers(2, 300))
            if trial % 2:  # 稠密：几个簇
                k = 5
                c = _norm(rng.normal(size=(k, 64)))
                v = c[rng.integers(0, k, n)] + rng.normal(scale=0.05, size=(n, 64))
            else:  # 稀疏：随机向量几乎无边
                v = rng.normal(size=(n, 64))
            v = _norm(v)
            adj = (v @ v.T) >= 0.4
            np.fill_diagonal(adj, False)

            labels = _connected_labels(adj)
            _, ref = connected_components(csr_matrix(adj), directed=False)
            with self.subTest(trial=trial, n=n):
                self._assert_partition_equal(labels, ref)


class TestDominantVectors(unittest.TestCase):
    def test_empty_and_single(self):
        self.assertEqual(len(dominant_vectors(np.empty((0, 512), dtype=np.float32), 0.5)), 0)
        v = np.eye(1, 512, dtype=np.float32)
        out = dominant_vectors(v, 0.5)
        self.assertEqual(len(out), 1)

    def test_picks_biggest_cluster(self):
        rng = np.random.default_rng(7)
        base_a, base_b = _norm(rng.normal(size=(2, 512)))
        a = _norm(base_a + rng.normal(scale=0.02, size=(4, 512)))
        b = _norm(base_b + rng.normal(scale=0.02, size=(2, 512)))
        out = dominant_vectors(np.vstack([a, b]).astype(np.float32), 0.5)
        self.assertEqual(len(out), 4)  # 只保留 4 个向量的主体簇


class TestDedupAgainstGallery(unittest.TestCase):
    def test_near_dups_removed(self):
        rng = np.random.default_rng(3)
        gal = _norm(rng.normal(size=(3, 512))).astype(np.float32)
        new = np.vstack([gal[0:1] + 1e-6, rng.normal(size=(1, 512))]).astype(np.float32)
        new = _norm(new)
        kept = dedup_against_gallery(new, gal)
        self.assertEqual(len(kept), 1)  # 近重复被剔除，新向量保留


class TestGalleryRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.npz"
            g = {
                "3": np.random.default_rng(1).normal(size=(4, 512)).astype(np.float32),
                "9": np.random.default_rng(2).normal(size=(2, 512)).astype(np.float32),
            }
            save_gallery(g, gp)
            g2 = load_gallery(gp)
            self.assertEqual(set(g2), set(g))
            for k in g:
                self.assertTrue(np.array_equal(g[k], g2[k]))
            self.assertFalse((gp.parent / (gp.name + ".tmp")).exists())

    def test_corrupt_gallery_backed_up(self):
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.npz"
            save_gallery({"1": np.zeros((1, 512), dtype=np.float32)}, gp)
            data = gp.read_bytes()
            gp.write_bytes(data[: len(data) // 2])  # 截断模拟写坏
            self.assertEqual(load_gallery(gp), {})  # 返回空画廊
            backup = gp.with_suffix(gp.suffix + ".corrupt")
            self.assertTrue(backup.exists())  # 原文件已复制备份


class TestPending(unittest.TestCase):
    def test_roundtrip_and_canonicalization(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "a.mp4"
            f.write_bytes(b"x")
            pp = Path(td) / "pending.txt"
            pp.write_text(f"{f}\t007\n", encoding="utf-8")
            p = load_pending(pp)
            self.assertEqual(list(p.values()), ["7"])  # 前导零规范化
            save_pending(pp, p)
            self.assertEqual(load_pending(pp), p)

    def test_legacy_format(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "a.mp4"
            f.write_bytes(b"x")
            pp = Path(td) / "pending.txt"
            pp.write_text(str(f) + "\n", encoding="utf-8")  # 旧格式：无 gid
            p = load_pending(pp)
            self.assertEqual(list(p.values()), [""])

    def test_missing_file_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            pp = Path(td) / "pending.txt"
            pp.write_text(str(Path(td) / "gone.mp4") + "\t5\n", encoding="utf-8")
            self.assertEqual(load_pending(pp), {})


class TestPersonNames(unittest.TestCase):
    def test_canonicalization_and_empty_drop(self):
        with tempfile.TemporaryDirectory() as td:
            np_ = Path(td) / "n.json"
            np_.write_text('{"07": "张三", "x": "y", "5": ""}', encoding="utf-8")
            names = load_person_names(np_)
            self.assertEqual(names, {"7": "张三", "x": "y"})

    def test_missing_file(self):
        self.assertEqual(load_person_names(Path("no_such_file.json")), {})


if __name__ == "__main__":
    unittest.main()
