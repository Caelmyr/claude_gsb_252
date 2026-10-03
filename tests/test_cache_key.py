"""回归测试：缓存等价判定必须把「连线关系」算进去。

背景 bug：canonical_key 旧实现只按拓扑序拼 (type, params)，完全忽略 inputs 边，
导致同样的节点「串行」与「从同一节点分叉」得到同一个缓存键，第二条流水线
直接命中第一条的结果图；错误结果再经保存流水线 / 历史恢复 / 批量处理扩散。

本测试覆盖：
1. 指纹层（无需 Pillow，stub 掉算法层即可）：
   - 不同连线（串行/分叉/末端独立/末端再接节点）指纹必不同；
   - 同一条流水线在「节点 id 重命名 / 数组顺序打乱 / 画布坐标变化」后指纹不变；
   - 参数变化、节点类型变化、重复计算的稳定性。
2. 端到端（需要 Pillow，无则跳过）：process_image 对分叉不复用串行结果，
   且缓存命中时 reused_from 指回首次产出该图的流水线。

运行：python3 tests/test_cache_key.py
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_algorithms():
    """用可返回任意属性的假模块替换依赖 Pillow 的算法层，仅为导入 pipeline。"""
    def stub(name):
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: (lambda *a, **k: None)  # noqa: E731
        sys.modules[name] = m
        return m

    alg = types.ModuleType("server.algorithms")
    sys.modules["server.algorithms"] = alg
    for short in ["color", "detection", "features", "filters",
                  "geometry", "segmentation", "style", "util"]:
        setattr(alg, short, stub(f"server.algorithms.{short}"))
    sys.modules["server.algorithms.style"].STYLES = {"oil": 1}


try:  # 有 Pillow 时用真实算法层，端到端用例才能真正产出图像
    import PIL  # noqa: F401
except Exception:
    _stub_algorithms()

from server import pipeline as P  # noqa: E402


def _node(i, t, params, ins, x=10, y=20):
    return {"id": i, "type": t, "params": params, "inputs": ins, "x": x, "y": y}


def _chain():
    return [
        _node("a", "brightness", {"amount": 30}, []),
        _node("b", "blur", {"radius": 2}, ["a"]),
        _node("c", "edges", {"method": "sobel"}, ["b"]),
    ]


class CanonicalKeyTests(unittest.TestCase):
    def test_wiring_changes_key(self):
        chain = _chain()
        fork = [
            _node("a", "brightness", {"amount": 30}, []),
            _node("b", "blur", {"radius": 2}, ["a"]),
            _node("c", "edges", {"method": "sobel"}, ["a"]),  # 分叉：c 接 a 而非 b
        ]
        detached = [
            _node("a", "brightness", {"amount": 30}, []),
            _node("b", "blur", {"radius": 2}, ["a"]),
            _node("c", "edges", {"method": "sobel"}, []),    # 末端独立
        ]
        extended = chain + [_node("d", "invert", {}, ["c"])]  # 末端下再接节点

        k = P.canonical_key(chain)
        self.assertNotEqual(k, P.canonical_key(fork), "串行与分叉绝不能共用缓存键")
        self.assertNotEqual(k, P.canonical_key(detached), "末端独立与串行绝不能共用缓存键")
        self.assertNotEqual(k, P.canonical_key(extended), "末端再加节点必须改变指纹")
        self.assertNotEqual(P.canonical_key(fork), P.canonical_key(detached))

    def test_key_invariant_to_identity_and_layout(self):
        """同一条流水线：id 重分配、数组重排、拖动节点，都应继续命中自己的结果。"""
        chain = _chain()
        renamed = [  # 模拟前端加载/历史恢复时 newId() 重新分配全部 id
            _node("x9", "brightness", {"amount": 30}, []),
            _node("y7", "blur", {"radius": 2}, ["x9"]),
            _node("z3", "edges", {"method": "sobel"}, ["y7"]),
        ]
        shuffled = [chain[2], chain[0], chain[1]]  # 数组顺序打乱、连线不变
        moved = [dict(n, x=n["x"] + 120, y=n["y"] + 60) for n in chain]

        k = P.canonical_key(chain)
        self.assertEqual(k, P.canonical_key(renamed))
        self.assertEqual(k, P.canonical_key(shuffled))
        self.assertEqual(k, P.canonical_key(moved))

    def test_params_and_types_change_key(self):
        chain = _chain()
        changed_param = [
            _node("a", "brightness", {"amount": 80}, []),
            _node("b", "blur", {"radius": 2}, ["a"]),
            _node("c", "edges", {"method": "sobel"}, ["b"]),
        ]
        changed_type = [
            _node("a", "contrast", {"amount": 30}, []),
            _node("b", "blur", {"radius": 2}, ["a"]),
            _node("c", "edges", {"method": "sobel"}, ["b"]),
        ]
        self.assertNotEqual(P.canonical_key(chain), P.canonical_key(changed_param))
        self.assertNotEqual(P.canonical_key(chain), P.canonical_key(changed_type))

    def test_stable_and_empty(self):
        self.assertEqual(P.canonical_key(_chain()), P.canonical_key(_chain()))
        self.assertEqual(P.canonical_key([]), P.canonical_key([]))

    def test_deeper_chain_depth_changes_key(self):
        # a->b 与 a->b->c 即便某节点恰好是 sink，也必须靠结构色区分
        two = [
            _node("a", "brightness", {"amount": 30}, []),
            _node("b", "blur", {"radius": 2}, ["a"]),
        ]
        self.assertNotEqual(P.canonical_key(two), P.canonical_key(_chain()))


class ProcessImageWiringTests(unittest.TestCase):
    """端到端：不同连线不得复用结果图；命中时 reused_from 可追溯。"""

    @classmethod
    def setUpClass(cls):
        try:
            import io  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            raise unittest.SkipTest("未安装 Pillow，跳过端到端测试")
        # 到这里才导入依赖 Pillow 的模块
        import tempfile
        import importlib
        from server import config, storage, image_store, cache as cache_mod
        from server import history as history_mod, batch
        cls._tempfile = tempfile
        cls._importlib = importlib
        cls._config = config
        cls._image_store_mod = image_store
        cls._cache_mod = cache_mod
        cls._history_mod = history_mod
        cls._batch = batch

    def setUp(self):
        self.tmp = self._tempfile.TemporaryDirectory()
        config = self._config
        config.DATA_DIR = self.tmp.name
        dirs = {"IMAGES_DIR": "images", "RESULTS_DIR": "results",
                "THUMBS_DIR": "thumbnails", "CACHE_DIR": "cache", "META_DIR": "metadata"}
        for attr, name in dirs.items():
            setattr(config, attr, os.path.join(self.tmp.name, name))
        for attr in ("IMAGES_JSON", "PIPELINES_JSON", "HISTORY_JSON", "PRESETS_JSON",
                     "QUEUE_JSON", "CACHE_JSON"):
            setattr(config, attr, os.path.join(self.tmp.name, "metadata", attr.lower()))
        for attr, name in dirs.items():
            os.makedirs(getattr(config, attr), exist_ok=True)
        os.makedirs(os.path.join(self.tmp.name, "metadata"), exist_ok=True)
        config.ensure_dirs()

        self._importlib.reload(self._image_store_mod)
        self._importlib.reload(self._cache_mod)
        self._importlib.reload(self._history_mod)
        self._importlib.reload(self._batch)
        self.image_store = self._image_store_mod.ImageStore()
        self.cache = self._cache_mod.ResultCache()
        self.history = self._history_mod.HistoryManager()

        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (120, 130, 140)).save(buf, "PNG")
        self.image_id = self.image_store.save_upload(buf.getvalue(), "t.png")["id"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_fork_does_not_reuse_chain_result(self):
        chain = _chain()
        fork = [
            _node("a", "brightness", {"amount": 30}, []),
            _node("b", "blur", {"radius": 2}, ["a"]),
            _node("c", "edges", {"method": "sobel"}, ["a"]),
        ]
        r1 = self._batch.process_image(self.image_store, self.cache, self.history,
                                       self.image_id, chain, pipeline_name="串行流水线")
        r2 = self._batch.process_image(self.image_store, self.cache, self.history,
                                       self.image_id, fork, pipeline_name="分叉流水线")
        self.assertFalse(r1["cache_hit"])
        self.assertFalse(r2["cache_hit"], "分叉连线首次运行必须重新计算，不得命中串行结果")
        self.assertNotEqual(r1["result_id"], r2["result_id"])

        # 同一条串行再跑一次：应当命中，且来源指向「串行流水线」
        r3 = self._batch.process_image(self.image_store, self.cache, self.history,
                                       self.image_id, chain, pipeline_name="串行流水线-副本")
        self.assertTrue(r3["cache_hit"])
        self.assertEqual(r3["result_id"], r1["result_id"])
        self.assertEqual((r3["reused_from"] or {}).get("pipeline_name"), "串行流水线")


if __name__ == "__main__":
    unittest.main(verbosity=2)
