"""回归测试：缓存等价判定必须包含连线关系，缓存命中必须能说明来源。

覆盖的坑：同一组节点（类型+参数完全相同），串链 / 分叉 / 换接 / 末端续接
曾被 canonical_key 视为同一次运行，导致跨流水线串用结果图。

运行：python tests/test_cache_topology.py
使用独立临时数据目录，不污染 data/。
"""
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在构造任何存储实例前，把数据目录重定向到临时目录
from server import config  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="cache-topo-test-")
config.DATA_DIR = _TMP
for _name in ("IMAGES_DIR", "RESULTS_DIR", "THUMBS_DIR", "CACHE_DIR", "META_DIR"):
    setattr(config, _name, os.path.join(_TMP, _name.lower()))
for _name in ("IMAGES_JSON", "PIPELINES_JSON", "HISTORY_JSON",
              "PRESETS_JSON", "QUEUE_JSON", "CACHE_JSON"):
    setattr(config, _name, os.path.join(config.META_DIR, _name.lower()))
config._ALL_DIRS = [config.DATA_DIR, config.IMAGES_DIR, config.RESULTS_DIR,
                    config.THUMBS_DIR, config.CACHE_DIR, config.META_DIR]
config.ensure_dirs()

from PIL import Image, ImageDraw  # noqa: E402

from server import pipeline as pipeline_engine  # noqa: E402
from server.batch import process_image  # noqa: E402
from server.cache import ResultCache  # noqa: E402
from server.history import HistoryManager  # noqa: E402
from server.image_store import ImageStore  # noqa: E402

FAILED = []


def check(label, cond):
    print(f"  {'✔' if cond else '✘'} {label}")
    if not cond:
        FAILED.append(label)


def _shapes(w=320, h=240):
    """确定性测试图：硬边色块，模糊前后边缘检测结果差异明显。"""
    img = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 30, 140, 150], fill=(200, 30, 30))
    d.rectangle([160, 60, 300, 200], fill=(30, 60, 200))
    d.ellipse([60, 120, 200, 220], fill=(30, 180, 60))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# 同一组节点（类型+参数完全一致），仅连线不同
CHAIN = [  # n1 -> n2 -> n3
    {"id": "n1", "type": "brightness", "params": {"amount": 25}, "inputs": []},
    {"id": "n2", "type": "blur", "params": {"radius": 4}, "inputs": ["n1"]},
    {"id": "n3", "type": "edges", "params": {"method": "sobel", "strength": 80}, "inputs": ["n2"]},
]
FORK = [  # n1 -> n2, n1 -> n3（分叉）
    {"id": "n1", "type": "brightness", "params": {"amount": 25}, "inputs": []},
    {"id": "n2", "type": "blur", "params": {"radius": 4}, "inputs": ["n1"]},
    {"id": "n3", "type": "edges", "params": {"method": "sobel", "strength": 80}, "inputs": ["n1"]},
]
REWIRED = [  # n2 -> n1 -> n3（同批节点换接）
    {"id": "n1", "type": "brightness", "params": {"amount": 25}, "inputs": ["n2"]},
    {"id": "n2", "type": "blur", "params": {"radius": 4}, "inputs": []},
    {"id": "n3", "type": "edges", "params": {"method": "sobel", "strength": 80}, "inputs": ["n1"]},
]
EXTENDED = CHAIN + [  # 末端续接一个节点
    {"id": "n4", "type": "invert", "params": {}, "inputs": ["n3"]},
]
# 与 CHAIN 同构：节点 id 不同、列表顺序不同、无坐标 -> 必须视为同一条流水线
CHAIN_RENAMED = [
    {"id": "zzz", "type": "edges", "params": {"method": "sobel", "strength": 80}, "inputs": ["mid"]},
    {"id": "abc", "type": "brightness", "params": {"amount": 25}, "inputs": []},
    {"id": "mid", "type": "blur", "params": {"radius": 4}, "inputs": ["abc"]},
]


def test_canonical_key():
    print("== canonical_key：连线关系必须参与指纹 ==")
    k_chain = pipeline_engine.canonical_key(CHAIN)
    check("串链 vs 分叉 -> 键不同", k_chain != pipeline_engine.canonical_key(FORK))
    check("串链 vs 换接 -> 键不同", k_chain != pipeline_engine.canonical_key(REWIRED))
    check("串链 vs 末端续接 -> 键不同", k_chain != pipeline_engine.canonical_key(EXTENDED))
    check("分叉 vs 换接 -> 键不同",
          pipeline_engine.canonical_key(FORK) != pipeline_engine.canonical_key(REWIRED))
    check("同构（换 id/换顺序）-> 键相同", k_chain == pipeline_engine.canonical_key(CHAIN_RENAMED))
    check("同一输入两次生成 -> 键稳定", k_chain == pipeline_engine.canonical_key(CHAIN))

    # 非法图（环/悬空引用）：键生成必须确定性地返回，不得死循环
    cyc = [{"id": "a", "type": "blur", "params": {}, "inputs": ["b"]},
           {"id": "b", "type": "blur", "params": {}, "inputs": ["a"]}]
    dangling = [{"id": "a", "type": "blur", "params": {}, "inputs": ["ghost"]}]
    check("含环流水线 -> 键可生成且稳定",
          pipeline_engine.canonical_key(cyc) == pipeline_engine.canonical_key(cyc))
    check("悬空引用 -> 键可生成且稳定",
          pipeline_engine.canonical_key(dangling) == pipeline_engine.canonical_key(dangling))
    check("含环流水线校验报错", bool(pipeline_engine.validate(cyc)))


def test_end_to_end():
    print("== 端到端：不同连线绝不串用结果，命中能说明来源 ==")
    image_store = ImageStore()
    cache = ResultCache()
    history = HistoryManager()
    rec = image_store.save_upload(_shapes(), "shapes.png")
    iid = rec["id"]

    r_chain = process_image(image_store, cache, history, iid, CHAIN, pipeline_name="链式")
    r_fork = process_image(image_store, cache, history, iid, FORK, pipeline_name="分叉")
    check("链式首跑 -> 重新计算", r_chain["cache_hit"] is False and r_chain["result_id"])
    check("分叉首跑 -> 不得命中链式缓存", r_fork["cache_hit"] is False and r_fork["result_id"])
    check("两次运行 -> result_id 不同", r_chain["result_id"] != r_fork["result_id"])

    with open(cache.result_path(r_chain["result_id"]), "rb") as f:
        chain_bytes = f.read()
    with open(cache.result_path(r_fork["result_id"]), "rb") as f:
        fork_bytes = f.read()
    check("两条流水线的结果图内容不同", chain_bytes != fork_bytes)

    # 原样重跑：命中自己的缓存，且来源指向自己
    r_chain2 = process_image(image_store, cache, history, iid, CHAIN, pipeline_name="链式")
    check("链式重跑 -> 命中缓存", r_chain2["cache_hit"] is True)
    check("链式重跑 -> 复用自己的结果", r_chain2["result_id"] == r_chain["result_id"])
    src = r_chain2.get("cache_source") or {}
    check("命中说明来源流水线名", src.get("pipeline_name") == "链式")
    check("命中说明来源节点数", src.get("node_count") == len(CHAIN))
    check("命中说明来源结果生成时间", bool(src.get("created_at")))

    # 换 id/换顺序的同构流水线：仍应命中链式的缓存（有益的复用要保留）
    r_iso = process_image(image_store, cache, history, iid, CHAIN_RENAMED, pipeline_name="链式副本")
    check("同构流水线 -> 命中缓存", r_iso["cache_hit"] is True)
    check("同构流水线 -> 复用链式结果", r_iso["result_id"] == r_chain["result_id"])
    check("同构命中 -> 来源仍指向最初计算的「链式」",
          (r_iso.get("cache_source") or {}).get("pipeline_name") == "链式")

    # 分叉重跑：必须命中分叉自己的结果，而不是链式的
    r_fork2 = process_image(image_store, cache, history, iid, FORK, pipeline_name="分叉")
    check("分叉重跑 -> 命中缓存", r_fork2["cache_hit"] is True)
    check("分叉重跑 -> 复用分叉的结果", r_fork2["result_id"] == r_fork["result_id"])
    check("分叉命中 -> 来源是「分叉」",
          (r_fork2.get("cache_source") or {}).get("pipeline_name") == "分叉")

    # 历史记录：命中条目必须携带来源，供历史页展示
    hit_entries = [e for e in history.list() if e.get("cache_hit")]
    check("历史中的命中记录都带 cache_source",
          bool(hit_entries) and all(e.get("cache_source") for e in hit_entries))
    check("历史来源含流水线名",
          all(e["cache_source"].get("pipeline_name") for e in hit_entries))


def main():
    test_canonical_key()
    test_end_to_end()
    if FAILED:
        print(f"\n{len(FAILED)} 项未通过：")
        for label in FAILED:
            print(f"  ✘ {label}")
        sys.exit(1)
    print("\n全部通过 ✔")


if __name__ == "__main__":
    main()
