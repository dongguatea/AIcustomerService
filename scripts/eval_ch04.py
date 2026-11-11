"""四策略(vector/bm25/hybrid/hybrid_rerank)分桶评估。需 Milvus Standalone + 已建库 + 上游可调通。

三段指标:
  1) 检索段(确定性):四策略 × A/B/C 桶 Recall@5 / MRR(按 expect_section 命中 section_path)。
  2) 证据覆盖度(确定性·跨策略):四策略召回 Top-K 证据盖住 expect_points 的机械比例。
  3) 生成段(聊天模型):hybrid_rerank 线上管线的 Faithfulness(A/B/C)+ D 桶拒答率。

--skip-gen 只跑前两段(不碰裁判模型)。产物落 data/ch04/reports/rag_eval.{txt,json}。
入口:make eval-rag
"""
import asyncio
import json
import pathlib
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from app.config import settings
from app.core import llm, retrieval
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.tools.business import run_query_faq

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "tests/data/eval_ch04.jsonl"
OUT_DIR = ROOT / "data/ch04/reports"
OUT_JSON = OUT_DIR / "rag_eval.json"
OUT_TXT = OUT_DIR / "rag_eval.txt"

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
GRADED_BUCKETS = ["A_policy", "B_model", "C_colloquial"]
STRATEGY_LABEL = {"vector": "纯向量", "bm25": "纯 BM25", "hybrid": "混合",
                  "hybrid_rerank": "混合+重排"}
BUCKET_LABEL = {"A_policy": "政策类", "B_model": "型号类",
                "C_colloquial": "口语类", "overall": "总体"}
K = 10                 # 检索深度(与 rerank_top_k 对齐)
RECALL_K = 5           # 召回率截断位
SKIP_GEN = "--skip-gen" in sys.argv
RETR_CONCURRENCY = 2
CALL_TIMEOUT = 45.0

_LINES: list[str] = []
_retr_sem = asyncio.Semaphore(RETR_CONCURRENCY)


class _Faith(BaseModel):
    faithful: bool = Field(description="是否忠实于证据")
    reason: str = Field(default="", description="判断依据")


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(msg)


def _norm(s: str) -> str:
    return "".join((s or "").split())


def _hit_rank(hits: list[dict], expect_section: list[str]) -> int:
    """返回命中最靠前的 1-based 名次;一条都没召回回 0。"""
    for i, h in enumerate(hits):
        sp = h.get("section_path") or ""
        if any(w in sp for w in expect_section):
            return i + 1
    return 0


def _recall_at(rank: int, k: int) -> float:
    return 1.0 if 0 < rank <= k else 0.0


def _rr(rank: int) -> float:
    return 1.0 / rank if rank else 0.0


def _evidence_text(hits: list[dict]) -> str:
    return "\n".join(f"{h.get('question', '')}:{h.get('answer', '')}" for h in hits)


def _coverage_mech(points: list[str], hits: list[dict]) -> float | None:
    if not points:
        return None
    ev = _norm(_evidence_text(hits))
    return sum(1 for p in points if _norm(p) in ev) / len(points)


def _load() -> list[dict]:
    return [json.loads(ln) for ln in EVAL.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _empty_acc() -> dict:
    return {
        st: {b: {"recall": [], "mrr": [], "coverage": []}
             for b in [*GRADED_BUCKETS, "overall"]}
        for st in STRATEGIES
    }


async def _retrieve_one(sample: dict, strategy: str) -> list[dict]:
    async with _retr_sem:
        for attempt in range(2):
            try:
                return await retrieval.search_knowledge(
                    sample["query"], strategy=strategy, top_k=K)
            except Exception as exc:  # noqa: BLE001 - 上游瞬时限流/抖动,退避重试一次,仍失败记空
                if attempt == 0:
                    await asyncio.sleep(1.5)
                    continue
                _log(f"  ! {sample['id']}/{strategy} 检索失败: {type(exc).__name__}")
        return []


async def _retrieval_segment(samples: list[dict]) -> tuple[dict, dict]:
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    acc = _empty_acc()
    hits_map: dict[tuple[str, str], list[dict]] = {}
    for strategy in STRATEGIES:
        results = await asyncio.gather(*[_retrieve_one(s, strategy) for s in graded])
        _log(f"[检索] {STRATEGY_LABEL[strategy]}: {len(results)} 题")
        for s, hits in zip(graded, results):
            bucket = s["bucket"]
            hits_map[(strategy, s["id"])] = hits
            rank = _hit_rank(hits, s["expect_section"])
            cov = _coverage_mech(s["expect_points"], hits)
            for b in (bucket, "overall"):
                acc[strategy][b]["recall"].append(_recall_at(rank, RECALL_K))
                acc[strategy][b]["mrr"].append(_rr(rank))
                if cov is not None:
                    acc[strategy][b]["coverage"].append(cov)
    return acc, hits_map


def _summarize(acc: dict) -> dict:
    out: dict = {}
    for st in STRATEGIES:
        out[st] = {}
        for b in [*GRADED_BUCKETS, "overall"]:
            out[st][b] = {
                "recall_at_5": round(_mean(acc[st][b]["recall"]), 4),
                "mrr": round(_mean(acc[st][b]["mrr"]), 4),
                "evidence_coverage": round(_mean(acc[st][b]["coverage"]), 4),
            }
    return out


async def _try(coro_factory, label: str):
    for attempt in range(2):
        try:
            return await asyncio.wait_for(coro_factory(), CALL_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            if attempt == 0:
                await asyncio.sleep(2.0)
                continue
            _log(f"  ! {label} 失败: {type(exc).__name__}")
    return None


async def _gen_answer(query: str, evidence: str) -> str:
    model = llm.get_chat_model(streaming=False, temperature=0)
    msg = await (RAG_ANSWER_PROMPT | model).ainvoke({"query": query, "evidence": evidence})
    return msg.content if isinstance(msg.content, str) else str(msg.content)


async def _judge_faithful(evidence: str, answer: str) -> dict:
    model = llm.structured(_Faith)
    r: _Faith = await (FAITHFULNESS_PROMPT | model).ainvoke(
        {"evidence": evidence, "answer": answer})
    return {"faithful": bool(r.faithful), "reason": r.reason or ""}


async def _generation_segment(samples: list[dict], hits_map: dict) -> dict | None:
    """hybrid_rerank 线上管线的 Faithfulness + D 桶拒答率。失败返回 None(不拖垮整轮)。"""
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    absent = [s for s in samples if s["bucket"] == "D_oob"]
    faith = {b: [] for b in [*GRADED_BUCKETS, "overall"]}
    cases: list[dict] = []
    for s in graded:
        hits = hits_map.get(("hybrid_rerank", s["id"]), [])
        evidence = _evidence_text(hits)
        answer = await _try(lambda: _gen_answer(s["query"], evidence), f"{s['id']} 生成")
        if answer is None:
            continue
        verdict = await _try(lambda: _judge_faithful(evidence, answer), f"{s['id']} 忠实度")
        if verdict is None:
            continue
        faithful = verdict["faithful"]
        for b in (s["bucket"], "overall"):
            faith[b].append(1.0 if faithful else 0.0)
        if not faithful:
            cases.append({"id": s["id"], "query": s["query"], "answer": answer,
                          "reason": verdict["reason"]})
        _log(f"  {s['id']} faithful={faithful} {verdict['reason'][:40]}")

    refusals = 0
    for s in absent:
        r = await _try(lambda: run_query_faq(s["query"]), f"{s['id']} 拒答")
        if r is not None and not r.sufficient:
            refusals += 1
    _log(f"[生成] 库外拒答 {refusals}/{len(absent)}")

    faithfulness = {b: round(_mean(faith[b]), 4) for b in [*GRADED_BUCKETS, "overall"]}
    refusal_rate = round(refusals / len(absent), 4) if absent else 0.0
    return {"faithfulness": faithfulness, "refusal_rate": refusal_rate, "cases": cases}


def _report(retrieval_summary: dict, generation: dict | None) -> dict:
    return {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "strategies": STRATEGIES,
            "buckets": GRADED_BUCKETS + ["D_oob"],
            "recall_k": RECALL_K,
            "retrieval_depth": K,
            "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model,
        },
        "retrieval": retrieval_summary,
        "generation": generation,
    }


def _print_report(retrieval_summary: dict, generation: dict | None) -> None:
    _log("\n== 检索段(Recall@5 / MRR / 证据覆盖度)==")
    for st in STRATEGIES:
        _log(f"【{STRATEGY_LABEL[st]}】")
        for b in [*GRADED_BUCKETS, "overall"]:
            row = retrieval_summary[st][b]
            _log(f"  {BUCKET_LABEL[b]:<4} Recall@5={row['recall_at_5']:.3f} "
                 f"MRR={row['mrr']:.3f} 覆盖度={row['evidence_coverage']:.3f}")
    if generation:
        _log("\n== 生成段(hybrid_rerank)==")
        _log("  " + "  ".join(
            f"{BUCKET_LABEL[b]}忠实度={generation['faithfulness'][b]:.3f}"
            for b in [*GRADED_BUCKETS, "overall"]))
        _log(f"  库外拒答率={generation['refusal_rate']:.3f}")
    else:
        _log("\n== 生成段:本次未完成(--skip-gen)==")


def main() -> None:
    samples = _load()
    _log(f"评估集 {len(samples)} 题;检索深度 K={K}, Recall@K={RECALL_K}")
    acc, hits_map = asyncio.run(_retrieval_segment(samples))
    retrieval_summary = _summarize(acc)

    generation: dict | None = None
    if not SKIP_GEN:
        try:
            generation = asyncio.run(_generation_segment(samples, hits_map))
        except Exception as exc:  # noqa: BLE001
            _log(f"[生成段未完成] {type(exc).__name__}: {exc}")

    _print_report(retrieval_summary, generation)
    report = _report(retrieval_summary, generation)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"\n产物已写入 {OUT_TXT} / {OUT_JSON}")


if __name__ == "__main__":
    main()
