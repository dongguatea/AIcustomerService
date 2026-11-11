"""评估集自检:对 data/kb/*.md 源文件校验 id/分桶/章节命中/要点逐字,不依赖 Milvus。"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "tests/data/eval_ch04.jsonl"
KB_DIR = ROOT / "data/kb"

EXPECTED_BUCKETS = {
    "A_policy": 20,
    "B_model": 20,
    "C_colloquial": 20,
    "D_oob": 20,
}
GRADED = {"A_policy", "B_model", "C_colloquial"}


def _norm(s: str) -> str:
    return "".join((s or "").split())


def _load_kb() -> tuple[str, str]:
    raw_parts: list[str] = []
    for f in sorted(KB_DIR.glob("*.md")):
        raw_parts.append(f.read_text(encoding="utf-8"))
    raw = "\n".join(raw_parts)
    headings: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^#{2,4}\s+(.+)$", line)
        if m:
            headings.append(m.group(1).strip())
    return raw, "\n".join(headings)


def main() -> int:
    raw, headings = _load_kb()
    norm_raw = _norm(raw)
    errors: list[str] = []
    samples = [json.loads(ln) for ln in EVAL.read_text(encoding="utf-8").splitlines() if ln.strip()]

    ids = [s["id"] for s in samples]
    queries = [s["query"] for s in samples]
    if len(ids) != len(set(ids)):
        errors.append("id 重复")
    if len(queries) != len(set(queries)):
        errors.append("query 重复")

    from collections import Counter

    counts = Counter(s["bucket"] for s in samples)
    for bucket, want in EXPECTED_BUCKETS.items():
        if counts.get(bucket) != want:
            errors.append(f"桶 {bucket} 题数 {counts.get(bucket, 0)} != {want}")

    for s in samples:
        bucket, sid = s["bucket"], s["id"]
        if bucket in GRADED:
            if s.get("should_refuse"):
                errors.append(f"{sid} 可作答桶却 should_refuse=true")
            for kw in s.get("expect_section") or []:
                if kw not in headings:
                    errors.append(f"{sid} expect_section 关键词 {kw!r} 不在任何章节标题")
            for p in s.get("expect_points") or []:
                if _norm(p) not in norm_raw:
                    errors.append(f"{sid} expect_point {p!r} 未逐字出现在知识库")
        elif bucket == "D_oob":
            if not s.get("should_refuse"):
                errors.append(f"{sid} 库外桶却 should_refuse=false")
            if s.get("expect_section") or s.get("expect_points"):
                errors.append(f"{sid} 库外桶不应带 ground truth")
        else:
            errors.append(f"{sid} 未知桶 {bucket!r}")

    if errors:
        print(f"评估集自检失败: {len(errors)} 处")
        for e in errors:
            print(" -", e)
        return 1
    print(f"评估集自检通过: {len(samples)} 题,四桶各 20,章节/要点均命中知识库")
    return 0


if __name__ == "__main__":
    sys.exit(main())
