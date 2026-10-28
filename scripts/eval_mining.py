import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.kb import mining

SAMPLES = (
    pathlib.Path(__file__).resolve().parent.parent
    / "tests/data/mining_samples.json"
)


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        pairs = await mining.extract_qa([s["conversation"]])
        questions = [p.question for p in pairs]
        if s.get("expect_empty"):
            ok = not questions
        else:
            ok = any(s["expect_question_contains"] in q for q in questions)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {s['conversation'][:24]!r} -> {questions}")
    print(f"\n挖知识正确 {len(samples) - failures}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
