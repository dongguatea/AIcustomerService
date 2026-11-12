"""ch05 意图识别评估:对标注样例跑真实模型,打印总体与分桶准确率。

用法:uv run python scripts/eval_intent_ch05.py(需真实 CHAT_* 配置)。
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from app.core import intent

SAMPLES = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "intent_samples.json"
)


async def main() -> None:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    correct = 0
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in samples:
        label = s["intent"]
        pred = (await intent.classify(s["text"]))["intent"]
        per[label][1] += 1
        if pred == label:
            correct += 1
            per[label][0] += 1
        print(f"{s['text']}  label={label}  pred={pred}")
    total = len(samples)
    print(f"\n总体准确率: {correct}/{total} = {correct / total:.2%}")
    for label in sorted(per):
        c, n = per[label]
        print(f"  {label}: {c}/{n} = {c / n:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
