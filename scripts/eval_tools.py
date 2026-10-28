"""对工具选择标注样例跑真实模型,核对选中的第一个工具。"""

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from app.core.llm import get_chat_model
from app.core.prompts import CUSTOMER_SERVICE_PROMPT
from app.tools.business import TOOLS

SAMPLES = pathlib.Path(__file__).parent.parent / "tests/data/tool_eval.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    model = get_chat_model().bind_tools(TOOLS)
    failures = 0
    for i, s in enumerate(samples, 1):
        messages = CUSTOMER_SERVICE_PROMPT.format_messages(
            history=[HumanMessage(s["text"])]
        )
        ai_msg = await model.ainvoke(messages)
        got = ai_msg.tool_calls[0]["name"] if ai_msg.tool_calls else None
        exp = s["expect_tool"]
        ok = got == exp
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] #{i} {s['text']}")
        print(f"       期望工具={exp} 实际工具={got}")
    print(f"\n{len(samples) - failures}/{len(samples)} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
