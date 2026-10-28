import asyncio

from langchain_core.tools import tool

from app.tools.registry import ToolRegistry

_flaky_calls = 0


@tool
def ok(x: str) -> str:
    """ok 工具。"""
    return f"ok:{x}"


@tool
def flaky(x: str) -> str:
    """前两次失败第三次成功。"""
    global _flaky_calls
    _flaky_calls += 1
    if _flaky_calls < 3:
        raise RuntimeError("boom")
    return "recovered"


@tool
def slow(x: str) -> str:
    """慢工具。"""
    asyncio.run(asyncio.sleep(0.2))
    return "late"


def make_registry() -> ToolRegistry:
    r = ToolRegistry()
    for t in (ok, flaky, slow):
        r.register(t)
    return r


def test_register_get_names_and_list():
    r = make_registry()
    assert set(r.names()) == {"ok", "flaky", "slow"}
    assert r.get("ok") is ok
    assert len(r.as_list()) == 3


def test_execute_validates_args_and_returns_error_text():
    r = make_registry()
    out = asyncio.run(r.execute("ok", {"x": 1}, timeout=1, max_retries=0))
    assert "参数校验失败" in out


def test_execute_retries_then_succeeds():
    global _flaky_calls
    _flaky_calls = 0
    r = make_registry()
    assert asyncio.run(r.execute("flaky", {"x": "a"}, timeout=1, max_retries=2)) == "recovered"


def test_execute_timeout_returns_error_text():
    r = make_registry()
    out = asyncio.run(r.execute("slow", {"x": "a"}, timeout=0.01, max_retries=0))
    assert "工具执行失败" in out


def test_execute_unknown_tool_returns_error_text():
    r = make_registry()
    out = asyncio.run(r.execute("nope", {}, timeout=1, max_retries=0))
    assert "工具不存在" in out
