"""建库材料清单:哪些文件进知识库、各自算什么内容类型。"""
import pathlib

KB_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "kb"

# 文件 → content_type
SOURCE_TYPES: dict[str, str] = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
}

CONTENT_TYPES: tuple[str, ...] = ("faq", "policy", "manual")
