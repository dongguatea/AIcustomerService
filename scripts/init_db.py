import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.seed import seed
from app.db.session import SessionLocal, init_db


def main() -> int:
    init_db()
    with SessionLocal() as session:
        seed(session)
    print("数据库初始化完成: 四张表已创建,FAQ 种子已灌入")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
