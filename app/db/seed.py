from sqlalchemy.orm import Session

from app.db.models import Faq

SEED_FAQS = [
    ("退货政策是什么?", "支持 7 天无理由退货,商品保持完好、不影响二次销售即可申请;以平台售后规则为准。", "售后"),
    ("换货流程怎么走?", "在订单详情页申请换货,审核通过后寄回并重新发货。", "售后"),
    ("发货时效是多久?", "现货商品 48 小时内发出,预售商品按页面标注时间发出。", "物流"),
    ("发票如何开具?", "下单时可选择开具电子发票,开票信息在订单确认后 30 天内可补充。", "发票"),
]


def seed(session: Session) -> None:
    for question, answer, category in SEED_FAQS:
        session.add(Faq(question=question, answer=answer, category=category))
    session.commit()
