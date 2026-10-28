from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Conversation, Faq, Message

SEED_FAQS = [
    ("退货政策是什么?", "支持 7 天无理由退货,商品保持完好、不影响二次销售即可申请;以平台售后规则为准。", "售后"),
    ("换货流程怎么走?", "在订单详情页申请换货,审核通过后寄回并重新发货。", "售后"),
    ("发货时效是多久?", "现货商品 48 小时内发出,预售商品按页面标注时间发出。", "物流"),
    ("发票如何开具?", "下单时可选择开具电子发票,开票信息在订单确认后 30 天内可补充。", "发票"),
]

SEED_CONVERSATIONS = [
    ("seed-u1", [
        ("user", "你们发货一般多久啊"),
        ("assistant", "现货商品付款后 48 小时内发货,预售以商品详情页标注时间为准。"),
    ]),
    ("seed-u2", [
        ("user", "满多少包邮"),
        ("assistant", "单笔订单满 99 元包邮,未满收取 10 元运费,偏远地区另计。"),
    ]),
]


def seed(session: Session) -> None:
    for question, answer, category in SEED_FAQS:
        session.add(Faq(question=question, answer=answer, category=category))
    for session_id, msgs in SEED_CONVERSATIONS:
        exists = session.scalar(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        if exists:
            continue
        conv = Conversation(session_id=session_id, user=session_id, status="open")
        session.add(conv)
        session.flush()
        for role, content in msgs:
            session.add(Message(conversation_id=conv.id, role=role, content=content))
    session.commit()
