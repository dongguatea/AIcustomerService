from app.db.models import Conversation, Faq, Message
from app.db.seed import seed
from app.repositories import search_faq


def test_seed_contains_return_policy_and_not_shipping_fee(session_factory):
    with session_factory() as s:
        seed(s)
        assert search_faq(s, "退货")
        assert search_faq(s, "邮费") == []
        assert search_faq(s, "运费") == []
        assert s.query(Faq).count() >= 4


def test_seed_creates_mining_conversations(session_factory):
    with session_factory() as s:
        seed(s)
        ids = ["seed-u1", "seed-u2"]
        assert s.query(Conversation).filter(Conversation.session_id.in_(ids)).count() == 2
        assert s.query(Message).count() == 4
