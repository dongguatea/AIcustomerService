from app.db.models import Faq
from app.db.seed import seed
from app.repositories import search_faq


def test_seed_contains_return_policy_and_not_shipping_fee(session_factory):
    with session_factory() as s:
        seed(s)
        assert search_faq(s, "退货")
        assert search_faq(s, "邮费") == []
        assert search_faq(s, "运费") == []
        assert s.query(Faq).count() >= 4
