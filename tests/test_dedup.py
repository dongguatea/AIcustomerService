from app.kb.dedup import dedupe, normalize_question


class Item:
    def __init__(self, question):
        self.question = question


def test_normalize_strips_punct_and_space():
    assert normalize_question(" 满 99 元包邮? ") == "满99元包邮"


def test_dedupe_removes_internal_duplicates():
    kept, discarded = dedupe(
        [Item("运费怎么算"), Item("运费怎么算?"), Item("多久发货")], []
    )
    assert [x.question for x in kept] == ["运费怎么算", "多久发货"]
    assert [x.question for x in discarded] == ["运费怎么算?"]


def test_dedupe_drops_existing_and_empty():
    kept, discarded = dedupe([Item("满99包邮"), Item("  ")], ["满99包邮"])
    assert kept == []
    assert len(discarded) == 2
