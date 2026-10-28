from app.kb import chunking


def test_overlap_is_whole_trailing_sentence():
    chunks = ["前面很多内容。中间一句话。最后收尾句。", "下一块正文。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=6)
    assert out[0] == chunks[0]
    assert out[1] == "最后收尾句。下一块正文。"


def test_overlap_takes_whole_sentence_even_if_over_size():
    chunks = ["这是一个非常非常非常长的句子没有中间标点结尾才有。", "新块。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=5)
    assert out[1] == "这是一个非常非常非常长的句子没有中间标点结尾才有。新块。"


def test_empty_and_single():
    assert chunking.apply_sentence_overlap([], 5) == []
    assert chunking.apply_sentence_overlap(["只有一块。"], 5) == ["只有一块。"]
