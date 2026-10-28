import asyncio

from app.core import embeddings


class FakeEmbeddings:
    def __init__(self, vectors):
        self._vectors = vectors
        self.calls = []
        self.embeddings = self

    async def create(self, *, model, input):
        self.calls.append((model, input))
        data = [type("D", (), {"embedding": self._vectors[i]})() for i in range(len(input))]
        return type("R", (), {"data": data})()


def test_embed_texts_uses_model_and_returns_vectors(monkeypatch):
    vec = [0.1, 0.2, 0.3]
    fake = FakeEmbeddings([vec, [0.4, 0.5, 0.6]])
    monkeypatch.setattr(embeddings, "_CLIENT", fake)
    out = asyncio.run(embeddings.embed_texts(["你好", "再见"]))
    assert fake.calls[0][0] == "BAAI/bge-m3"
    assert fake.calls[0][1] == ["你好", "再见"]
    assert out == [vec, [0.4, 0.5, 0.6]]
    assert asyncio.run(embeddings.embed_query("你好")) == vec
