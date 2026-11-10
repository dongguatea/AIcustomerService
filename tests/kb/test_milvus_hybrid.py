from pymilvus import FunctionType, RRFRanker

from app.kb import milvus_client as mc


class FakeSchema:
    def __init__(self):
        self.fields = []
        self.functions = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, getattr(dtype, "name", str(dtype)), kwargs))

    def add_function(self, fn):
        self.functions.append(fn)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


def _hit(id, distance=0.8, question="q", answer="a", section_path="s",
         content_type="faq", category="c"):
    return {"id": id, "distance": distance, "entity": {
        "question": question, "answer": answer, "section_path": section_path,
        "content_type": content_type, "category": category}}


class FakeClient:
    def __init__(self):
        self._cols = {}
        self.searches = []
        self.hybrid_calls = []
        self._search_result = [[_hit(1)]]
        self._hybrid_result = [[_hit(1)]]

    def has_collection(self, name):
        return name in self._cols

    def load_collection(self, name):
        self._cols[name]["loaded"] = True

    def create_schema(self, auto_id=False):
        return FakeSchema()

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, name, schema=None, index_params=None):
        self._cols[name] = {"schema": schema, "index_params": index_params, "loaded": False}

    def search(self, name, data, anns_field, limit, output_fields, search_params, filter):
        self.searches.append(dict(name=name, data=data, anns_field=anns_field, limit=limit,
                                  output_fields=output_fields, search_params=search_params,
                                  filter=filter))
        return self._search_result

    def hybrid_search(self, name, reqs, ranker, limit, output_fields):
        self.hybrid_calls.append(dict(name=name, reqs=reqs, ranker=ranker,
                                      limit=limit, output_fields=output_fields))
        return self._hybrid_result

    def drop_collection(self, name):
        self._cols.pop(name, None)


def test_ensure_collection_builds_bm25_schema():
    client = FakeClient()
    mc.ensure_collection(client, "c")
    schema = client._cols["c"]["schema"]
    assert [(n, d) for n, d, _ in schema.fields] == [
        ("id", "INT64"), ("dense", "FLOAT_VECTOR"), ("text", "VARCHAR"),
        ("sparse", "SPARSE_FLOAT_VECTOR"), ("question", "VARCHAR"),
        ("answer", "VARCHAR"), ("section_path", "VARCHAR"),
        ("content_type", "VARCHAR"), ("category", "VARCHAR"),
    ]
    text_kwargs = [k for n, d, k in schema.fields if n == "text"][0]
    assert text_kwargs.get("enable_analyzer") is True
    assert text_kwargs.get("analyzer_params") == {"type": "chinese"}
    fn = schema.functions[0]
    assert fn.name == "text_bm25"
    assert fn.input_field_names == ["text"]
    assert fn.output_field_names == ["sparse"]
    assert fn.type == FunctionType.BM25
    assert client._cols["c"]["index_params"].indexes == [
        {"field_name": "dense", "index_type": "AUTOINDEX", "metric_type": "COSINE"},
        {"field_name": "sparse", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
    ]


def test_bm25_hits_model_number():
    client = FakeClient()
    mc.ensure_collection(client, "c")
    client._search_result = [[_hit(2, section_path="商品手册 / 猫砂盆")]]
    hits = mc.bm25_search(client, "猫砂盆 Pro 型号", top_k=2, collection="c")
    assert hits[0]["id"] == 2
    assert hits[0]["section_path"] == "商品手册 / 猫砂盆"
    s = client.searches[-1]
    assert s["anns_field"] == "sparse"
    assert s["search_params"] == {"metric_type": "BM25"}


def test_hybrid_returns_rrf_hits():
    client = FakeClient()
    mc.ensure_collection(client, "c")
    client._hybrid_result = [[_hit(1), _hit(2)]]
    hits = mc.hybrid_search(client, [0.0] * mc.DIM, "运费", top_k=2, recall=50, collection="c")
    assert [h["id"] for h in hits] == [1, 2]
    assert {"id", "score", "question", "answer", "section_path", "content_type", "category"} <= hits[0].keys()
    call = client.hybrid_calls[-1]
    assert isinstance(call["ranker"], RRFRanker)
    assert [r.anns_field for r in call["reqs"]] == ["dense", "sparse"]
    assert call["limit"] == 2


def test_category_filter_applies_to_bm25_and_hybrid():
    client = FakeClient()
    mc.ensure_collection(client, "c")
    mc.bm25_search(client, "型号 运费", top_k=5, category="运费", collection="c")
    assert client.searches[-1]["filter"] == 'category == "运费"'
    mc.hybrid_search(client, [0.0] * mc.DIM, "运费", top_k=2, category="运费", collection="c")
    call = client.hybrid_calls[-1]
    assert all(r.expr == 'category == "运费"' for r in call["reqs"])
