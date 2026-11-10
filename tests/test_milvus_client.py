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


class FakeClient:
    def __init__(self):
        self._cols = {}
        self.upserts = []
        self.searches = []
        self.dropped = []

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

    def upsert(self, name, rows):
        self.upserts.append((name, rows))

    def search(self, name, data, anns_field, limit, output_fields, search_params, filter):
        self.searches.append(dict(name=name, anns_field=anns_field, limit=limit,
                                  output_fields=output_fields, search_params=search_params,
                                  filter=filter))
        return [[{"id": 1, "distance": 0.87,
                  "entity": {"question": "q0", "answer": "a0", "section_path": "",
                             "content_type": "faq", "category": "商品"}}]]

    def query(self, name, filter, output_fields):
        return [{"count(*)": 7}]

    def drop_collection(self, name):
        self.dropped.append(name)
        self._cols.pop(name, None)


def test_ensure_collection_builds_dense_cosine_schema():
    client = FakeClient()
    mc.ensure_collection(client, "c")
    mc.ensure_collection(client, "c")  # 幂等
    schema = client._cols["c"]["schema"]
    assert [(n, d) for n, d, _ in schema.fields] == [
        ("id", "INT64"), ("dense", "FLOAT_VECTOR"), ("text", "VARCHAR"),
        ("sparse", "SPARSE_FLOAT_VECTOR"), ("question", "VARCHAR"),
        ("answer", "VARCHAR"), ("section_path", "VARCHAR"),
        ("content_type", "VARCHAR"), ("category", "VARCHAR"),
    ]
    assert client._cols["c"]["index_params"].indexes == [
        {"field_name": "dense", "index_type": "AUTOINDEX", "metric_type": "COSINE"},
        {"field_name": "sparse", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
    ]
    assert client._cols["c"]["loaded"] is True


def test_upsert_and_dense_search_mapping():
    client = FakeClient()
    mc.ensure_collection(client, "c")
    row = {"id": 1, "dense": [0.0] * mc.DIM, "question": "q0", "answer": "a0",
           "section_path": "", "content_type": "faq", "category": "商品"}
    mc.upsert_vectors(client, [row], "c")
    hits = mc.dense_search(client, [0.0] * mc.DIM, top_k=5, category="商品", collection="c")
    assert hits == [{"id": 1, "score": 0.87, "question": "q0", "answer": "a0",
                     "section_path": "", "content_type": "faq", "category": "商品"}]
    s = client.searches[-1]
    assert s["anns_field"] == "dense" and s["limit"] == 5
    assert s["search_params"] == {"metric_type": "COSINE"}
    assert s["filter"] == 'category == "商品"'


def test_dense_search_no_category_uses_empty_filter():
    client = FakeClient()
    mc.ensure_collection(client, "c")
    mc.dense_search(client, [0.0] * mc.DIM, top_k=3, collection="c")
    assert client.searches[-1]["filter"] == ""


def test_count_and_drop():
    client = FakeClient()
    mc.ensure_collection(client, "c")
    assert mc.count(client, "c") == 7
    mc.drop(client, "c")
    assert client.dropped == ["c"] and not client.has_collection("c")
