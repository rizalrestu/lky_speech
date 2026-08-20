"""Self-checks for the pure logic. No Qdrant, no Ollama."""
from types import SimpleNamespace

import numpy as np

import core
from index import point_id
from nas_speech.pipelines import safe


class FakeClient:
    """Returns fixed hits, ignoring the query."""
    def __init__(self, hits):
        self.hits = hits
        self.kwargs = None

    def query_points(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(points=self.hits)


def hit(uid, score):
    return SimpleNamespace(score=score, payload={"uid": uid, "text": uid, "title": "t",
                                                 "speaker": "s", "date": "d"})


def test_retrieve_caps_chunks_per_document():
    embedder = SimpleNamespace(encode=lambda xs, **kw: np.zeros((1, 3)))
    client = FakeClient([hit("a", 0.9), hit("a", 0.8), hit("a", 0.7),
                         hit("b", 0.6), hit("c", 0.5), hit("d", 0.4)])
    uids = [h.payload["uid"] for h in core.retrieve("q", embedder, client)]
    assert uids == ["a", "a", "b", "c", "d"], uids   # 3rd "a" dropped
    assert client.kwargs["score_threshold"] == core.SCORE_THRESHOLD
    assert client.kwargs["limit"] == core.FETCH_K


def test_retrieve_returns_nothing_when_all_below_threshold():
    embedder = SimpleNamespace(encode=lambda xs, **kw: np.zeros((1, 3)))
    assert core.retrieve("q", embedder, FakeClient([])) == []


def test_answer_question_refuses_without_context():
    embedder = SimpleNamespace(encode=lambda xs, **kw: np.zeros((1, 3)))
    answer, hits = core.answer_question("q", embedder, FakeClient([]))
    assert answer == core.NO_RECORDS and hits == []
    assert core.NO_RECORDS in core.SYSTEM_PROMPT   # one refusal string, not three


def test_point_id_is_content_addressed():
    assert point_id("x_y_0") == point_id("x_y_0")
    assert point_id("x_y_0") != point_id("x_y_1")
    # inserting a chunk must not renumber its neighbours
    before = [point_id(c) for c in ["a_0", "c_0"]]
    after = [point_id(c) for c in ["a_0", "b_0", "c_0"]]
    assert before == [after[0], after[2]]


def test_safe_filenames_stay_inside_the_store():
    assert "/" not in safe("../../etc/passwd") and "\\" not in safe(r"..\..\win.ini")
    assert not safe("../evil").startswith(".")
    assert safe("speech-1965.pdf") == "speech-1965.pdf"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
