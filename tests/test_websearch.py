"""Web search learning unit tests (offline-safe where possible)."""

from carvustrain.websearch import WebLearner, _sentences_from_text, _strip_html


def test_strip_html():
    raw = "<html><script>evil()</script><body><p>Hello world. This is a longer sentence for testing.</p></body></html>"
    text = _strip_html(raw)
    assert "evil" not in text
    assert "Hello world" in text


def test_sentences_from_text():
    text = (
        "Photosynthesis converts light energy into chemical energy in plants. "
        "Cellular respiration releases energy from food molecules. "
        "Click here to subscribe now!"
    )
    facts = _sentences_from_text(text, max_sentences=10)
    assert any("Photosynthesis" in f for f in facts)
    assert not any("subscribe" in f.lower() for f in facts)


def test_web_learner_ingest_manual(monkeypatch):
    from carvustrain import KnowledgeBase
    import carvustrain.websearch as ws

    def fake_learn(query, **kwargs):
        return (
            [f"Fact about {query}.", f"Another fact about {query}."],
            [
                {"category": "web", "source": "https://example.com", "query": query, "title": "T"},
                {"category": "web", "source": "https://example.com", "query": query, "title": "T"},
            ],
        )

    monkeypatch.setattr(ws, "learn_from_search", fake_learn)
    kb = KnowledgeBase()
    learner = WebLearner(kb)
    added = learner.learn("gravity")
    assert added == 2
    assert len(kb) == 2
