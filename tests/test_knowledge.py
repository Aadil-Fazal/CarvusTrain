"""Curriculum knowledge and learning feature tests."""

from carvustrain import KnowledgeBase, Model, knowledge


def test_load_curriculum():
    model = Model(name="CurriculumTest", auto_load=False)
    loaded = model.load_curriculum()
    assert loaded["grammar"] > 0
    assert loaded["math"] > 0
    assert loaded["science"] > 0
    assert len(model.knowledge_base) == sum(loaded.values())


def test_math_and_science_search():
    kb = KnowledgeBase()
    kb.load_math_knowledge()
    kb.load_science_knowledge()
    math_hits = kb.search_math("quadratic formula")
    science_hits = kb.search_science("Newton second law")
    assert math_hits
    assert science_hits


def test_grammar_still_works():
    kb = KnowledgeBase()
    n = kb.load_grammar_knowledge()
    assert n > 0
    hits = kb.search_grammar("Nouns represent people", topic="parts_of_speech")
    assert hits
    assert any("Noun" in h[0] for h in hits)


def test_builtin_topic_counts():
    topics = knowledge.all_builtin_topics()
    assert topics["grammar"] > 0
    assert topics["math"] > 0
    assert topics["science"] > 0


def test_learn_and_ask():
    model = Model(name="AskTest", auto_load=False)
    model.learn("CarvusTrain teaches math, science, and English grammar.")
    answer = model.ask("What does CarvusTrain teach?")
    assert isinstance(answer, str)
    assert len(answer) > 0
