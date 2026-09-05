from app.services.datasets import content_hash, slice_tags


def test_hash_sensitive_to_order_text_and_labels():
    rows = [(0, "good", 1), (1, "bad", 0)]
    assert content_hash(rows) == content_hash(rows)
    assert content_hash(rows) != content_hash(rows[::-1])
    assert content_hash(rows) != content_hash([(0, "good", 0), (1, "bad", 0)])


def test_slice_boundaries():
    assert "len_long" not in slice_tags("word " * 25)
    assert "len_long" in slice_tags("word " * 26)
    assert "has_negation" in slice_tags("It wasn't good, but never boring")
    assert "has_contrast" in slice_tags("It wasn't good, but never boring")
    assert slice_tags("Nobody buys a notebook") == []
