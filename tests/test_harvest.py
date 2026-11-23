import pytest

from frsbot.scrape.harvest import select_entries


def test_selection_all_boards_unique_and_owner_excluded_everywhere():
    def e(rid, **kw):
        return {"replay": rid, "shared": True, "name": "other", **kw}
    boards = [{"entries": [e("a"), e("b")]}, {"responses": [{"data": {
        "entries": [e("a")], "duos": [e("b", names=["other", " 404 "]), e("c", teams=[254, 2910])],
        "rows": [e("private", shared=False)]}}]}]
    keep, excluded = select_entries(boards)
    assert {r["replay"] for r in keep} == {"a", "c"}
    assert excluded == {"b"}
    assert select_entries([]) == ([], set())
    with pytest.raises(ValueError, match="invalid replay"):
        select_entries([{"entries": [e("../bad")]}])
