import pytest

from freightwatch.pipeline import run, write_outputs
from freightwatch.qa import answer


@pytest.fixture(scope="module")
def out_csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("out")
    write_outputs(*run(use_llm=False), out_dir=d)
    return d / "output.csv"


def test_unflagged_route_is_not_answered_with_another_routes_rows(out_csv):
    a = answer("why did Kolkata-Bhubaneswar get pricier in March?", out_path=out_csv)
    assert a.startswith("No finding for Kolkata-Bhubaneswar") and "Chennai" not in a


def test_flood_question_cites_the_flood_note(out_csv):
    a = answer("why did Chennai-Bangalore get pricier in March 2025?", out_path=out_csv)
    assert "N001" in a


def test_offtopic_question_gets_no_finding(out_csv):
    assert answer("tell me a joke", out_path=out_csv).startswith("No finding")
