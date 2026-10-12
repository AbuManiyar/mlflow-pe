from pathlib import Path

from signet_experiments.splits import create_pairs, split_person_ids


def test_person_split_is_reproducible_and_disjoint():
    ids = [f"person-{index}" for index in range(20)]

    first = split_person_ids(ids, seed=19)
    second = split_person_ids(reversed(ids), seed=19)

    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]).isdisjoint(first[2])
    assert set(first[1]).isdisjoint(first[2])
    assert set().union(*map(set, first)) == set(ids)


def test_split_rejects_fractions_without_training_people():
    try:
        split_person_ids(["a", "b", "c"], validation_fraction=0.5, test_fraction=0.5)
    except ValueError as error:
        assert "sum to less than 1" in str(error)
    else:
        raise AssertionError("Invalid fractions should be rejected.")


def test_pair_generation_uses_expected_labels_and_relative_paths(tmp_path: Path):
    (tmp_path / "001").mkdir()
    (tmp_path / "001_forg").mkdir()
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / "001" / name).touch()
    (tmp_path / "001_forg" / "forged.png").touch()

    pairs = create_pairs(tmp_path, ["001"])

    assert pairs == [
        ("001/a.jpg", "001/b.jpg", 0),
        ("001/a.jpg", "001_forg/forged.png", 1),
        ("001/b.jpg", "001_forg/forged.png", 1),
    ]
