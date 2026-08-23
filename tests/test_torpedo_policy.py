from bot import torpedo_evasion_threat


def test_torpedo_evasion_requires_destroyer_or_cruiser_within_ten_km():
    assert torpedo_evasion_threat(True, 9.8, {"destroyer"})
    assert torpedo_evasion_threat(True, 10.0, {"cruiser"})
    assert not torpedo_evasion_threat(True, 10.1, {"destroyer"})
    assert not torpedo_evasion_threat(True, 8.0, {"battleship"})
    assert not torpedo_evasion_threat(False, 8.0, {"cruiser"})
    assert not torpedo_evasion_threat(True, 8.0, set())
