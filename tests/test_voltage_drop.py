from app.tools import voltage_drop


def test_voltage_drop_matches_hand_calculation_and_flags_limits():
    ten = voltage_drop.invoke({"current_a": 32, "length_m": 60, "cable_mm2": 10, "supply_v": 400, "phases": 3})
    six = voltage_drop.invoke({"current_a": 32, "length_m": 60, "cable_mm2": 6, "supply_v": 400, "phases": 3})
    assert "6.09 V = 1.52%" in ten and "within the common 3%" in ten
    assert "10.24 V = 2.56%" in six  # smaller cable -> larger drop
    one_ph = voltage_drop.invoke({"current_a": 18, "length_m": 25, "cable_mm2": 2.5, "supply_v": 230, "phases": 1})
    assert "6.67 V" in one_ph
    assert voltage_drop.invoke({"current_a": 1, "length_m": 1, "cable_mm2": 7, "supply_v": 230}).startswith("Error")
