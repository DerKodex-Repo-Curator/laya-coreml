from benchmarks.common import public_differences


def test_public_decimal_boundary_and_actual_overrun():
    assert not public_differences({"p": 0.2}, {"p": 0.1999}, 1e-4)
    assert public_differences({"p": 0.2}, {"p": 0.1998}, 1e-4)
    assert public_differences({"p": 0.2}, {"p": float("nan")}, 1e-4)
    assert public_differences({"count": 0}, {"count": False}, 1e-4)
