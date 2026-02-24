from src.metrics import inversions_count, kendall_tau_distance, success_at_n

def test_inversions():
    assert inversions_count([0,1,2,3]) == 0
    assert inversions_count([3,2,1,0]) == 6

def test_kendall():
    assert kendall_tau_distance([0,1,2]) == 0
    assert kendall_tau_distance([2,1,0]) == 3

def test_success():
    assert success_at_n([True, False, True]) == 2/3
