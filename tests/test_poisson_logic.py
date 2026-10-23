import math
import pytest
from scipy.stats import poisson

def test_poisson_tail_reference_example():
    # P(X >= 30) for a Poisson distribution with lambda=20 is about 2.18%.
    assert poisson.sf(29, 20) == pytest.approx(0.0218, abs=0.0002)

def test_burst_threshold_flags_high_count():
    assert poisson.sf(47, 20) < 0.01

def test_ordinary_count_is_not_burst_at_one_percent():
    assert poisson.sf(24, 20) >= 0.01

def test_probability_bounds():
    p = poisson.sf(29, 20)
    assert 0 <= p <= 1
