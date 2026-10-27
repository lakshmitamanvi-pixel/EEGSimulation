"""
tests.py - Sanity tests for event-triggered encoders.
Tests:
1. Constant signal -> 0 events after the first.
2. For fixed delta, max |x - xh| < delta + 1e-9 at every sample.
3. Smaller delta -> events never decrease.
Prints PASS/FAIL for each.
"""
import sys
import numpy as np
from encoders import encode_fixed, encode_adaptive
from config import SWEEP_VALUES


def test_constant_signal() -> bool:
    """Test 1: Constant signal -> 0 events after the first."""
    fs = 256.0
    x_const = np.full(2000, 15.0)

    # For fixed encoder with an arbitrary delta or standard scaling
    res_fixed = encode_fixed(x_const, fs, c=0.2)
    events_after_first_fixed = res_fixed["num_events"] - 1

    # For adaptive encoder
    res_adapt = encode_adaptive(x_const, fs, k=0.2)
    events_after_first_adapt = res_adapt["num_events"] - 1

    passed = (events_after_first_fixed == 0) and (events_after_first_adapt == 0)
    print(f"Test 1 [Constant signal -> 0 events after the first]: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"   Details: fixed events after first={events_after_first_fixed}, adaptive={events_after_first_adapt}")
    return passed


def test_fixed_delta_bound() -> bool:
    """Test 2: For fixed delta, max |x - xh| < delta + 1e-9 at every sample."""
    fs = 256.0
    np.random.seed(42)
    # Generate arbitrary test signals: sine, random walk, white noise
    t = np.linspace(0, 10, int(10 * fs))
    test_signals = [
        np.sin(2 * np.pi * 5 * t) * 50.0,
        np.cumsum(np.random.randn(len(t))),
        np.random.randn(len(t)) * 20.0,
    ]

    all_passed = True
    for x in test_signals:
        for c in SWEEP_VALUES:
            res = encode_fixed(x, fs, c=c)
            xh = res["xh"]
            delta = res["delta"]
            max_err = float(np.max(np.abs(x - xh)))
            if max_err >= delta + 1e-9:
                all_passed = False
                print(f"   Violation: c={c}, delta={delta:.6f}, max_err={max_err:.6f}")

    print(f"Test 2 [For fixed delta, max |x - xh| < delta + 1e-9 at every sample]: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_smaller_delta_monotonicity() -> bool:
    """Test 3: Smaller delta -> events never decrease."""
    fs = 256.0
    np.random.seed(123)
    t = np.linspace(0, 10, int(10 * fs))
    x = np.sin(2 * np.pi * 3 * t) * 40.0 + np.random.randn(len(t)) * 5.0

    # SWEEP_VALUES is [0.05, 0.1, 0.2, 0.3, 0.5, 0.8] (increasing threshold factor c)
    # Smaller c means smaller delta, which must result in >= events.
    # Sorted from smallest c to largest c:
    c_ascending = sorted(SWEEP_VALUES)
    fixed_counts = []
    for c in c_ascending:
        res = encode_fixed(x, fs, c=c)
        fixed_counts.append(res["num_events"])

    # For c ascending (delta increasing), num_events should be monotonically non-increasing
    # i.e., as delta gets smaller (reverse order), events never decrease
    passed = True
    for i in range(len(fixed_counts) - 1):
        if fixed_counts[i] < fixed_counts[i + 1]:
            passed = False
            print(f"   Violation: c={c_ascending[i]} gave {fixed_counts[i]} < c={c_ascending[i+1]} gave {fixed_counts[i+1]}")

    print(f"Test 3 [Smaller delta -> events never decrease]: {'PASS' if passed else 'FAIL'}")
    return passed


def run_all_tests() -> bool:
    """Run all sanity tests and report overall status."""
    print("=" * 60)
    print("RUNNING ENCODER SANITY TESTS")
    print("=" * 60)
    t1 = test_constant_signal()
    t2 = test_fixed_delta_bound()
    t3 = test_smaller_delta_monotonicity()
    all_ok = t1 and t2 and t3
    print("=" * 60)
    print(f"OVERALL SANITY TESTS RESULT: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
