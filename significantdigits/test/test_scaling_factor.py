import significantdigits as sd
from significantdigits._significantdigits import _compute_scaling_factor
import numpy as np


def test_compute_scaling_factor():
    """
    Test _compute_scaling_factor function with various inputs
    Tests:
    1. 1D array, reference_is_random_variable=False
    2. 1D array, reference_is_random_variable=True
    3. Array with zero values
    4. 2D array, different axes
    5. Negative values
    """
    # Test 1: 1D array, reference_is_random_variable=False
    y1 = np.array([1.0, 2.0, 4.0, 8.0])
    expected1 = np.array([1, 2, 3, 4])  # floor(log2(|y|)) + 1
    result1 = _compute_scaling_factor(y1, axis=0, reference_is_random_variable=False)
    assert np.all(np.equal(result1, expected1))

    # Test 2: 1D array, reference_is_random_variable=True
    y2 = np.array([1.0, 2.0, 4.0, 8.0])
    # Mean of y2 is 3.75, floor(log2(3.75)) + 1 = 1 + 1 = 2
    expected2 = 2
    result2 = _compute_scaling_factor(y2, axis=0, reference_is_random_variable=True)
    assert np.all(np.equal(result2, expected2))

    # Test 3: Array with zero values
    y3 = np.array([0.0, 2.0, 4.0, 0.0])
    expected3 = np.array([1, 2, 3, 1])  # zeros replaced with 0 + 1 = 1
    result3 = _compute_scaling_factor(y3, axis=0, reference_is_random_variable=False)
    assert np.all(np.equal(result3, expected3))

    # Test 4a: 2D array, axis=0
    y4 = np.array([[1.0, 2.0], [4.0, 8.0]])
    expected4 = np.array([2, 3])  # floor(log2(|[2.5, 5.0]|)) + 1
    result4 = _compute_scaling_factor(y4, axis=0, reference_is_random_variable=True)
    assert np.all(np.equal(result4, expected4))

    # Test 4b: 2D array, axis=1
    y5 = np.array([[1.0, 2.0], [4.0, 8.0]])
    expected5 = np.array([1, 3])  # floor(log2(|[1.5, 6.0]|)) + 1
    result5 = _compute_scaling_factor(y5, axis=1, reference_is_random_variable=True)
    assert np.all(np.equal(result5, expected5))

    # Test 5: Negative values
    y6 = np.array([-1.0, -2.0, -4.0, -8.0])
    expected6 = np.array([1, 2, 3, 4])  # floor(log2(|y|)) + 1
    result6 = _compute_scaling_factor(y6, axis=0, reference_is_random_variable=False)
    assert np.all(np.equal(result6, expected6))

    # Test 6: 3D array
    y7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    # Mean along axis=0: [[3.0, 4.0], [5.0, 6.0]]
    # floor(log2(|[3.0, 4.0, 5.0, 6.0]|)) + 1 = [2, 3, 3, 3]
    expected7 = np.array([[2, 3], [3, 3]])
    result7 = _compute_scaling_factor(y7, axis=0, reference_is_random_variable=True)
    assert np.all(np.equal(result7, expected7))

    # Test 7: All zeros array
    y8 = np.zeros(4)
    expected8 = np.ones(4)  # All zeros become 1
    result8 = _compute_scaling_factor(y8, axis=0, reference_is_random_variable=False)
    assert np.all(np.equal(result8, expected8))
