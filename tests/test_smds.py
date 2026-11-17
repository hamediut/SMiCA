"""
Unit tests for SMDS calculation functions.
"""

import numpy as np
import pytest
from src.micro_gui.analysis.smds import (
    two_point_correlation,
    two_point_correlation3D,
    calculate_s2,
    calculate_s2_3d,
    calculate_s2_periodic,
    calculate_c2,
    calculate_polytopes_python,
    compute_2d_polytope_curves,
    average_curve_series,
)


class TestTwoPointCorrelation:
    """Tests for 2D two-point correlation function."""

    def test_uniform_image_all_ones(self):
        """Test correlation on uniform image of all ones."""
        img = np.ones((50, 50), dtype=np.uint8)
        result_x = two_point_correlation(img, dim=0, var=1)
        result_y = two_point_correlation(img, dim=1, var=1)

        # For uniform image of all 1s, correlation should be 1 everywhere
        assert np.allclose(result_x, 1.0)
        assert np.allclose(result_y, 1.0)

    def test_uniform_image_all_zeros(self):
        """Test correlation on uniform image of all zeros."""
        img = np.zeros((50, 50), dtype=np.uint8)
        result_x = two_point_correlation(img, dim=0, var=1)
        result_y = two_point_correlation(img, dim=1, var=1)

        # For uniform image of all 0s with var=1, correlation should be 0
        assert np.allclose(result_x, 0.0)
        assert np.allclose(result_y, 0.0)

    def test_output_shape(self):
        """Test that output shape matches input dimensions."""
        img = np.random.randint(0, 2, size=(100, 80), dtype=np.uint8)
        result_x = two_point_correlation(img, dim=0, var=1)
        result_y = two_point_correlation(img, dim=1, var=1)

        # Shape should be (y_dim, x_dim) for dim=0
        assert result_x.shape == (80, 100)
        # Shape should be (x_dim, y_dim) for dim=1
        assert result_y.shape == (100, 80)


class TestTwoPointCorrelation3D:
    """Tests for 3D two-point correlation function."""

    def test_uniform_3d_image_all_ones(self):
        """Test 3D correlation on uniform volume of all ones."""
        img = np.ones((20, 20, 20), dtype=np.uint8)
        result = two_point_correlation3D(img, dim=0, var=1)

        # For uniform image of all 1s, correlation should be 1 everywhere
        assert np.allclose(result, 1.0)

    def test_uniform_3d_image_all_zeros(self):
        """Test 3D correlation on uniform volume of all zeros."""
        img = np.zeros((20, 20, 20), dtype=np.uint8)
        result = two_point_correlation3D(img, dim=0, var=1)

        # For uniform image of all 0s with var=1, correlation should be 0
        assert np.allclose(result, 0.0)


class TestCalculateS2:
    """Tests for 2D S2 calculation."""

    def test_calculate_s2_output_size(self):
        """Test that S2 output is approximately half the input size."""
        img = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8)
        result = calculate_s2(img)

        # Result should be approximately half of input dimension
        assert result.shape[0] == 50

    def test_calculate_s2_range(self):
        """Test that S2 values are in valid range [0, 1]."""
        img = np.random.randint(0, 2, size=(50, 50), dtype=np.uint8)
        result = calculate_s2(img)

        # All S2 values should be between 0 and 1
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_calculate_s2_uniform_ones(self):
        """Test S2 on uniform image."""
        img = np.ones((50, 50), dtype=np.uint8)
        result = calculate_s2(img)

        # For uniform image of 1s, S2 should be 1
        assert np.allclose(result, 1.0)


class TestCalculateS2_3D:
    """Tests for 3D S2 calculation."""

    def test_calculate_s2_3d_output_size(self):
        """Test that 3D S2 output is approximately half the minimum dimension."""
        img = np.random.randint(0, 2, size=(40, 40, 40), dtype=np.uint8)
        result = calculate_s2_3d(img)

        # Result should be half of minimum dimension
        assert result.shape[0] == 20

    def test_calculate_s2_3d_range(self):
        """Test that 3D S2 values are in valid range [0, 1]."""
        img = np.random.randint(0, 2, size=(30, 30, 30), dtype=np.uint8)
        result = calculate_s2_3d(img)

        # All S2 values should be between 0 and 1
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_calculate_s2_3d_uniform_ones(self):
        """Test 3D S2 on uniform volume."""
        img = np.ones((20, 20, 20), dtype=np.uint8)
        result = calculate_s2_3d(img)

        # For uniform volume of 1s, S2 should be 1
        assert np.allclose(result, 1.0)


def _shift(a, dr, dc):
    """a[(r + dr) % n_rows, (c + dc) % n_cols] at every (r, c): rows and columns wrap separately."""
    return np.roll(a, (-dr, -dc), axis=(0, 1))


class TestNonSquare2D:
    """2D SMDs on non-square images: periodic, each axis wrapping at its own length, Nt = min(shape) // 2."""

    SHAPE = (23, 34)  # rectangular on purpose; nt = 11
    ALL_2D = ['s2', 'c2', 'L', 'p3h', 'p3v', 'p4', 'p6']

    @pytest.fixture
    def image(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 2, size=self.SHAPE, dtype=np.uint8)

    def test_all_functions_accept_non_square_image(self, image):
        """Every 2D function runs on a rectangular image and returns min(shape) // 2 points."""
        raw, scaled = compute_2d_polytope_curves(image, self.ALL_2D)
        nt = min(self.SHAPE) // 2
        for name in self.ALL_2D:
            assert raw[name].shape == (nt, 2), name
            assert scaled[name].shape == (nt, 2), name
            assert np.array_equal(raw[name][:, 0], np.arange(nt)), name

    def test_value_at_r0_is_phase_fraction(self, image):
        """No padding: every raw curve starts exactly at the phase fraction, every scaled curve at 1."""
        raw, scaled = compute_2d_polytope_curves(image, self.ALL_2D)
        for name in self.ALL_2D:
            assert raw[name][0, 1] == pytest.approx(image.mean(), abs=1e-12), name
            assert scaled[name][0, 1] == pytest.approx(1.0, abs=1e-12), name

    def test_s2_c2_l_p4_invariant_under_transpose(self, image):
        """Rows and columns are averaged with equal weight, so transposing changes nothing."""
        names = ['s2', 'c2', 'L', 'p4']
        raw, _ = compute_2d_polytope_curves(image, names)
        raw_t, _ = compute_2d_polytope_curves(image.T.copy(), names)
        for name in names:
            np.testing.assert_allclose(raw[name], raw_t[name], atol=1e-12, err_msg=name)

    def test_p3h_is_p3v_of_transpose(self, image):
        """P3H and P3V are mirror images: swapping rows and columns turns one into the other."""
        raw, _ = calculate_polytopes_python(image, polytopes=('p3h',))
        raw_t, _ = calculate_polytopes_python(image.T.copy(), polytopes=('p3v',))
        np.testing.assert_allclose(raw['p3h'], raw_t['p3v'], atol=1e-12)

    def test_s2_and_p4_match_np_roll_reference(self, image):
        """Independent check of the per-axis periodic wrap on a rectangular image."""
        nt = min(self.SHAPE) // 2
        b = image.astype(float)
        s2_ref = [(np.mean(b * _shift(b, 0, r)) + np.mean(b * _shift(b, r, 0))) / 2 for r in range(nt)]
        p4_ref = [np.mean(b * _shift(b, 0, R) * _shift(b, R, 0) * _shift(b, R, R)) for R in range(nt)]
        raw, _ = calculate_polytopes_python(image, polytopes=('s2', 'p4'))
        np.testing.assert_allclose(calculate_s2_periodic(image), s2_ref, atol=1e-12)
        np.testing.assert_allclose(raw['s2'][:, 1], s2_ref, atol=1e-12)
        np.testing.assert_allclose(raw['p4'][:, 1], p4_ref, atol=1e-12)

    def test_c2_never_exceeds_s2(self, image):
        """Same cluster implies same phase, so C2(r) <= S2(r) at every r."""
        assert np.all(calculate_c2(image) <= calculate_s2_periodic(image) + 1e-12)

    def test_fully_foreground_row_counts_as_a_ring_in_L(self):
        """A periodic row that is entirely phase holds a segment of any length at each of its positions."""
        img = np.zeros((8, 13), dtype=np.uint8)
        img[2, :] = 1  # the only foreground: one complete row
        raw, _ = calculate_polytopes_python(img, polytopes=('L',))
        # rows: 13 positions for every r; columns: only isolated pixels (r = 0) -> averaged over 2 * 8 * 13 trials
        expected = np.full(4, 13 / (2 * 8 * 13))
        expected[0] = (13 + 13) / (2 * 8 * 13)
        np.testing.assert_allclose(raw['L'][:, 1], expected, atol=1e-12)


class TestAverageCurveSeries:
    """Average of one function's curves over slices (the slice evolution "Average" tab)."""

    def test_mean_per_r_skipping_nan(self):
        """Mean point by point in r; a NaN on one slice is skipped, all-NaN stays NaN."""
        r = np.arange(3.0)
        slice_a = np.column_stack((r, [1.0, 2.0, np.nan]))
        slice_b = np.column_stack((r, [3.0, 4.0, np.nan]))
        slice_c = np.column_stack((r, [5.0, np.nan, np.nan]))
        average = average_curve_series([slice_a, slice_b, slice_c])
        np.testing.assert_array_equal(average[:, 0], r)
        np.testing.assert_allclose(average[:2, 1], [3.0, 3.0])  # r=1: mean of 2 and 4 only
        assert np.isnan(average[2, 1])

    def test_average_of_slices_matches_mean_of_curves(self):
        """On real per-slice SMD curves, the average equals the plain mean over slices."""
        rng = np.random.default_rng(1)
        volume = rng.integers(0, 2, size=(4, 21, 30), dtype=np.uint8)  # 4 non-square slices
        raws = [compute_2d_polytope_curves(s, ['s2', 'p4'])[0] for s in volume]
        for name in ('s2', 'p4'):
            average = average_curve_series([raw[name] for raw in raws])
            expected = np.mean([raw[name][:, 1] for raw in raws], axis=0)
            np.testing.assert_allclose(average[:, 1], expected)
            assert average.shape == (10, 2)  # min(21, 30) // 2 points


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
