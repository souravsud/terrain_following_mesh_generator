"""Tests for roughness map preparation via WorldCover land-use classification.

These tests verify that:
- The WORLDCOVER_Z0_LOOKUP table maps every known ESA WorldCover class code to a
  physically plausible z0 value.
- prepare_roughness_from_worldcover() correctly translates a class-code map to a
  z0 roughness map using the lookup table.
- Unknown / unclassified pixels become NaN in the output.
- A diagnostic summary (classes present and their z0 values) can be produced for
  any WorldCover map to aid debugging when roughness values look suspicious.
"""

import numpy as np
import pytest

from terrain_mesh.utils import (
    WORLDCOVER_CLASS_NAMES,
    WORLDCOVER_Z0_LOOKUP,
    prepare_roughness_from_worldcover,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise_roughness_map(worldcover_data: np.ndarray, z0_map: np.ndarray) -> None:
    """Print a diagnostic table of classes found in *worldcover_data*.

    This helper is intentionally verbose so that it can be called during test
    development or attached to a failing test to understand what values were
    seen vs. expected.

    Args:
        worldcover_data: 2-D integer array of WorldCover class codes.
        z0_map: Corresponding z0 roughness array produced by
            :func:`prepare_roughness_from_worldcover`.
    """
    print("\n--- WorldCover classification summary ---")
    print(f"{'Code':>5}  {'Name':<30}  {'Lookup z0 (m)':>14}  {'Pixels':>8}  {'Map z0 min':>12}  {'Map z0 max':>12}")
    print("-" * 90)

    unique_codes = np.unique(worldcover_data)
    for code in unique_codes.astype(int):
        mask = worldcover_data == code
        name = WORLDCOVER_CLASS_NAMES.get(code, "Unknown")
        lookup_z0 = WORLDCOVER_Z0_LOOKUP.get(code, float("nan"))
        z0_pixels = z0_map[mask]
        valid = z0_pixels[~np.isnan(z0_pixels)]
        if valid.size:
            print(
                f"{code:>5}  {name:<30}  {lookup_z0:>14.6f}  {mask.sum():>8}  "
                f"{valid.min():>12.6f}  {valid.max():>12.6f}"
            )
        else:
            print(f"{code:>5}  {name:<30}  {lookup_z0:>14.6f}  {mask.sum():>8}  {'NaN':>12}  {'NaN':>12}")

    unrecognised = np.sum(np.isnan(z0_map))
    if unrecognised:
        print(f"\n  {unrecognised} pixel(s) are NaN in z0_map (unrecognised class codes).")

    print("\n--- Lookup table (full) ---")
    print(f"{'Code':>5}  {'Name':<30}  {'z0 (m)':>10}")
    print("-" * 50)
    for code, z0 in sorted(WORLDCOVER_Z0_LOOKUP.items()):
        print(f"{code:>5}  {WORLDCOVER_CLASS_NAMES[code]:<30}  {z0:>10.6f}")
    print()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_worldcover_map():
    """3×4 WorldCover class array covering a representative subset of classes."""
    return np.array(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 95, 100, 10],
        ],
        dtype=np.int32,
    )


@pytest.fixture
def mixed_map_with_unknown():
    """5×5 map that contains one unknown class code (255) and one repeated class."""
    data = np.array(
        [
            [10, 10, 30, 30, 30],
            [10, 40, 40, 80, 80],
            [60, 60, 60, 80, 80],
            [255, 255, 60, 80, 10],  # 255 is not a valid WorldCover code
            [30, 30, 30, 30, 30],
        ],
        dtype=np.int32,
    )
    return data


# ---------------------------------------------------------------------------
# Lookup table integrity tests
# ---------------------------------------------------------------------------

class TestWorldCoverLookupTable:
    """Sanity checks on the default WORLDCOVER_Z0_LOOKUP table."""

    def test_all_known_classes_have_entry(self):
        """Every class in WORLDCOVER_CLASS_NAMES must appear in the lookup table."""
        missing = set(WORLDCOVER_CLASS_NAMES) - set(WORLDCOVER_Z0_LOOKUP)
        assert not missing, f"Missing z0 entries for class codes: {missing}"

    def test_all_lookup_entries_have_name(self):
        """Every entry in WORLDCOVER_Z0_LOOKUP must have a corresponding name."""
        missing = set(WORLDCOVER_Z0_LOOKUP) - set(WORLDCOVER_CLASS_NAMES)
        assert not missing, f"Missing names for class codes: {missing}"

    def test_z0_values_are_positive(self):
        """All z0 values must be strictly positive."""
        non_positive = {code: z0 for code, z0 in WORLDCOVER_Z0_LOOKUP.items() if z0 <= 0}
        assert not non_positive, f"Non-positive z0 values found: {non_positive}"

    def test_z0_values_within_physical_range(self):
        """All z0 values should be within the physically meaningful range (1e-5 – 5 m)."""
        out_of_range = {
            code: z0
            for code, z0 in WORLDCOVER_Z0_LOOKUP.items()
            if not (1e-5 <= z0 <= 5.0)
        }
        assert not out_of_range, (
            f"z0 values outside expected physical range [1e-5, 5] m: {out_of_range}"
        )

    def test_water_is_smoother_than_forest(self):
        """Permanent water bodies (80) should be smoother than tree cover (10)."""
        assert WORLDCOVER_Z0_LOOKUP[80] < WORLDCOVER_Z0_LOOKUP[10]

    def test_water_is_smoother_than_urban(self):
        """Permanent water bodies (80) should be smoother than built-up areas (50)."""
        assert WORLDCOVER_Z0_LOOKUP[80] < WORLDCOVER_Z0_LOOKUP[50]

    def test_bare_ground_is_smoother_than_shrubland(self):
        """Bare/sparse vegetation (60) should be smoother than shrubland (20)."""
        assert WORLDCOVER_Z0_LOOKUP[60] < WORLDCOVER_Z0_LOOKUP[20]


# ---------------------------------------------------------------------------
# prepare_roughness_from_worldcover tests
# ---------------------------------------------------------------------------

class TestPrepareRoughnessFromWorldcover:
    """Tests for the prepare_roughness_from_worldcover() function."""

    def test_output_shape_matches_input(self, simple_worldcover_map):
        """Output array must have the same shape as the input class map."""
        z0 = prepare_roughness_from_worldcover(simple_worldcover_map)
        assert z0.shape == simple_worldcover_map.shape

    def test_output_dtype_is_float64(self, simple_worldcover_map):
        """Output array must be float64 regardless of input integer dtype."""
        z0 = prepare_roughness_from_worldcover(simple_worldcover_map)
        assert z0.dtype == np.float64

    def test_known_classes_map_to_correct_z0(self, simple_worldcover_map):
        """Every pixel with a known class code should map to the exact lookup value."""
        z0 = prepare_roughness_from_worldcover(simple_worldcover_map)
        for class_code, expected_z0 in WORLDCOVER_Z0_LOOKUP.items():
            mask = simple_worldcover_map == class_code
            if not np.any(mask):
                continue
            actual = z0[mask]
            np.testing.assert_array_equal(
                actual,
                expected_z0,
                err_msg=f"Class {class_code} ({WORLDCOVER_CLASS_NAMES[class_code]}): "
                        f"expected z0={expected_z0}, got {np.unique(actual)}",
            )

    def test_unknown_class_produces_nan(self, mixed_map_with_unknown):
        """Pixels with unknown class codes (e.g. 255) should produce NaN in z0."""
        z0 = prepare_roughness_from_worldcover(mixed_map_with_unknown)
        unknown_mask = mixed_map_with_unknown == 255
        assert np.all(np.isnan(z0[unknown_mask])), (
            "Expected NaN for unknown class 255, got "
            f"{z0[unknown_mask]}"
        )

    def test_known_pixels_not_nan(self, mixed_map_with_unknown):
        """Pixels with known class codes must NOT be NaN."""
        z0 = prepare_roughness_from_worldcover(mixed_map_with_unknown)
        unknown_mask = mixed_map_with_unknown == 255
        known_mask = ~unknown_mask
        assert not np.any(np.isnan(z0[known_mask])), (
            "Unexpected NaN values for known class codes."
        )

    def test_custom_lookup_overrides_default(self):
        """A custom lookup dict must be used instead of the default table."""
        data = np.array([[10, 20], [30, 40]], dtype=np.int32)
        custom = {10: 99.0, 20: 88.0}
        z0 = prepare_roughness_from_worldcover(data, lookup=custom)
        assert z0[0, 0] == 99.0
        assert z0[0, 1] == 88.0
        # Classes 30 and 40 are not in the custom lookup → NaN
        assert np.isnan(z0[1, 0])
        assert np.isnan(z0[1, 1])

    def test_all_pixels_covered_in_pure_map(self, simple_worldcover_map):
        """With all-known classes, there should be no NaN values in the output."""
        z0 = prepare_roughness_from_worldcover(simple_worldcover_map)
        assert not np.any(np.isnan(z0)), (
            f"Unexpected NaN at positions: {np.argwhere(np.isnan(z0))}"
        )

    def test_no_zero_z0_values(self, simple_worldcover_map):
        """z0 values of exactly zero would cause division-by-zero in CFD solvers."""
        z0 = prepare_roughness_from_worldcover(simple_worldcover_map)
        valid = z0[~np.isnan(z0)]
        assert np.all(valid > 0), f"Zero or negative z0 values found: {valid[valid <= 0]}"


# ---------------------------------------------------------------------------
# Diagnostic / integration test
# ---------------------------------------------------------------------------

class TestRoughnessMapDiagnostics:
    """Diagnostic tests that print detailed output to aid debugging.

    These tests print the classes found in the map, the lookup table, and
    per-class statistics.  Run ``pytest -s`` to see the printed output.
    """

    def test_diagnostic_output_for_simple_map(self, simple_worldcover_map):
        """Run the full diagnostic summary for a known synthetic WorldCover map."""
        z0 = prepare_roughness_from_worldcover(simple_worldcover_map)
        _summarise_roughness_map(simple_worldcover_map, z0)

        # Spot-check a few well-known mappings to catch obvious lookup errors.
        water_mask = simple_worldcover_map == 80
        np.testing.assert_allclose(z0[water_mask], WORLDCOVER_Z0_LOOKUP[80], rtol=1e-6)

        forest_mask = simple_worldcover_map == 10
        np.testing.assert_allclose(z0[forest_mask], WORLDCOVER_Z0_LOOKUP[10], rtol=1e-6)

    def test_diagnostic_output_for_mixed_map(self, mixed_map_with_unknown):
        """Diagnostic summary for a map that contains an unknown class code (255)."""
        z0 = prepare_roughness_from_worldcover(mixed_map_with_unknown)
        _summarise_roughness_map(mixed_map_with_unknown, z0)

        # Total NaN count should equal the number of unknown-class pixels.
        n_unknown = int(np.sum(mixed_map_with_unknown == 255))
        n_nan = int(np.sum(np.isnan(z0)))
        assert n_nan == n_unknown, (
            f"Expected {n_unknown} NaN pixel(s) for unknown class, found {n_nan}."
        )

    def test_each_class_maps_to_its_own_lookup_value(self, simple_worldcover_map):
        """Each class in the map should produce exactly the z0 from the lookup table.

        Note: multiple classes may legitimately share the same z0 value (e.g. tree
        cover and built-up are both ~1 m), so we check per-class correctness rather
        than expecting globally unique z0 values.
        """
        z0 = prepare_roughness_from_worldcover(simple_worldcover_map)
        for code in np.unique(simple_worldcover_map):
            expected = WORLDCOVER_Z0_LOOKUP[int(code)]
            actual = np.unique(z0[simple_worldcover_map == code])
            assert len(actual) == 1, f"Class {code}: expected one z0 value, got {actual}"
            np.testing.assert_allclose(
                actual[0], expected, rtol=1e-9,
                err_msg=f"Class {code} ({WORLDCOVER_CLASS_NAMES[int(code)]}): "
                        f"expected z0={expected}, got {actual[0]}",
            )
