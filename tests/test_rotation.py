"""Tests for the rotation scheme in rotate_coordinates.

Verifies that:
- geographic=True (UTM, y=North) maps the downwind vector to positive y_rot so that
  the blockmesh inlet face (at minimum j / minimum y_rot) is on the upwind side.
- geographic=False (pixel, y=South) maps the downwind vector to positive flow_x, as
  used by the boundary-treatment and visualiser code.
- The forward and inverse rotations are each other's inverses (round-trip identity).
- 45° and 225° give the same results as the pre-fix formula (those angles were already
  correct and must not regress).
"""

import numpy as np
import pytest
from terrain_mesh.utils import rotate_coordinates


# Angles to test: cardinal directions + 45° diagonals + the originally-failing 210°
TEST_ANGLES = [0, 45, 90, 135, 180, 210, 225, 270, 315]


class TestGeographicRotation:
    """geographic=True: UTM convention, y increases northward."""

    def _downwind_utm(self, rotation_deg):
        """Unit downwind vector in UTM (East, North) for a meteorological angle."""
        return (
            -np.sin(np.radians(rotation_deg)),
            -np.cos(np.radians(rotation_deg)),
        )

    @pytest.mark.parametrize("rot", TEST_ANGLES)
    def test_downwind_maps_to_positive_y_rot(self, rot):
        """Downwind direction must map to positive y_rot (outlet side)."""
        dw_x, dw_y = self._downwind_utm(rot)
        _, y_rot = rotate_coordinates(dw_x, dw_y, 0, 0, rot, inverse=True, geographic=True)
        assert y_rot > 0.5, (
            f"rot={rot}°: downwind y_rot={y_rot:.4f} should be strongly positive"
        )

    @pytest.mark.parametrize("rot", TEST_ANGLES)
    def test_upwind_maps_to_negative_y_rot(self, rot):
        """Upwind direction must map to negative y_rot (inlet side)."""
        dw_x, dw_y = self._downwind_utm(rot)
        _, y_rot = rotate_coordinates(-dw_x, -dw_y, 0, 0, rot, inverse=True, geographic=True)
        assert y_rot < -0.5, (
            f"rot={rot}°: upwind y_rot={y_rot:.4f} should be strongly negative"
        )

    @pytest.mark.parametrize("rot", TEST_ANGLES)
    def test_round_trip_identity(self, rot):
        """inverse=True followed by inverse=False should recover the original point."""
        pts = np.array([[1000.0, 500.0], [-800.0, 300.0], [0.0, -600.0]])
        for pt in pts:
            xr, yr = rotate_coordinates(pt[0], pt[1], 0, 0, rot, inverse=True, geographic=True)
            xb, yb = rotate_coordinates(xr, yr, 0, 0, rot, inverse=False, geographic=True)
            np.testing.assert_allclose(
                [xb, yb], pt, atol=1e-6,
                err_msg=f"rot={rot}°: round-trip failed for point {pt}",
            )

    @pytest.mark.parametrize("rot", TEST_ANGLES)
    def test_inlet_face_in_upwind_direction(self, rot):
        """The face of a square terrain with minimum y_rot lies on the upwind side.

        In the blockmesh, the inlet patch is placed at minimum j (minimum y_rot).
        This verifies that the inlet patch faces upwind for every meteorological angle.
        """
        H = 10_000.0
        # Dense grid covering the square terrain
        n = 40
        xs = np.linspace(-H, H, n)
        ys = np.linspace(-H, H, n)
        X, Y = np.meshgrid(xs, ys)

        _, Y_rot = rotate_coordinates(
            X.ravel(), Y.ravel(), 0, 0, rot, inverse=True, geographic=True
        )
        Y_rot = Y_rot.reshape(n, n)

        # Pixels on the inlet face (min y_rot strip)
        min_y_rot = Y_rot.min()
        inlet_mask = Y_rot < (min_y_rot + H * 0.02)

        inlet_cx = X[inlet_mask].mean()
        inlet_cy = Y[inlet_mask].mean()

        # The inlet centroid should be in the upwind (source) direction
        upwind_x = np.sin(np.radians(rot))
        upwind_y = np.cos(np.radians(rot))
        dot = inlet_cx * upwind_x + inlet_cy * upwind_y

        assert dot > 0, (
            f"rot={rot}°: inlet centroid ({inlet_cx/H:.2f}H, {inlet_cy/H:.2f}H) "
            f"is not in the upwind direction ({upwind_x:.2f}, {upwind_y:.2f})"
        )


class TestPixelRotation:
    """geographic=False (default): pixel convention, y increases southward."""

    def _downwind_pixel(self, rotation_deg):
        """Unit downwind vector in pixel space (East=+x, South=+y).

        In pixel space +y points South, which is the *negative* of UTM-North.
        The downwind UTM-North component is -cos(rotation_deg), so the
        pixel-South component is +cos(rotation_deg).
        """
        return (
            -np.sin(np.radians(rotation_deg)),   # East (same in both conventions)
            np.cos(np.radians(rotation_deg)),     # Pixel-South = -UTM-North = cos(rot)
        )

    @pytest.mark.parametrize("rot", TEST_ANGLES)
    def test_downwind_maps_to_positive_flow_x(self, rot):
        """Downwind direction must map to positive flow_x (boundary-treatment 'east')."""
        dw_xpx, dw_ypx = self._downwind_pixel(rot)
        flow_x, _ = rotate_coordinates(
            dw_xpx, dw_ypx, 0, 0, rot, inverse=True, geographic=False
        )
        assert flow_x > 0.5, (
            f"rot={rot}°: pixel downwind flow_x={flow_x:.4f} should be strongly positive"
        )

    @pytest.mark.parametrize("rot", TEST_ANGLES)
    def test_round_trip_identity(self, rot):
        """Round-trip identity for the pixel convention."""
        pts = np.array([[300.0, 200.0], [-500.0, 100.0]])
        for pt in pts:
            xr, yr = rotate_coordinates(pt[0], pt[1], 0, 0, rot, inverse=True, geographic=False)
            xb, yb = rotate_coordinates(xr, yr, 0, 0, rot, inverse=False, geographic=False)
            np.testing.assert_allclose([xb, yb], pt, atol=1e-6)


class TestPreFixAnglesUnchanged:
    """45° and 225° gave correct results before the fix; they must not regress."""

    def _old_rotate(self, x, y, rot, inverse=False):
        """Pre-fix formula for reference."""
        theta = np.radians(rot - 270)
        if inverse:
            theta = -theta
        c, s = np.cos(theta), np.sin(theta)
        return c * x - s * y, s * x + c * y

    @pytest.mark.parametrize("rot", [45, 225])
    def test_results_unchanged_for_working_angles(self, rot):
        """geographic=True must give the same result as the old formula for 45° and 225°."""
        pts = [(500.0, 300.0), (-200.0, 400.0), (0.0, -800.0)]
        for pt in pts:
            xr_old, yr_old = self._old_rotate(pt[0], pt[1], rot, inverse=True)
            xr_new, yr_new = rotate_coordinates(
                pt[0], pt[1], 0, 0, rot, inverse=True, geographic=True
            )
            np.testing.assert_allclose(
                [xr_new, yr_new], [xr_old, yr_old], atol=1e-6,
                err_msg=f"rot={rot}°: geographic=True result changed for point {pt}",
            )
