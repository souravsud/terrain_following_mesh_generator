"""3-zone boundary treatment with smooth-step blending.

Replaces the previous multi-scale pyramid approach. The terrain is divided into
three zones:

  AOI          – central region, original terrain fully preserved.
  Smooth-step  – blend from 100 % real terrain at the AOI edge to the target
                 elevation at the flat-zone boundary.  No additional smoothing
                 is applied to the DEM; only the blend weight changes.
  Flat zone    – thin strip at each enabled boundary face held at a constant
                 target elevation, ensuring the ABL inlet/outlet profiles are
                 applied on flat ground.

Four target-elevation strategies are available via BoundaryConfig.target_strategy:

  'boundary_strip'           Legacy: percentile-filtered mean of the flat-zone
                             pixels (what the old code used).
  'aoi_mean'                 Mean elevation of the AOI region.
  'transition_extrapolation' Fit a robust line through the transition zone
                             (collapsed to 1-D by taking the per-x-bin median),
                             then extrapolate to the flat-zone boundary.
  'clamped_extrapolation'    Same as above but clamped so the target never
                             exceeds the median of the actual flat-zone terrain
                             (prevents artificial highs from an upward-sloping
                             extrapolation).
"""

import logging
import numpy as np
from scipy.ndimage import generic_filter
from typing import Dict, Tuple

from .config import BoundaryConfig
from .utils import rotate_coordinates

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class BoundaryTreatment:
    """3-zone boundary treatment: AOI → smooth-step transition → flat."""

    def process_boundaries(
        self,
        elevation_data: np.ndarray,
        crop_mask: np.ndarray,
        config: BoundaryConfig,
        rotation_deg: float,
    ) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """Apply boundary treatment and return treated elevation data.

        Args:
            elevation_data: 2-D elevation array (may contain NaN outside crop).
            crop_mask:      Boolean mask – True where terrain data is valid.
            config:         BoundaryConfig controlling zone sizes and strategy.
            rotation_deg:   Wind/flow rotation angle (degrees CW from North).

        Returns:
            (treated_elevation, boundary_elevations, treated_mask, zones)
        """
        self._validate_config(config)

        elevation = self._fill_nan_pixels(elevation_data, crop_mask)

        if config.boundary_mode == "directional":
            result, zones, targets = self._apply_directional(
                elevation, crop_mask, config, rotation_deg
            )
        else:
            result, zones, targets = self._apply_radial(
                elevation, crop_mask, config
            )

        treated_mask = crop_mask.copy()
        result[~treated_mask] = np.nan

        logger.debug("Boundary treatment complete. Targets used: %s", targets)
        return result, targets, treated_mask, zones

    # ──────────────────────────────────────────────────────────────────────────
    # Directional mode
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_directional(
        self,
        elevation: np.ndarray,
        crop_mask: np.ndarray,
        config: BoundaryConfig,
        rotation_deg: float,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        result = elevation.copy()

        fc = self._get_flow_coordinates(crop_mask, rotation_deg)
        flow_x = fc["flow_x"]
        flow_y = fc["flow_y"]
        bounds = fc["bounds"]

        terrain_width = bounds["max_x"] - bounds["min_x"]
        terrain_height = bounds["max_y"] - bounds["min_y"]
        terrain_size = min(terrain_width, terrain_height)

        aoi_half = terrain_size * config.aoi_fraction / 2
        cx = (bounds["min_x"] + bounds["max_x"]) / 2
        cy = (bounds["min_y"] + bounds["max_y"]) / 2

        # Flat zone thickness (same geometry as original code: half of the
        # flat_boundary_thickness_fraction strip is truly flat, matching
        # existing config values)
        flat_thick = terrain_width * config.flat_boundary_thickness_fraction / 2

        aoi_mask = (
            (np.abs(flow_x - cx) <= aoi_half)
            & (np.abs(flow_y - cy) <= aoi_half)
            & crop_mask
        )

        # Per-face zone masks and targets
        all_flat = np.zeros_like(crop_mask, dtype=bool)
        all_smooth = np.zeros_like(crop_mask, dtype=bool)
        targets: Dict[str, float] = {}

        for direction in config.enabled_boundaries:
            f_flat, f_smooth, aoi_edge_x, flat_start_x = self._face_masks(
                flow_x, crop_mask, aoi_mask, bounds, cx, aoi_half,
                flat_thick, direction
            )
            if not np.any(f_flat) and not np.any(f_smooth):
                continue

            target = self._calculate_target(
                elevation, aoi_mask, f_smooth, f_flat,
                flow_x, direction, flat_start_x, config
            )
            targets[direction] = round(target, 3)
            logger.debug("%s target (%s): %.2f m", direction, config.target_strategy, target)

            # Apply flat zone
            result[f_flat] = target

            # Apply smooth-step blend in transition zone
            result = self._smooth_step_blend(
                result, elevation, f_smooth, flow_x,
                aoi_edge_x, flat_start_x, target, direction
            )

            all_flat |= f_flat
            all_smooth |= f_smooth

        # Corner pixels (inside crop, outside AOI, not in any enabled-face zone)
        # are left as original terrain. They correspond to the side boundaries
        # (north/south in the flow frame) which don't require flat treatment.
        # No blending is applied — there is no discontinuity because the
        # smooth-step zones adjacent to the corners have t=0 (w=0) at their
        # AOI edge, so they also equal original terrain at the junction.

        zones = {
            "aoi": aoi_mask,
            "transition": all_smooth,
            "blend": np.zeros_like(crop_mask, dtype=bool),  # no separate blend zone
            "flat": all_flat,
            "center": (fc["center_row"], fc["center_col"]),
        }
        return result, zones, targets

    def _face_masks(
        self, flow_x, crop_mask, aoi_mask, bounds, cx, aoi_half, flat_thick, direction
    ):
        """Return (flat_mask, smooth_mask, aoi_edge_x, flat_start_x) for one face."""
        if direction == "east":
            flat_start_x = bounds["max_x"] - flat_thick
            aoi_edge_x = cx + aoi_half
            f_flat = (flow_x >= flat_start_x) & crop_mask & ~aoi_mask
            f_smooth = (
                (flow_x >= aoi_edge_x)
                & (flow_x < flat_start_x)
                & crop_mask
                & ~aoi_mask
            )
        elif direction == "west":
            flat_start_x = bounds["min_x"] + flat_thick
            aoi_edge_x = cx - aoi_half
            f_flat = (flow_x <= flat_start_x) & crop_mask & ~aoi_mask
            f_smooth = (
                (flow_x <= aoi_edge_x)
                & (flow_x > flat_start_x)
                & crop_mask
                & ~aoi_mask
            )
        else:
            return (
                np.zeros_like(crop_mask, dtype=bool),
                np.zeros_like(crop_mask, dtype=bool),
                0.0,
                0.0,
            )
        return f_flat, f_smooth, aoi_edge_x, flat_start_x

    # ──────────────────────────────────────────────────────────────────────────
    # Radial mode
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_radial(
        self,
        elevation: np.ndarray,
        crop_mask: np.ndarray,
        config: BoundaryConfig,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        result = elevation.copy()

        rows, cols = np.where(crop_mask)
        cr, cc = np.mean(rows), np.mean(cols)
        yg, xg = np.mgrid[0:result.shape[0], 0:result.shape[1]]
        dist = np.sqrt((xg - cc) ** 2 + (yg - cr) ** 2)
        max_dist = np.max(dist[crop_mask])

        aoi_r = max_dist * config.aoi_fraction
        flat_thick = max_dist * config.flat_boundary_thickness_fraction / 2
        flat_start_r = max_dist - flat_thick

        aoi_mask = (dist <= aoi_r) & crop_mask
        flat_mask = (dist >= flat_start_r) & crop_mask
        smooth_mask = (dist > aoi_r) & (dist < flat_start_r) & crop_mask

        target = self._calculate_target_radial(elevation, aoi_mask, smooth_mask, flat_mask, dist, flat_start_r, config)
        targets = {"uniform": round(target, 3)}
        logger.debug("Radial target (%s): %.2f m", config.target_strategy, target)

        result[flat_mask] = target

        if np.any(smooth_mask):
            t = (dist[smooth_mask] - aoi_r) / (flat_start_r - aoi_r)
            t = np.clip(t, 0.0, 1.0)
            w = _smooth_step(t)
            result[smooth_mask] = elevation[smooth_mask] * (1.0 - w) + target * w

        zones = {
            "aoi": aoi_mask,
            "transition": smooth_mask,
            "blend": np.zeros_like(crop_mask, dtype=bool),
            "flat": flat_mask,
            "center": (cr, cc),
        }
        return result, zones, targets

    # ──────────────────────────────────────────────────────────────────────────
    # Smooth-step application
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _smooth_step_blend(
        result: np.ndarray,
        real_terrain: np.ndarray,
        mask: np.ndarray,
        flow_x: np.ndarray,
        aoi_edge_x: float,
        flat_start_x: float,
        target: float,
        direction: str,
    ) -> np.ndarray:
        """Blend real terrain toward target using a smooth-step weight.

        t = 0 at the AOI edge (pure terrain), t = 1 at the flat-zone start
        (pure target).  The smooth-step kernel w = 3t²-2t³ gives C¹
        continuity at both endpoints (zero slope at t=0 and t=1), preventing
        kinks where the zones meet.
        """
        if not np.any(mask):
            return result

        x = flow_x[mask]
        span = abs(flat_start_x - aoi_edge_x)
        if span < 1e-6:
            result[mask] = target
            return result

        if direction == "east":
            t = (x - aoi_edge_x) / span
        else:  # west
            t = (aoi_edge_x - x) / span

        t = np.clip(t, 0.0, 1.0)
        w = _smooth_step(t)
        result[mask] = real_terrain[mask] * (1.0 - w) + target * w
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Target-elevation strategies (directional)
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_target(
        self,
        elevation: np.ndarray,
        aoi_mask: np.ndarray,
        smooth_mask: np.ndarray,
        flat_mask: np.ndarray,
        flow_x: np.ndarray,
        direction: str,
        flat_start_x: float,
        config: BoundaryConfig,
    ) -> float:
        s = config.target_strategy
        if s == "boundary_strip":
            return _target_boundary_strip(elevation, flat_mask)
        if s == "aoi_mean":
            return _target_aoi_mean(elevation, aoi_mask)
        if s == "transition_extrapolation":
            return _target_extrapolation(
                elevation, smooth_mask, flat_mask, flow_x, direction, flat_start_x
            )
        if s == "clamped_extrapolation":
            extrap = _target_extrapolation(
                elevation, smooth_mask, flat_mask, flow_x, direction, flat_start_x
            )
            boundary_median = (
                float(np.median(elevation[flat_mask]))
                if np.any(flat_mask)
                else extrap
            )
            return min(extrap, boundary_median)
        raise ValueError(f"Unknown target_strategy: '{s}'")

    def _calculate_target_radial(
        self,
        elevation: np.ndarray,
        aoi_mask: np.ndarray,
        smooth_mask: np.ndarray,
        flat_mask: np.ndarray,
        dist: np.ndarray,
        flat_start_r: float,
        config: BoundaryConfig,
    ) -> float:
        s = config.target_strategy
        if s == "aoi_mean":
            return _target_aoi_mean(elevation, aoi_mask)
        if s in ("transition_extrapolation", "clamped_extrapolation"):
            extrap = _target_extrapolation_radial(elevation, smooth_mask, flat_mask, dist, flat_start_r)
            if s == "clamped_extrapolation" and np.any(flat_mask):
                extrap = min(extrap, float(np.median(elevation[flat_mask])))
            return extrap
        # boundary_strip (default for radial)
        return _target_boundary_strip(elevation, flat_mask)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_config(config: BoundaryConfig) -> None:
        if not (0 < config.aoi_fraction < 1):
            raise ValueError("aoi_fraction must be in (0, 1)")
        if not (0 < config.flat_boundary_thickness_fraction < 1):
            raise ValueError("flat_boundary_thickness_fraction must be in (0, 1)")
        valid = {"boundary_strip", "aoi_mean", "transition_extrapolation", "clamped_extrapolation"}
        if config.target_strategy not in valid:
            raise ValueError(
                f"target_strategy must be one of {valid}, got '{config.target_strategy}'"
            )

    @staticmethod
    def _fill_nan_pixels(elevation_data: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """Fill NaN pixels inside the crop mask using 3×3 neighbourhood mean."""
        out = elevation_data.copy()
        nan_mask = np.isnan(out) & crop_mask
        if not np.any(nan_mask):
            return out

        logger.debug("Filling %d NaN pixels inside crop mask...", np.sum(nan_mask))

        def _nan_mean(vals):
            v = vals[~np.isnan(vals)]
            return float(np.mean(v)) if v.size else 0.0

        filled = generic_filter(out, _nan_mean, size=3, mode="nearest")
        out[nan_mask] = filled[nan_mask]
        return out

    @staticmethod
    def _get_flow_coordinates(crop_mask: np.ndarray, rotation_deg: float) -> Dict:
        rows, cols = np.where(crop_mask)
        cr, cc = np.mean(rows), np.mean(cols)
        yg, xg = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
        rx, ry = xg - cc, yg - cr
        fx, fy = rotate_coordinates(rx, ry, 0, 0, rotation_deg, inverse=True)
        vfx, vfy = fx[crop_mask], fy[crop_mask]
        return {
            "flow_x": fx,
            "flow_y": fy,
            "center_row": cr,
            "center_col": cc,
            "bounds": {
                "min_x": vfx.min(),
                "max_x": vfx.max(),
                "min_y": vfy.min(),
                "max_y": vfy.max(),
            },
        }

    # kept for backward compat with any external callers
    def _create_zones_for_visualization(
        self, crop_mask: np.ndarray, config: BoundaryConfig, rotation_deg: float
    ) -> Dict:
        fc = self._get_flow_coordinates(crop_mask, rotation_deg)
        flow_x, flow_y = fc["flow_x"], fc["flow_y"]
        bounds = fc["bounds"]
        terrain_width = bounds["max_x"] - bounds["min_x"]
        terrain_size = min(terrain_width, bounds["max_y"] - bounds["min_y"])
        aoi_half = terrain_size * config.aoi_fraction / 2
        flat_thick = terrain_width * config.flat_boundary_thickness_fraction / 2
        cx = (bounds["min_x"] + bounds["max_x"]) / 2
        cy = (bounds["min_y"] + bounds["max_y"]) / 2

        aoi_mask = (
            (np.abs(flow_x - cx) <= aoi_half)
            & (np.abs(flow_y - cy) <= aoi_half)
            & crop_mask
        )
        flat_mask = np.zeros_like(crop_mask, dtype=bool)
        smooth_mask = np.zeros_like(crop_mask, dtype=bool)
        for direction in config.enabled_boundaries:
            if direction == "east":
                flat_mask |= (flow_x >= (bounds["max_x"] - flat_thick)) & crop_mask
                smooth_mask |= (
                    (flow_x >= (cx + aoi_half))
                    & (flow_x < (bounds["max_x"] - flat_thick))
                    & crop_mask
                )
            elif direction == "west":
                flat_mask |= (flow_x <= (bounds["min_x"] + flat_thick)) & crop_mask
                smooth_mask |= (
                    (flow_x <= (cx - aoi_half))
                    & (flow_x > (bounds["min_x"] + flat_thick))
                    & crop_mask
                )
        flat_mask &= ~aoi_mask
        smooth_mask &= ~aoi_mask & ~flat_mask
        rows, cols = np.where(crop_mask)
        return {
            "aoi": aoi_mask,
            "transition": smooth_mask,
            "blend": np.zeros_like(crop_mask, dtype=bool),
            "flat": flat_mask,
            "center": (np.mean(rows), np.mean(cols)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (no instance state needed)
# ──────────────────────────────────────────────────────────────────────────────

def _smooth_step(t: np.ndarray) -> np.ndarray:
    """C¹-continuous smooth-step: w = 3t² − 2t³.  Input must be in [0, 1]."""
    return 3.0 * t**2 - 2.0 * t**3


def _target_boundary_strip(elevation: np.ndarray, flat_mask: np.ndarray) -> float:
    """Percentile-filtered mean of flat-zone pixels (legacy strategy)."""
    if not np.any(flat_mask):
        return 0.0
    z = elevation[flat_mask]
    lo, hi = np.percentile(z, 10), np.percentile(z, 80)
    z_f = z[(z >= lo) & (z <= hi)]
    return float(np.mean(z_f)) if z_f.size else float(np.mean(z))


def _target_aoi_mean(elevation: np.ndarray, aoi_mask: np.ndarray) -> float:
    """Mean elevation of the AOI region."""
    if not np.any(aoi_mask):
        return 0.0
    return float(np.nanmean(elevation[aoi_mask]))


def _target_extrapolation(
    elevation: np.ndarray,
    smooth_mask: np.ndarray,
    flat_mask: np.ndarray,
    flow_x: np.ndarray,
    direction: str,
    flat_start_x: float,
    min_bins: int = 4,
) -> float:
    """Robust linear extrapolation from the transition zone to the flat boundary.

    Collapses the transition zone to a 1-D profile (median elevation per
    x-flow bin) then fits a line with the Theil-Sen estimator.  Falls back
    to ``_target_boundary_strip`` when there are too few bins for a reliable
    fit.
    """
    if not np.any(smooth_mask):
        return _target_boundary_strip(elevation, flat_mask)

    x_vals = flow_x[smooth_mask]
    z_vals = elevation[smooth_mask]
    x_min, x_max = x_vals.min(), x_vals.max()

    if x_max <= x_min:
        return _target_boundary_strip(elevation, flat_mask)

    n_bins = int(np.clip(np.sqrt(np.sum(smooth_mask)), 10, 60))
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bx, bz = [], []
    for i in range(n_bins):
        in_bin = (x_vals >= edges[i]) & (x_vals < edges[i + 1])
        if np.sum(in_bin) >= 3:
            bx.append(centers[i])
            bz.append(float(np.median(z_vals[in_bin])))

    if len(bx) < min_bins:
        logger.debug(
            "Too few transition-zone bins (%d) for extrapolation, falling back to boundary_strip.",
            len(bx),
        )
        return _target_boundary_strip(elevation, flat_mask)

    bx_arr = np.array(bx)
    bz_arr = np.array(bz)

    try:
        from scipy.stats import theilslopes
        res = theilslopes(bz_arr, bx_arr)
        slope, intercept = float(res.slope), float(res.intercept)
    except Exception:
        slope, intercept = np.polyfit(bx_arr, bz_arr, 1)

    return float(slope * flat_start_x + intercept)


def _target_extrapolation_radial(
    elevation: np.ndarray,
    smooth_mask: np.ndarray,
    flat_mask: np.ndarray,
    dist: np.ndarray,
    flat_start_r: float,
) -> float:
    """Radial version: fit a line of z vs radius through the transition zone."""
    if not np.any(smooth_mask):
        return _target_boundary_strip(elevation, flat_mask)

    r_vals = dist[smooth_mask]
    z_vals = elevation[smooth_mask]
    r_min, r_max = r_vals.min(), r_vals.max()
    if r_max <= r_min:
        return _target_boundary_strip(elevation, flat_mask)

    n_bins = int(np.clip(np.sqrt(np.sum(smooth_mask)), 10, 60))
    edges = np.linspace(r_min, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bx, bz = [], []
    for i in range(n_bins):
        in_bin = (r_vals >= edges[i]) & (r_vals < edges[i + 1])
        if np.sum(in_bin) >= 3:
            bx.append(centers[i])
            bz.append(float(np.median(z_vals[in_bin])))

    if len(bx) < 4:
        return _target_boundary_strip(elevation, flat_mask)

    try:
        from scipy.stats import theilslopes
        res = theilslopes(np.array(bz), np.array(bx))
        slope, intercept = float(res.slope), float(res.intercept)
    except Exception:
        slope, intercept = np.polyfit(np.array(bx), np.array(bz), 1)

    return float(slope * flat_start_r + intercept)
