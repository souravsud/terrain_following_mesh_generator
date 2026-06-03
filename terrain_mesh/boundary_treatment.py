"""3-zone boundary treatment with smooth-step blending.

The terrain is divided into three zones outside the enabled boundary faces:

  AOI          – central region, original terrain fully preserved.
  Smooth-step  – C¹-continuous blend (w = 3t²−2t³) from 100 % real terrain
                 at the AOI edge to the target elevation at the flat-zone
                 boundary.  No additional DEM smoothing is applied.
  Flat zone    – thin strip at each enabled face held at a constant target
                 elevation so ABL inlet/outlet profiles are applied on flat
                 ground.

Target elevation is computed by fitting a Theil-Sen line through the 1-D
median profile of the transition zone and extrapolating to the flat-zone
boundary.  When ``BoundaryConfig.clamp_target`` is True, the result is
capped at the median of the real flat-zone terrain to prevent the flat zone
from being raised above the actual terrain on upward-sloping domains.
"""

import logging
import numpy as np
from scipy.ndimage import generic_filter
from typing import Dict, Tuple

from .config import BoundaryConfig
from .utils import rotate_coordinates

logger = logging.getLogger(__name__)


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
            elevation_data: 2-D elevation array (NaN outside crop).
            crop_mask:      Boolean mask – True where terrain data is valid.
            config:         BoundaryConfig controlling zone sizes and clamping.
            rotation_deg:   Wind/flow rotation angle (degrees CW from North).

        Returns:
            (treated_elevation, boundary_elevations, treated_mask, zones)
        """
        _validate_config(config)

        elevation = _fill_nan_pixels(elevation_data, crop_mask)

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

        logger.debug("Boundary treatment complete. Targets: %s", targets)
        return result, targets, treated_mask, zones

    # ── Directional mode ──────────────────────────────────────────────────────

    def _apply_directional(
        self,
        elevation: np.ndarray,
        crop_mask: np.ndarray,
        config: BoundaryConfig,
        rotation_deg: float,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        result = elevation.copy()

        fc = _get_flow_coordinates(crop_mask, rotation_deg)
        flow_x = fc["flow_x"]
        flow_y = fc["flow_y"]
        bounds = fc["bounds"]

        terrain_width = bounds["max_x"] - bounds["min_x"]
        terrain_size  = min(terrain_width, bounds["max_y"] - bounds["min_y"])

        aoi_half  = terrain_size * config.aoi_fraction / 2
        flat_thick = terrain_width * config.flat_boundary_thickness_fraction / 2
        cx = (bounds["min_x"] + bounds["max_x"]) / 2
        cy = (bounds["min_y"] + bounds["max_y"]) / 2

        aoi_mask = (
            (np.abs(flow_x - cx) <= aoi_half)
            & (np.abs(flow_y - cy) <= aoi_half)
            & crop_mask
        )

        all_flat   = np.zeros_like(crop_mask, dtype=bool)
        all_smooth = np.zeros_like(crop_mask, dtype=bool)
        targets: Dict[str, float] = {}

        for direction in config.enabled_boundaries:
            f_flat, f_smooth, aoi_edge_x, flat_start_x = _face_masks(
                flow_x, crop_mask, aoi_mask, bounds, cx, aoi_half,
                flat_thick, direction
            )
            if not np.any(f_flat) and not np.any(f_smooth):
                continue

            target = _calculate_target(
                elevation, f_smooth, f_flat,
                flow_x, direction, flat_start_x, config
            )
            targets[direction] = round(target, 3)
            logger.debug("%s target (clamp=%s): %.2f m",
                         direction, config.clamp_target, target)

            result[f_flat] = target
            result = _smooth_step_blend(
                result, elevation, f_smooth, flow_x,
                aoi_edge_x, flat_start_x, target, direction
            )

            all_flat   |= f_flat
            all_smooth |= f_smooth

        # Corner pixels (inside crop, outside AOI, not in any face zone) are
        # left as original terrain — they correspond to the side boundaries
        # which carry no flat-terrain requirement.

        zones = {
            "aoi":        aoi_mask,
            "transition": all_smooth,
            "blend":      np.zeros_like(crop_mask, dtype=bool),
            "flat":       all_flat,
            "center":     (fc["center_row"], fc["center_col"]),
        }
        return result, zones, targets

    # ── Radial mode ───────────────────────────────────────────────────────────

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
        dist     = np.sqrt((xg - cc) ** 2 + (yg - cr) ** 2)
        max_dist = np.max(dist[crop_mask])

        aoi_r        = max_dist * config.aoi_fraction
        flat_thick   = max_dist * config.flat_boundary_thickness_fraction / 2
        flat_start_r = max_dist - flat_thick

        aoi_mask    = (dist <= aoi_r) & crop_mask
        flat_mask   = (dist >= flat_start_r) & crop_mask
        smooth_mask = (dist > aoi_r) & (dist < flat_start_r) & crop_mask

        target = _calculate_target_radial(
            elevation, smooth_mask, flat_mask, dist, flat_start_r, config
        )
        targets = {"uniform": round(target, 3)}
        logger.debug("Radial target (clamp=%s): %.2f m", config.clamp_target, target)

        result[flat_mask] = target

        if np.any(smooth_mask):
            t = (dist[smooth_mask] - aoi_r) / (flat_start_r - aoi_r)
            t = np.clip(t, 0.0, 1.0)
            w = _smooth_step(t)
            result[smooth_mask] = elevation[smooth_mask] * (1.0 - w) + target * w

        zones = {
            "aoi":        aoi_mask,
            "transition": smooth_mask,
            "blend":      np.zeros_like(crop_mask, dtype=bool),
            "flat":       flat_mask,
            "center":     (cr, cc),
        }
        return result, zones, targets

    # kept for any external callers that inspect zone geometry without running
    # the full treatment (e.g. visualiser previews)
    def _create_zones_for_visualization(
        self, crop_mask: np.ndarray, config: BoundaryConfig, rotation_deg: float
    ) -> Dict:
        fc = _get_flow_coordinates(crop_mask, rotation_deg)
        flow_x, flow_y = fc["flow_x"], fc["flow_y"]
        bounds = fc["bounds"]
        terrain_width = bounds["max_x"] - bounds["min_x"]
        terrain_size  = min(terrain_width, bounds["max_y"] - bounds["min_y"])
        aoi_half   = terrain_size * config.aoi_fraction / 2
        flat_thick = terrain_width * config.flat_boundary_thickness_fraction / 2
        cx = (bounds["min_x"] + bounds["max_x"]) / 2
        cy = (bounds["min_y"] + bounds["max_y"]) / 2

        aoi_mask  = (
            (np.abs(flow_x - cx) <= aoi_half)
            & (np.abs(flow_y - cy) <= aoi_half)
            & crop_mask
        )
        flat_mask   = np.zeros_like(crop_mask, dtype=bool)
        smooth_mask = np.zeros_like(crop_mask, dtype=bool)
        for direction in config.enabled_boundaries:
            if direction == "east":
                flat_mask   |= (flow_x >= (bounds["max_x"] - flat_thick)) & crop_mask
                smooth_mask |= (
                    (flow_x >= (cx + aoi_half))
                    & (flow_x < (bounds["max_x"] - flat_thick))
                    & crop_mask
                )
            elif direction == "west":
                flat_mask   |= (flow_x <= (bounds["min_x"] + flat_thick)) & crop_mask
                smooth_mask |= (
                    (flow_x <= (cx - aoi_half))
                    & (flow_x > (bounds["min_x"] + flat_thick))
                    & crop_mask
                )
        flat_mask   &= ~aoi_mask
        smooth_mask &= ~aoi_mask & ~flat_mask
        rows, cols = np.where(crop_mask)
        return {
            "aoi":        aoi_mask,
            "transition": smooth_mask,
            "blend":      np.zeros_like(crop_mask, dtype=bool),
            "flat":       flat_mask,
            "center":     (np.mean(rows), np.mean(cols)),
        }


# ── Module-level helpers ──────────────────────────────────────────────────────

def _validate_config(config: BoundaryConfig) -> None:
    if not (0 < config.aoi_fraction < 1):
        raise ValueError("aoi_fraction must be in (0, 1)")
    if not (0 < config.flat_boundary_thickness_fraction < 1):
        raise ValueError("flat_boundary_thickness_fraction must be in (0, 1)")


def _fill_nan_pixels(elevation_data: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    """Fill NaN pixels inside the crop mask using a 3×3 neighbourhood mean."""
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


def _get_flow_coordinates(crop_mask: np.ndarray, rotation_deg: float) -> Dict:
    rows, cols = np.where(crop_mask)
    cr, cc = np.mean(rows), np.mean(cols)
    yg, xg = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
    fx, fy = rotate_coordinates(xg - cc, yg - cr, 0, 0, rotation_deg, inverse=True)
    vfx, vfy = fx[crop_mask], fy[crop_mask]
    return {
        "flow_x": fx, "flow_y": fy,
        "center_row": cr, "center_col": cc,
        "bounds": {
            "min_x": vfx.min(), "max_x": vfx.max(),
            "min_y": vfy.min(), "max_y": vfy.max(),
        },
    }


def _face_masks(flow_x, crop_mask, aoi_mask, bounds, cx, aoi_half, flat_thick, direction):
    """Return (flat_mask, smooth_mask, aoi_edge_x, flat_start_x) for one face."""
    if direction == "east":
        flat_start_x = bounds["max_x"] - flat_thick
        aoi_edge_x   = cx + aoi_half
        f_flat   = (flow_x >= flat_start_x) & crop_mask & ~aoi_mask
        f_smooth = (flow_x >= aoi_edge_x) & (flow_x < flat_start_x) & crop_mask & ~aoi_mask
    elif direction == "west":
        flat_start_x = bounds["min_x"] + flat_thick
        aoi_edge_x   = cx - aoi_half
        f_flat   = (flow_x <= flat_start_x) & crop_mask & ~aoi_mask
        f_smooth = (flow_x <= aoi_edge_x) & (flow_x > flat_start_x) & crop_mask & ~aoi_mask
    else:
        empty = np.zeros_like(crop_mask, dtype=bool)
        return empty, empty, 0.0, 0.0
    return f_flat, f_smooth, aoi_edge_x, flat_start_x


def _smooth_step(t: np.ndarray) -> np.ndarray:
    """C¹-continuous kernel: w = 3t² − 2t³.  t must be in [0, 1]."""
    return 3.0 * t**2 - 2.0 * t**3


def _smooth_step_blend(
    result, real_terrain, mask, flow_x, aoi_edge_x, flat_start_x, target, direction
) -> np.ndarray:
    """Blend real terrain toward target with a smooth-step weight."""
    if not np.any(mask):
        return result
    span = abs(flat_start_x - aoi_edge_x)
    if span < 1e-6:
        result[mask] = target
        return result
    x = flow_x[mask]
    t = (x - aoi_edge_x) / span if direction == "east" else (aoi_edge_x - x) / span
    w = _smooth_step(np.clip(t, 0.0, 1.0))
    result[mask] = real_terrain[mask] * (1.0 - w) + target * w
    return result


def _calculate_target(
    elevation, smooth_mask, flat_mask, flow_x, direction, flat_start_x, config
) -> float:
    """Extrapolate transition-zone trend to the flat boundary, with optional clamp."""
    target = _extrapolate_target(elevation, smooth_mask, flat_mask, flow_x, flat_start_x)
    if config.clamp_target and np.any(flat_mask):
        target = min(target, float(np.median(elevation[flat_mask])))
    return target


def _calculate_target_radial(
    elevation, smooth_mask, flat_mask, dist, flat_start_r, config
) -> float:
    """Radial version of _calculate_target."""
    target = _extrapolate_target_radial(elevation, smooth_mask, flat_mask, dist, flat_start_r)
    if config.clamp_target and np.any(flat_mask):
        target = min(target, float(np.median(elevation[flat_mask])))
    return target


def _extrapolate_target(
    elevation, smooth_mask, flat_mask, flow_x, flat_start_x, min_bins: int = 4
) -> float:
    """Theil-Sen line fit through transition zone, extrapolated to flat boundary.

    Collapses the 2-D transition zone to a 1-D profile by taking the median
    elevation across the cross-flow direction for each flow-x bin, then fits
    a robust line and evaluates it at ``flat_start_x``.  Falls back to the
    median of the flat-zone pixels when the transition zone is too thin for a
    reliable fit.
    """
    if not np.any(smooth_mask):
        return _flat_zone_median(elevation, flat_mask)

    x_vals = flow_x[smooth_mask]
    z_vals = elevation[smooth_mask]
    x_min, x_max = x_vals.min(), x_vals.max()
    if x_max <= x_min:
        return _flat_zone_median(elevation, flat_mask)

    n_bins = int(np.clip(np.sqrt(np.sum(smooth_mask)), 10, 60))
    edges   = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bx, bz = [], []
    for i in range(n_bins):
        in_bin = (x_vals >= edges[i]) & (x_vals < edges[i + 1])
        if np.sum(in_bin) >= 3:
            bx.append(centers[i])
            bz.append(float(np.median(z_vals[in_bin])))

    if len(bx) < min_bins:
        logger.debug("Too few bins (%d) for extrapolation, using flat-zone median.", len(bx))
        return _flat_zone_median(elevation, flat_mask)

    bx_arr, bz_arr = np.array(bx), np.array(bz)
    try:
        from scipy.stats import theilslopes
        res = theilslopes(bz_arr, bx_arr)
        slope, intercept = float(res.slope), float(res.intercept)
    except Exception:
        slope, intercept = np.polyfit(bx_arr, bz_arr, 1)

    return float(slope * flat_start_x + intercept)


def _extrapolate_target_radial(
    elevation, smooth_mask, flat_mask, dist, flat_start_r
) -> float:
    """Radial version: fit z vs radius through the transition zone."""
    if not np.any(smooth_mask):
        return _flat_zone_median(elevation, flat_mask)

    r_vals = dist[smooth_mask]
    z_vals = elevation[smooth_mask]
    r_min, r_max = r_vals.min(), r_vals.max()
    if r_max <= r_min:
        return _flat_zone_median(elevation, flat_mask)

    n_bins = int(np.clip(np.sqrt(np.sum(smooth_mask)), 10, 60))
    edges   = np.linspace(r_min, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bx, bz = [], []
    for i in range(n_bins):
        in_bin = (r_vals >= edges[i]) & (r_vals < edges[i + 1])
        if np.sum(in_bin) >= 3:
            bx.append(centers[i])
            bz.append(float(np.median(z_vals[in_bin])))

    if len(bx) < 4:
        return _flat_zone_median(elevation, flat_mask)

    bx_arr, bz_arr = np.array(bx), np.array(bz)
    try:
        from scipy.stats import theilslopes
        res = theilslopes(bz_arr, bx_arr)
        slope, intercept = float(res.slope), float(res.intercept)
    except Exception:
        slope, intercept = np.polyfit(bx_arr, bz_arr, 1)

    return float(slope * flat_start_r + intercept)


def _flat_zone_median(elevation: np.ndarray, flat_mask: np.ndarray) -> float:
    """Robust fallback: median of whatever terrain is in the flat zone."""
    if not np.any(flat_mask):
        return 0.0
    return float(np.median(elevation[flat_mask]))
