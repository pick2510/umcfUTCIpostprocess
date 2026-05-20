#!/usr/bin/env python3
"""Smoke tests for _bin_vtk_to_grid NN-fill logic."""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from run_utci import _bin_vtk_to_grid

import pyvista as pv


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_surface_vtk(xs, ys, z_fn=None):
    """
    Write a StructuredGrid surface VTK to a temp file and return its path.
    xs, ys: 1-D arrays; z_fn(xg, yg) → z array (None = flat at z=0).
    """
    xg, yg = np.meshgrid(xs, ys)
    zg = z_fn(xg, yg) if z_fn is not None else np.zeros_like(xg)
    grid = pv.StructuredGrid(xg, yg, zg)
    surface = grid.extract_surface(algorithm='dataset_surface')
    path = tempfile.mktemp(suffix='.vtk')
    surface.save(path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_fill_disabled_returns_occupied_only():
    """fill_radius=0 must return only the bins that contain a face centre."""
    # 3 m spacing → only every 3rd 1 m bin is occupied
    xs = np.arange(0.0, 31.0, 3.0)
    ys = np.arange(0.0, 31.0, 3.0)
    path = _make_surface_vtk(xs, ys)
    try:
        pos_off = _bin_vtk_to_grid(path, dx=1.0, dy=1.0, z_offset=2.0, fill_radius=0.0)
        pos_on  = _bin_vtk_to_grid(path, dx=1.0, dy=1.0, z_offset=2.0, fill_radius=-1.0)
        assert len(pos_on) > len(pos_off), (
            f"adaptive fill should add points: off={len(pos_off)} on={len(pos_on)}")
        # No-fill count must equal the number of distinct snapped occupied bins
        # (≤ number of cell centres)
        assert len(pos_off) <= len(xs) * len(ys)
        print(f"  PASS  fill_disabled: {len(pos_off)} pts  (adaptive: {len(pos_on)})")
    finally:
        os.unlink(path)


def test_adaptive_fill_covers_coarse_gaps():
    """Adaptive fill should produce uniform-ish 1 m coverage across both regions."""
    # Fine 1 m region x∈[0,10], coarse 4 m region x∈[10,30]
    fine_xs   = np.arange(0.0,  11.0, 1.0)
    coarse_xs = np.arange(10.0, 31.0, 4.0)
    ys = np.arange(0.0, 11.0, 1.0)

    fine_path   = _make_surface_vtk(fine_xs,   ys)
    coarse_path = _make_surface_vtk(coarse_xs, ys)

    # Merge into one VTK by combining PolyData
    fine_surf   = pv.read(fine_path)
    coarse_surf = pv.read(coarse_path)
    combined    = fine_surf.merge(coarse_surf)
    combined_path = tempfile.mktemp(suffix='.vtk')
    combined.save(combined_path)

    try:
        pos_off = _bin_vtk_to_grid(combined_path, dx=1.0, dy=1.0,
                                   z_offset=2.0, fill_radius=0.0)
        pos_on  = _bin_vtk_to_grid(combined_path, dx=1.0, dy=1.0,
                                   z_offset=2.0, fill_radius=-1.0)
        assert len(pos_on) > len(pos_off), (
            f"fill should add points in coarse region: off={len(pos_off)} on={len(pos_on)}")
        # Z offset must be applied
        assert all(abs(p[2] - 2.0) < 1e-6 for p in pos_on), "z_offset not applied"
        print(f"  PASS  adaptive_fill: no-fill={len(pos_off)}  filled={len(pos_on)}")
    finally:
        for p in (fine_path, coarse_path, combined_path):
            os.unlink(p)


def test_explicit_radius_is_monotone():
    """Larger explicit radius must yield >= as many points as smaller radius."""
    xs = np.arange(0.0, 6.0, 2.0)
    ys = np.arange(0.0, 6.0, 2.0)
    path = _make_surface_vtk(xs, ys)
    try:
        n2 = len(_bin_vtk_to_grid(path, dx=1.0, dy=1.0, z_offset=2.0, fill_radius=2.0))
        n4 = len(_bin_vtk_to_grid(path, dx=1.0, dy=1.0, z_offset=2.0, fill_radius=4.0))
        n8 = len(_bin_vtk_to_grid(path, dx=1.0, dy=1.0, z_offset=2.0, fill_radius=8.0))
        assert n2 <= n4 <= n8, f"not monotone: r2={n2} r4={n4} r8={n8}"
        print(f"  PASS  explicit_radius: r=2→{n2}  r=4→{n4}  r=8→{n8}")
    finally:
        os.unlink(path)


def test_terrain_z_follows_slope():
    """Terrain-mode fill must interpolate Z from nearest occupied bin."""
    xs = np.arange(0.0, 11.0, 2.0)
    ys = np.arange(0.0, 6.0,  2.0)
    path = _make_surface_vtk(xs, ys, z_fn=lambda xg, yg: xg * 0.5)  # z = 0.5·x
    try:
        pos = _bin_vtk_to_grid(path, dx=1.0, dy=1.0, z_offset=2.0, fill_radius=-1.0)
        assert len(pos) > 0
        # Every point's Z minus the offset should be non-negative (slope ≥ 0)
        assert all(p[2] >= 2.0 - 1e-6 for p in pos), (
            "z should be ≥ z_offset on a non-negative slope")
        # Filled points at x≈5 should have z ≈ 0.5*5 + 2 = 4.5, not 0+2
        pts_x5 = [(x, y, z) for x, y, z in pos if abs(x - 5.0) < 0.5]
        if pts_x5:
            avg_z = sum(p[2] for p in pts_x5) / len(pts_x5)
            assert avg_z > 3.0, f"Z at x=5 should be > 3, got {avg_z:.2f}"
        print(f"  PASS  terrain_z_slope: {len(pos)} pts, "
              f"z at x=5 ≈ {sum(p[2] for p in pts_x5)/max(1,len(pts_x5)):.2f}")
    finally:
        os.unlink(path)


def test_bbox_clips_fill():
    """Points outside the bbox must not appear even with large fill radius."""
    xs = np.arange(0.0, 21.0, 1.0)
    ys = np.arange(0.0, 21.0, 1.0)
    path = _make_surface_vtk(xs, ys)
    bbox = (0.0, 10.0, 0.0, 10.0)
    try:
        pos = _bin_vtk_to_grid(path, dx=1.0, dy=1.0, z_offset=2.0,
                               bbox=bbox, fill_radius=100.0)
        assert all(0.0 <= p[0] <= 10.0 and 0.0 <= p[1] <= 10.0 for p in pos), \
            "points found outside bbox"
        print(f"  PASS  bbox_clips_fill: {len(pos)} pts within bbox")
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        test_fill_disabled_returns_occupied_only,
        test_adaptive_fill_covers_coarse_gaps,
        test_explicit_radius_is_monotone,
        test_terrain_z_follows_slope,
        test_bbox_clips_fill,
    ]
    passed = failed = 0
    print(f"Running {len(tests)} smoke tests for _bin_vtk_to_grid ...\n")
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'OK' if failed == 0 else 'FAILED'}  {passed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
