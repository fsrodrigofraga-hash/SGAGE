from __future__ import annotations

# ============================================================
# ParametersEstimator.interpolation — N-D multilinear interpolation
# with a memoization cache
# ============================================================
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import ParamGridSpec
from .io_utils import _theta_to_axis_coord


# N-D multilinear interpolation + memo cache
# ============================================================
def _nd_multilinear_interp(point: Sequence[float], axes: List[np.ndarray], values: np.ndarray) -> np.ndarray:
    D = len(axes)
    if D == 0:
        raise ValueError("No axes provided for interpolation.")
    if values.ndim != D + 1:
        raise ValueError(f"values must have ndim={D+1}, got {values.ndim}")

    idx0: List[int] = []
    t: List[float] = []
    for d in range(D):
        g = axes[d]
        x = float(np.clip(point[d], g[0], g[-1]))
        i = int(np.searchsorted(g, x) - 1)
        i = int(np.clip(i, 0, len(g) - 2))
        x0, x1 = float(g[i]), float(g[i + 1])
        td = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        idx0.append(i)
        t.append(td)

    out = np.zeros(values.shape[-1], dtype=np.float64)
    for mask in range(1 << D):
        w = 1.0
        slicer = []
        for d in range(D):
            bit = (mask >> d) & 1
            w *= (t[d] if bit else (1.0 - t[d]))
            slicer.append(idx0[d] + bit)
        out += w * values[tuple(slicer) + (slice(None),)]
    return out


class InterpModelNDWithCache:
    def __init__(self, specs: List[ParamGridSpec], axes: List[np.ndarray], grid_values: np.ndarray, subdiv: int = 4):
        self.specs = specs
        self.axes = [np.asarray(a, dtype=np.float64) for a in axes]
        self.grid_values = np.asarray(grid_values, dtype=np.float64)

        self.qsteps = []
        for ax in self.axes:
            step = float(ax[1] - ax[0])
            self.qsteps.append(step / float(subdiv))

        self.cache: Dict[Tuple[float, ...], np.ndarray] = {}

    def _key_from_axis_point(self, axis_point: Sequence[float]) -> Tuple[float, ...]:
        key = []
        for d, x in enumerate(axis_point):
            q = self.qsteps[d]
            key.append(round(float(x) / q) * q)
        return tuple(key)

    def model(self, theta: Dict[str, float]) -> np.ndarray:
        axis_point = []
        for spec in self.specs:
            axis_point.append(_theta_to_axis_coord(spec, float(theta[spec.name])))

        key = self._key_from_axis_point(axis_point)
        if key in self.cache:
            return self.cache[key]

        vec = _nd_multilinear_interp(axis_point, self.axes, self.grid_values)
        self.cache[key] = vec
        return vec
