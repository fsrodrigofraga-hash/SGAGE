from __future__ import annotations

# ============================================================
# ParametersEstimator.cpu — automatic core allocation  [MEM-2]
# ============================================================
from multiprocessing import cpu_count
from typing import Optional, Tuple


# CPU auto-detection helpers  [MEM-2]
# ============================================================
def _resolve_cpu_jobs(n_cores: int) -> Tuple[int, int, int]:
    """
    Return (grid_jobs, sampler_pool, cpop_jobs) based on core count,
    following the RAM-aware heuristic.
    """
    grid_jobs    = min(max(1, n_cores), 8)
    sampler_pool = min(max(1, n_cores), 8)
    cpop_jobs    = max(1, min(n_cores // 4, 4))
    return grid_jobs, sampler_pool, cpop_jobs


def _effective_jobs(cfg_value: Optional[int], auto_value: int) -> int:
    """Return cfg override if set, otherwise the auto-detected value."""
    if cfg_value is not None and int(cfg_value) > 0:
        return int(cfg_value)
    return int(auto_value)
