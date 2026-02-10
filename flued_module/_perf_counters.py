"""
Temporary performance counters for profiling.
DELETE THIS FILE after profiling is complete.

Usage from any module:
    from ._perf_counters import COUNTERS
    COUNTERS['ht_kernel_calls'] += 1
"""

COUNTERS = {
    'timesteps': 0,
    'picard_iterations': 0,
    'ht_kernel_calls': 0,
    'triple_wall_substep_events': 0,
    'triple_wall_total_substeps': 0,
    'condensation_iters_total': 0,
}


def reset():
    for k in COUNTERS:
        COUNTERS[k] = 0


def summary(elapsed_s: float) -> str:
    c = COUNTERS
    ts = c['timesteps'] or 1  # avoid div-by-zero
    ht = c['ht_kernel_calls'] or 1

    lines = [
        "",
        "=" * 66,
        "  PERFORMANCE COUNTERS (temporary instrumentation)",
        "=" * 66,
        f"  Simulation wall-clock time:   {elapsed_s:,.1f} s",
        f"  Timesteps completed:          {c['timesteps']:,}",
        f"  Picard coupling iterations:   {c['picard_iterations']:,}"
        f"  (avg {c['picard_iterations']/ts:.2f} / timestep)",
        f"  HT kernel calls:             {c['ht_kernel_calls']:,}"
        f"  (avg {c['ht_kernel_calls']/ts:.0f} / timestep)",
        f"  Condensation iters (total):   {c['condensation_iters_total']:,}"
        f"  (avg {c['condensation_iters_total']/ht:.1f} / kernel call)",
        f"  Triple-wall substep events:   {c['triple_wall_substep_events']:,}",
        f"  Triple-wall total substeps:   {c['triple_wall_total_substeps']:,}",
        "=" * 66,
        "",
    ]
    return "\n".join(lines)
