"""
Front-end integration entry point for flue simulation.

Usage:
    # As a Python function:
    from run_simulation import run_simulation
    results = run_simulation(config)

    # As a CLI script:
    python run_simulation.py config.json
    python run_simulation.py config.json --output results.json
"""
import sys
import json
import numpy as np

from flued_module import Chimney, make_smooth_door_schedule


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _build_inlet_temp_func(spec):
    """Convert a JSON-friendly inlet temp spec into a callable.

    Accepts:
        float/int         -> constant temperature [K]
        list of [t, T_K]  -> linear-interpolated time series
        None              -> no inlet temp override
    """
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        T_const = float(spec)
        return lambda t: T_const
    if isinstance(spec, list):
        times = [pt[0] for pt in spec]
        temps = [pt[1] for pt in spec]
        return lambda t: float(np.interp(t, times, temps))
    raise ValueError(f"Unknown inlet_temp_profile format: {type(spec)}")


def _build_door_opening_func(spec, total_time):
    """Convert a JSON-friendly door schedule into a callable.

    Accepts:
        float/int                     -> constant opening percent
        list of [t, opening_percent]  -> smooth ramp schedule
        None                          -> no dynamic door (use inlet_restriction default)
    """
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        pct = float(spec)
        return lambda t: pct
    if isinstance(spec, list):
        events = [(pt[0], pt[1]) for pt in spec]
        return make_smooth_door_schedule(events)
    raise ValueError(f"Unknown door_schedule format: {type(spec)}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_simulation(config: dict) -> dict:
    """Run a chimney simulation from a JSON-friendly config dict.

    Parameters
    ----------
    config : dict with keys:
        parts : list[dict]          -- chimney geometry (required)
        elbows : list[dict]         -- fittings between parts (optional)
        dt : float                  -- time step [s] (default 0.5)
        total_time : float          -- simulation duration [s] (required)
        inlet_restriction : dict    -- door/damper config (optional)
        inlet_temp_profile : float | list[[t,T_K]] -- inlet temperature (optional)
        door_schedule : float | list[[t,pct]]      -- door opening (optional)
        top_boundary : dict         -- wind cap config (optional)
        fittings_thermal : dict     -- fitting thermal model (optional)
        door_temp_dilution : dict   -- inlet air mixing (optional)
        house_stack : dict          -- house stack effect {'enable', 'npp_height_m'} (optional)
        verbose : bool              -- print solver details (default False)

    Returns
    -------
    dict with JSON-serializable results:
        time_s, mass_flow_kg_s, exit_temp_K, inlet_temp_K,
        T_gas_K, T_wall_K, velocity_m_s, pressure_Pa, density_kg_m3,
        segment_heights_m, summary, pressure_profile
    """
    # --- Required ---
    # Strip underscore-prefixed keys (JSON "comments") from parts/elbows
    parts = [{k: v for k, v in p.items() if not k.startswith('_')} for p in config['parts']]
    total_time = config['total_time']

    # --- Optional with defaults ---
    dt = config.get('dt', 1.0)
    elbows = config.get('elbows', None)
    inlet_restriction = config.get('inlet_restriction', None)
    top_boundary = config.get('top_boundary', None)
    fittings_thermal = config.get('fittings_thermal', None)
    door_temp_dilution = config.get('door_temp_dilution', None)
    house_stack = config.get('house_stack', None)
    verbose = config.get('verbose', False)

    # --- Build callables from JSON-friendly specs ---
    inlet_temp_func = _build_inlet_temp_func(config.get('inlet_temp_profile'))
    door_opening_func = _build_door_opening_func(config.get('door_schedule'), total_time)

    # --- Create and run ---
    ch = Chimney(
        parts=parts,
        dt=dt,
        total_time=total_time,
        elbows=elbows,
        inlet_restriction=inlet_restriction,
        top_boundary=top_boundary,
        fittings_thermal=fittings_thermal,
        door_temp_dilution=door_temp_dilution,
        house_stack=house_stack,
        inlet_temp_func=inlet_temp_func,
        door_opening_func=door_opening_func,
        use_coupled_solver=True,
        verbose=verbose,
    )
    ch.run_simulation()

    # --- Collect results ---
    results = _extract_results(ch)
    return results


def _extract_results(ch) -> dict:
    """Pull simulation outputs into a JSON-serializable dict."""
    n_segs = ch.total_segments

    # Segment midpoint heights [m] for plotting x-axis
    seg_heights = []
    cumulative = 0.0
    for seg in range(n_segs):
        seg_len = ch.segment_lengths[seg]
        seg_heights.append(cumulative + seg_len / 2)
        cumulative += seg_len

    # Pressure profile at final timestep
    dp = ch.get_stack_differential_pressure(timestep=-1)

    results = {
        # Time series (1-D)
        'time_s': ch.time_array.tolist(),
        'mass_flow_kg_s': ch.mass_flow_rate.tolist(),

        # Spatial profiles at every saved timestep (2-D: time x segments)
        'T_gas_K': ch.T_gas.tolist(),
        'T_wall_K': ch.T_wall.tolist(),
        'velocity_m_s': ch.velocity.tolist(),
        'pressure_Pa': ch.pressure.tolist(),
        'density_kg_m3': ch.density.tolist(),

        # Convenience: inlet/exit temps over time (1-D)
        'inlet_temp_K': ch.T_gas[:, 0].tolist(),
        'exit_temp_K': ch.T_gas[:, -1].tolist(),

        # Geometry
        'segment_heights_m': seg_heights,
        'segment_lengths_m': ch.segment_lengths.tolist(),
        'total_segments': n_segs,

        # Pressure profile at final timestep
        'pressure_profile': {
            'height_m': dp['height_m'].tolist(),
            'dP_Pa': dp['dP_Pa'].tolist(),
            'dP_inWC': dp['dP_inWC'].tolist(),
            'dP_inlet_Pa': float(dp['dP_inlet_Pa']),
            'dP_inlet_inWC': float(dp['dP_inlet_inWC']),
        },

        # Summary scalars
        'summary': {
            'final_mass_flow_kg_s': float(ch.mass_flow_rate[-1]),
            'final_exit_temp_C': float(ch.T_gas[-1, -1] - 273.15),
            'final_inlet_temp_C': float(ch.T_gas[-1, 0] - 273.15),
            'final_velocity_exit_m_s': float(ch.velocity[-1, -1]),
            'total_height_m': float(sum(ch.segment_lengths)),
            'backdraft_detected': bool(getattr(ch, 'backdraft_detected', False)),
        },
    }

    # Door opening history if available
    if hasattr(ch, 'door_opening_history') and ch.door_opening_history is not None:
        results['door_opening_pct'] = ch.door_opening_history.tolist()

    return results


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_simulation.py <config.json> [--output results.json]")
        sys.exit(1)

    config_path = sys.argv[1]
    output_path = None
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_path = sys.argv[idx + 1]

    with open(config_path, 'r') as f:
        config = json.load(f)

    results = run_simulation(config)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f)
        print(f"Results written to {output_path}")
    else:
        # Print summary to stdout, full results are too large
        s = results['summary']
        print(f"Exit temp:  {s['final_exit_temp_C']:.1f} C")
        print(f"Mass flow:  {s['final_mass_flow_kg_s']:.4f} kg/s")
        print(f"Exit vel:   {s['final_velocity_exit_m_s']:.2f} m/s")
        print(f"Draft:      {results['pressure_profile']['dP_inlet_inWC']:.4f} inWC")
        print(f"Backdraft:  {s['backdraft_detected']}")
        # Write full results to file anyway
        with open('results.json', 'w') as f:
            json.dump(results, f)
        print("Full results written to results.json")


if __name__ == '__main__':
    main()
