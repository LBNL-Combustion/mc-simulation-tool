"""
Backdraft Detection Analysis from Experimental Data

Analyzes pressure and temperature data to identify backdraft conditions.
Backdraft indicators:
1. Pitot differential pressure goes negative (reversed flow)
2. Gauge pressure at TOP becomes positive (pressure pushing into chimney)
3. Pressure gradient inversion (normal draft: lower at top; backdraft: higher at top)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'dev' / 'exp_data'))

import numpy as np
import pandas as pd
from experiment_loader import ExperimentLoader


def analyze_backdraft_conditions(loader, test_id, window_size=10):
    """
    Analyze a test for backdraft conditions.

    Args:
        loader: ExperimentLoader instance
        test_id: Test ID to analyze
        window_size: Rolling window for smoothing (seconds)

    Returns:
        dict with backdraft analysis results
    """
    df, config = loader.load_test(test_id)
    col_mapping = config['column_mapping']['pressure']

    # Get column names
    top_col = col_mapping['TOP']['column']
    amb_col = col_mapping['ambient']['column']
    pitot_dp_col = col_mapping.get('pitot_dp', {}).get('column')
    tc1_col = col_mapping['TC1']['column']

    # Calculate gauge pressure at TOP (relative to ambient)
    df['top_gauge_Pa'] = df[top_col] - df[amb_col]

    # Calculate gauge pressure at TC1 (base)
    df['tc1_gauge_Pa'] = df[tc1_col] - df[amb_col]

    # Pressure gradient: normally negative (lower pressure at top = good draft)
    # Positive gradient indicates potential backdraft conditions
    df['pressure_gradient_Pa'] = df['top_gauge_Pa'] - df['tc1_gauge_Pa']

    # Smooth with rolling window
    df['top_gauge_smooth'] = df['top_gauge_Pa'].rolling(window=window_size, center=True).mean()
    df['gradient_smooth'] = df['pressure_gradient_Pa'].rolling(window=window_size, center=True).mean()

    if pitot_dp_col and pitot_dp_col in df.columns:
        df['pitot_dp_smooth'] = df[pitot_dp_col].rolling(window=window_size, center=True).mean()

    # Identify backdraft events
    # Criteria 1: TOP gauge pressure > 0 (pressure pushing into chimney)
    # Criteria 2: Pitot differential < 0 (reversed flow)
    # Criteria 3: Pressure gradient > 0 (inverted)

    results = {
        'test_id': test_id,
        'name': config['name'],
        'chimney_type': config['chimney_type'],
        'conditions': config['conditions'],
        'door_events': config.get('door_events', {}),
        'data': df,
    }

    # Analyze during test period (from test start)
    test_mask = df['time_from_test_s'] >= 0
    test_data = df[test_mask].copy()

    # Find backdraft events based on positive TOP gauge pressure
    backdraft_threshold = 0.5  # Pa - consider positive gauge as backdraft risk
    backdraft_mask = test_data['top_gauge_smooth'] > backdraft_threshold

    results['backdraft_events'] = []
    if backdraft_mask.any():
        # Find contiguous backdraft periods
        backdraft_periods = find_contiguous_periods(
            test_data['time_from_test_s'].values,
            backdraft_mask.values
        )
        results['backdraft_events'] = backdraft_periods

    # Also check pitot differential for flow reversal
    if pitot_dp_col and pitot_dp_col in df.columns:
        flow_reversal_mask = test_data['pitot_dp_smooth'] < -0.05  # Pa
        results['flow_reversal_events'] = find_contiguous_periods(
            test_data['time_from_test_s'].values,
            flow_reversal_mask.values
        )

    # Statistics
    results['stats'] = {
        'top_gauge_mean': test_data['top_gauge_Pa'].mean(),
        'top_gauge_std': test_data['top_gauge_Pa'].std(),
        'top_gauge_min': test_data['top_gauge_Pa'].min(),
        'top_gauge_max': test_data['top_gauge_Pa'].max(),
        'gradient_mean': test_data['pressure_gradient_Pa'].mean(),
        'gradient_std': test_data['pressure_gradient_Pa'].std(),
        'pct_positive_gauge': (test_data['top_gauge_Pa'] > 0).mean() * 100,
        'pct_backdraft_risk': (test_data['top_gauge_Pa'] > backdraft_threshold).mean() * 100,
    }

    if pitot_dp_col and pitot_dp_col in df.columns:
        results['stats']['pitot_dp_mean'] = test_data[pitot_dp_col].mean()
        results['stats']['pitot_dp_min'] = test_data[pitot_dp_col].min()
        results['stats']['pct_negative_pitot'] = (test_data[pitot_dp_col] < 0).mean() * 100

    return results


def find_contiguous_periods(times, mask):
    """Find contiguous True periods in a boolean mask."""
    periods = []
    in_period = False
    start_time = None

    for i, (t, m) in enumerate(zip(times, mask)):
        if m and not in_period:
            # Start of period
            in_period = True
            start_time = t
        elif not m and in_period:
            # End of period
            in_period = False
            periods.append({
                'start_s': start_time,
                'end_s': times[i-1] if i > 0 else t,
                'duration_s': (times[i-1] if i > 0 else t) - start_time
            })

    # Handle period that extends to end
    if in_period:
        periods.append({
            'start_s': start_time,
            'end_s': times[-1],
            'duration_s': times[-1] - start_time
        })

    return periods


def analyze_early_phase(loader, test_id, early_window=(0, 300)):
    """
    Analyze the early phase (first 5 minutes) for backdraft risk.
    This is typically when backdraft is most likely before draft establishes.
    """
    df, config = loader.load_test(test_id)
    col_mapping = config['column_mapping']['pressure']
    door_events = config.get('door_events', {}).get('offsets_s', {})

    # Get door timing
    door_open_key = 'door_to_2in' if 'door_to_2in' in door_events else 'door_to_3in'
    t_door_cracked = door_events.get(door_open_key, 60)
    t_door_closed = door_events.get('door_closed', 180)

    # Get column names
    top_col = col_mapping['TOP']['column']
    amb_col = col_mapping['ambient']['column']
    pitot_dp_col = col_mapping.get('pitot_dp', {}).get('column')

    # Filter to early phase
    mask = (df['time_from_test_s'] >= early_window[0]) & (df['time_from_test_s'] <= early_window[1])
    early_data = df[mask].copy()

    # Calculate gauge pressure
    early_data['top_gauge_Pa'] = early_data[top_col] - early_data[amb_col]

    # Analyze by phase
    phases = {
        'pre_door_crack': (0, t_door_cracked),
        'door_cracked': (t_door_cracked, t_door_closed),
        'door_closed': (t_door_closed, early_window[1]),
    }

    phase_stats = {}
    for phase_name, (t_start, t_end) in phases.items():
        phase_mask = (early_data['time_from_test_s'] >= t_start) & (early_data['time_from_test_s'] < t_end)
        phase_data = early_data[phase_mask]

        if len(phase_data) > 0:
            phase_stats[phase_name] = {
                'duration_s': t_end - t_start,
                'top_gauge_mean': phase_data['top_gauge_Pa'].mean(),
                'top_gauge_std': phase_data['top_gauge_Pa'].std(),
                'top_gauge_min': phase_data['top_gauge_Pa'].min(),
                'top_gauge_max': phase_data['top_gauge_Pa'].max(),
                'pct_positive': (phase_data['top_gauge_Pa'] > 0).mean() * 100,
                'pct_backdraft_risk': (phase_data['top_gauge_Pa'] > 0.5).mean() * 100,
            }

            if pitot_dp_col and pitot_dp_col in early_data.columns:
                phase_stats[phase_name]['pitot_dp_mean'] = phase_data[pitot_dp_col].mean()
                phase_stats[phase_name]['pct_negative_pitot'] = (phase_data[pitot_dp_col] < 0).mean() * 100

    return {
        'test_id': test_id,
        'door_events': door_events,
        'phases': phase_stats,
    }


def print_backdraft_report(results):
    """Print a formatted backdraft analysis report."""
    print("\n" + "="*80)
    print(f"BACKDRAFT ANALYSIS: {results['name']} (Test {results['test_id']})")
    print("="*80)

    print(f"\nChimney: {results['chimney_type']}")
    print(f"Conditions: T_indoor={results['conditions'].get('T_indoor_C')}°C, "
          f"T_outdoor={results['conditions'].get('T_outdoor_C')}°C, "
          f"Wind={results['conditions'].get('wind_speed_m_s')} m/s")

    stats = results['stats']
    print(f"\n--- TOP Sensor Gauge Pressure Statistics ---")
    print(f"  Mean:   {stats['top_gauge_mean']:+.2f} Pa")
    print(f"  Std:    {stats['top_gauge_std']:.2f} Pa")
    print(f"  Min:    {stats['top_gauge_min']:+.2f} Pa")
    print(f"  Max:    {stats['top_gauge_max']:+.2f} Pa")
    print(f"  % Positive (>0 Pa): {stats['pct_positive_gauge']:.1f}%")
    print(f"  % Backdraft Risk (>0.5 Pa): {stats['pct_backdraft_risk']:.1f}%")

    if 'pitot_dp_mean' in stats:
        print(f"\n--- Pitot Differential Pressure ---")
        print(f"  Mean:   {stats['pitot_dp_mean']:+.3f} Pa")
        print(f"  Min:    {stats['pitot_dp_min']:+.3f} Pa")
        print(f"  % Negative (reversed flow): {stats['pct_negative_pitot']:.1f}%")

    print(f"\n--- Pressure Gradient (TOP - TC1) ---")
    print(f"  Mean:   {stats['gradient_mean']:+.2f} Pa")
    print(f"  Std:    {stats['gradient_std']:.2f} Pa")
    print(f"  (Negative = normal draft pulling up, Positive = backdraft pushing down)")

    backdraft_events = results.get('backdraft_events', [])
    if backdraft_events:
        print(f"\n--- BACKDRAFT EVENTS DETECTED ({len(backdraft_events)} events) ---")
        total_duration = sum(e['duration_s'] for e in backdraft_events)
        print(f"  Total backdraft duration: {total_duration:.1f}s")
        for i, event in enumerate(backdraft_events[:10]):  # Show first 10
            print(f"  Event {i+1}: {event['start_s']:.1f}s - {event['end_s']:.1f}s "
                  f"(duration: {event['duration_s']:.1f}s)")
        if len(backdraft_events) > 10:
            print(f"  ... and {len(backdraft_events) - 10} more events")
    else:
        print(f"\n  No significant backdraft events detected based on TOP gauge > 0.5 Pa")

    flow_reversal = results.get('flow_reversal_events', [])
    if flow_reversal:
        print(f"\n--- FLOW REVERSAL EVENTS ({len(flow_reversal)} events) ---")
        total_duration = sum(e['duration_s'] for e in flow_reversal)
        print(f"  Total flow reversal duration: {total_duration:.1f}s")


def print_early_phase_report(results):
    """Print early phase analysis report."""
    print(f"\n--- Early Phase Analysis (Test {results['test_id']}) ---")

    door_events = results['door_events']
    door_open_key = 'door_to_2in' if 'door_to_2in' in door_events else 'door_to_3in'
    print(f"  Door cracked at: {door_events.get(door_open_key, 'N/A')}s")
    print(f"  Door closed at: {door_events.get('door_closed', 'N/A')}s")

    for phase_name, stats in results['phases'].items():
        print(f"\n  {phase_name}:")
        print(f"    Duration: {stats['duration_s']:.0f}s")
        print(f"    TOP gauge mean: {stats['top_gauge_mean']:+.2f} Pa")
        print(f"    % Backdraft risk: {stats['pct_backdraft_risk']:.1f}%")
        if 'pct_negative_pitot' in stats:
            print(f"    % Negative pitot: {stats['pct_negative_pitot']:.1f}%")


def identify_backdraft_predictors(loader):
    """
    Analyze all tests to identify predictors of backdraft conditions.
    """
    print("\n" + "="*80)
    print("BACKDRAFT PREDICTOR ANALYSIS - ALL TESTS")
    print("="*80)

    all_results = []

    for test_id in loader.test_ids:
        try:
            results = analyze_backdraft_conditions(loader, test_id)
            early = analyze_early_phase(loader, test_id)

            # Combine key metrics
            combined = {
                'test_id': test_id,
                'name': results['name'],
                'chimney_ft': results['chimney_type'].replace('ft', ''),
                'T_outdoor_C': results['conditions'].get('T_outdoor_C'),
                'T_indoor_C': results['conditions'].get('T_indoor_C'),
                'delta_T_C': results['conditions'].get('T_indoor_C', 20) - results['conditions'].get('T_outdoor_C', 15),
                'wind_m_s': results['conditions'].get('wind_speed_m_s'),
                'top_gauge_mean': results['stats']['top_gauge_mean'],
                'pct_backdraft_risk': results['stats']['pct_backdraft_risk'],
                'n_backdraft_events': len(results.get('backdraft_events', [])),
            }

            if 'pitot_dp_mean' in results['stats']:
                combined['pitot_dp_mean'] = results['stats']['pitot_dp_mean']
                combined['pct_negative_pitot'] = results['stats']['pct_negative_pitot']

            all_results.append(combined)

        except Exception as e:
            print(f"Warning: Failed to analyze {test_id}: {e}")

    # Print summary table
    print("\n" + "-"*100)
    print(f"{'Test':<12} {'Chimney':<8} {'ΔT(°C)':<8} {'Wind':<8} {'TOP_gauge':<12} {'%Risk':<8} {'%NegPitot':<10} {'Events':<8}")
    print("-"*100)

    for r in all_results:
        print(f"{r['name']:<12} {r['chimney_ft']:<8} {r['delta_T_C']:<8} "
              f"{r['wind_m_s']:<8} {r['top_gauge_mean']:+.2f} Pa    "
              f"{r['pct_backdraft_risk']:<8.1f} {r.get('pct_negative_pitot', 0):<10.1f} "
              f"{r['n_backdraft_events']:<8}")

    return all_results


def main():
    """Run complete backdraft analysis."""
    # Initialize loader
    exp_data_path = Path(__file__).parent.parent / 'dev' / 'exp_data'
    loader = ExperimentLoader(metadata_path=exp_data_path / 'experiments_metadata.json')

    print("Analyzing experimental data for backdraft conditions...")
    print(f"Available tests: {loader.test_ids}")

    # Analyze each test
    for test_id in loader.test_ids:
        try:
            results = analyze_backdraft_conditions(loader, test_id)
            print_backdraft_report(results)

            early = analyze_early_phase(loader, test_id)
            print_early_phase_report(early)

        except Exception as e:
            print(f"\nError analyzing test {test_id}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-test analysis
    identify_backdraft_predictors(loader)

    print("\n" + "="*80)
    print("BACKDRAFT DETECTION CRITERIA SUMMARY")
    print("="*80)
    print("""
Key indicators for backdraft detection:

1. TOP GAUGE PRESSURE > 0 Pa
   - Positive gauge pressure at chimney top indicates air pushing INTO chimney
   - Risk threshold: > 0.5 Pa sustained

2. PITOT DIFFERENTIAL < 0 Pa
   - Negative pitot ΔP indicates reversed (downward) flow
   - Strong indicator of active backdraft

3. PRESSURE GRADIENT INVERSION
   - Normal draft: P_top < P_bottom (pressure decreases with height)
   - Backdraft: P_top > P_bottom (inverted gradient)

4. CONDITIONS FAVORING BACKDRAFT:
   - Low ΔT (indoor-outdoor): Less buoyancy drive
   - High wind speed: Wind-induced pressure fluctuations
   - Shorter chimney: Less draft height
   - Door closed: Restricted air supply can stall flow
   - Cold start: Chimney not pre-heated

5. TIMING:
   - Early phase (0-60s): Most vulnerable before draft establishes
   - Door closure: Transition period when airflow restricted
""")


if __name__ == '__main__':
    main()
