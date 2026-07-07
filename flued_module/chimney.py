"""
Main Chimney class for flue simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .physics import PhysicalProperties
from .comprehensive_heat_transfer import ComprehensiveHeatTransfer
from .coupled_solver import CoupledFlowThermalSolver
from .diagnostics_recorder import DiagnosticsRecorder
from .room_plenum import RoomPlenum, RoomPlenumConfig
from .early_diagnostics import EarlyTimeDiagnostics
from .chimney_geometry_plot import plot_chimney_geometry, plot_chimney_snapshot
from .wind_cap_effects import WindCapConfig, wind_cap_deltaP
from .flow.elbows import base_K
import pandas as pd
import time as _time                       # TEMP: profiling timer
from ._perf_counters import COUNTERS as _PC, reset as _pc_reset, summary as _pc_summary  # TEMP: profiling

@dataclass
class InletRestrictionConfig:
    """Inlet flow restriction (door, damper, etc.)

    Attributes:
        enable: Activate restriction
        door_type: 'swivel' or 'sliding'
        door_opening_percent: 0-100%
        door_width_ft: Door width [ft]
        door_height_ft: Door height [ft]
        door_cracked_percent: Enables dynamic door schedule if set
        door_wide_open_time_s: Duration wide open [s] (default 90 = 1.5 min)
        door_closed_time_s: Time to close the door [s] (default 210 = 3.5 min;
            set to None to leave the door cracked indefinitely)
        Cd: Discharge coefficient
        A_leak_m2: Leakage area when closed [m²]
        max_effective_area_m2: Cap A_eff at downstream restriction [m²]
        K_loss_override: Manual K-loss override
        door_area_m2: Computed full door area [m²]
        A_eff_raw_m2: Raw effective area from geometry [m²]
        effective_area_m2: Total effective area [m²]
        K_equiv: Equivalent loss coefficient = 1/Cd²
    """
    enable: bool = False
    door_type: str = 'swivel'  # 'swivel' or 'sliding'
    door_opening_percent: float = 100.0  # 0-100%
    door_width_ft: float = 1.0  # ft
    door_height_ft: float = 1.0  # ft

    # For more control, use door_opening_func in Chimney instead
    door_cracked_percent: Optional[float] = None
    door_wide_open_time_s: float = 90.0   # 1.5 min: door cracks open
    door_closed_time_s: Optional[float] = 210.0  # 3.5 min: door fully closes

    Cd: float = 0.62

    # Default 0.0045 m² = 45 cm², gaps around a closed door
    A_leak_m2: float = 0.0045

    # Hard floor: prevents singular resistance when A_eff -> 0. Set to 0 to disable.
    inlet_area_min_m2: float = 0.003  # m²

    # Cap A_eff at downstream restriction (tee/pipe). None = no cap.
    max_effective_area_m2: Optional[float] = None

    # Stove body K-loss: fuel bed + baffle + flue collar entry
    # Total K_stove ~ 4-6 for typical woodstove
    # R_stove = K_stove / (2 * rho * A_pipe^2). Set to 0 to disable.
    K_stove: float = 0.0

    # Pipe entrance K-loss (separate from door orifice).
    # When door is open, A_door >> A_pipe and the orifice R vanishes;
    # the pipe entrance loss then dominates the inlet resistance.
    # Typical values: 0.5 (sharp-edged entry), 0.2 (rounded).
    K_entrance: float = 0.0

    # "linear": A_eff = A_eff_raw + A_leak
    # "quadrature": A_eff = sqrt(A_eff_raw^2 + A_leak^2) -- smoother near A_eff_raw=0
    inlet_area_mode: str = "linear"

    # "none": no additional cap; "pipe": cap at first part's pipe area
    inlet_area_cap_mode: str = "none"

    K_loss_override: Optional[float] = None  # bypasses Cd calculation

    # Computed values (set during initialization)
    door_area_m2: float = 0.0  # m²
    A_eff_raw_m2: float = 0.0  # m²
    effective_area_m2: float = 0.0  # m²
    effective_area_uncapped_m2: float = 0.0  # m² (before pipe-area cap)
    K_equiv: float = 0.0  # = 1/Cd² unless K_loss_override


@dataclass
class CatalyticBypassConfig:
    """Catalytic bypass damper configuration.

    Attributes:
        enable: Activate bypass
        bypass_state: 'open', 'closed', or 0-100 float
        segment_index: Segment containing catalyst (0-indexed)
        K_bypass_open: K-loss when bypass open (~0.3-1.0)
        K_catalyst_closed: K-loss through catalyst (~2-5)
        K_effective: Computed K-loss based on bypass_state
        bypass_fraction: 0.0 (closed) to 1.0 (open)
    """
    enable: bool = False
    bypass_state: Any = 'open'  # 'open', 'closed', or 0-100 float
    segment_index: int = 0
    K_bypass_open: float = 0.5
    K_catalyst_closed: float = 3.0

    # Computed values (set during initialization)
    K_effective: float = 0.0
    bypass_fraction: float = 1.0  # 0=closed, 1=open

@dataclass
class LeakageConfig:
    """Distributed pipe joint leakage (VENT-II style orifice model).

    Attributes:
        enable: Activate leakage
        total_leakage_area_cm2: Total leakage area [cm²]
        black_pipe_fraction: Fraction of leakage in black stove pipe (0-1)
        insulated_fraction: Fraction of leakage in insulated pipe (0-1)
        Cd_leak: Discharge coefficient for leaks
    """
    enable: bool = False
    total_leakage_area_cm2: float = 10.0  # Total leakage area in cm²
    black_pipe_fraction: float = 0.6     # 60% of leakage in black pipe
    insulated_fraction: float = 0.4      # 40% of leakage in insulated pipe
    Cd_leak: float = 0.61                # Discharge coefficient for leaks


@dataclass
class ChimneyPart:
    """Single or triple-wall chimney section configuration."""
    # BASIC CONFIGURATION
    type: str  # 'single' or 'triple'
    inner_diameter_ft: float  # ft
    height_ft: float  # ft
    z_ft: float  # ft, elevation above reference
    orientation: str  # 'vertical' or 'horizontal'
    T_ambient_K: float  # K
    Pa: float  # Pa
    num_segments: int

    # SINGLE WALL PROPERTIES
    thickness_w_m: float = 0.001  # m
    density_w_kg_m3: float = 7850  # kg/m³, steel default
    cp_w_J_kg_K: float = 460  # J/(kg·K)
    k_w_W_m_K: float = 50  # W/(m·K)
    epsilon_w: float = 0.8

    # TRIPLE WALL PROPERTIES
    # Inner wall (liner)
    thickness_inner_m: float = 0.0008  # m, stainless steel
    density_inner_kg_m3: float = 8000  # kg/m³, stainless steel
    cp_inner_J_kg_K: float = 500  # J/(kg·K)
    k_inner_W_m_K: float = 16  # W/(m·K), stainless steel
    epsilon_inner: float = 0.3  # polished metal

    # Middle layer (insulation)
    thickness_middle_m: float = 0.025  # m
    density_middle_kg_m3: float = 100  # kg/m³
    cp_middle_J_kg_K: float = 840  # J/(kg·K)
    k_middle_W_m_K: float = 0.04  # W/(m·K)

    # Outer wall (structural)
    thickness_outer_m: float = 0.001  # m
    density_outer_kg_m3: float = 7850  # kg/m³
    cp_outer_J_kg_K: float = 460  # J/(kg·K)
    k_outer_W_m_K: float = 50  # W/(m·K)
    epsilon_outer: float = 0.8

    # AIR GAPS (triple wall)
    air_gap_inner_m: float = 0.0  # m
    air_gap_outer_m: float = 0.0  # m

    # GENERAL PROPERTIES
    film_w: float = 5e-5  # m, max condensate film thickness
    has_fittings: bool = False
    roughness_m: float = 1e-6  # m
    humidity_ratio: float = 0.01  # kg water / kg dry air

    # CORRUGATION (optional). None = smooth pipe with base roughness_m.
    corrugation: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Convert units, compute geometry, and validate."""
        self.inner_diameter_m = self.inner_diameter_ft * 0.3048
        self.height_m = self.height_ft * 0.3048
        self.z_m = self.z_ft * 0.3048

        self.inner_area_m2 = np.pi * (self.inner_diameter_m / 2) ** 2
        self.inner_perimeter_m = np.pi * self.inner_diameter_m

        if self.type == 'single':
            self.outer_diameter_m = self.inner_diameter_m + 2 * self.thickness_w_m
        elif self.type == 'triple':
            total_thickness = (self.thickness_inner_m + self.air_gap_inner_m +
                             self.thickness_middle_m + self.air_gap_outer_m +
                             self.thickness_outer_m)
            self.outer_diameter_m = self.inner_diameter_m + 2 * total_thickness

            self.liner_outer_diameter_m = self.inner_diameter_m + 2 * self.thickness_inner_m
            self.middle_inner_diameter_m = self.liner_outer_diameter_m + 2 * self.air_gap_inner_m
            self.middle_outer_diameter_m = self.middle_inner_diameter_m + 2 * self.thickness_middle_m
            self.outer_inner_diameter_m = self.middle_outer_diameter_m + 2 * self.air_gap_outer_m
        else:
            raise ValueError(f"Unknown wall type: {self.type}. Must be 'single' or 'triple'")

        if self.num_segments < 1:
            raise ValueError("num_segments must be at least 1")
        if self.inner_diameter_m <= 0:
            raise ValueError("inner_diameter must be positive")
        if self.height_m <= 0:
            raise ValueError("height must be positive")

        self._process_corrugation_properties()

    def _process_corrugation_properties(self):
        """Fold corrugation effects into self.roughness_m."""
        if self.corrugation is None:
            return

        corr_type = self.corrugation.get('type', 'none').lower()

        if corr_type == 'none':
            return
        elif corr_type == 'spiral':
            self._process_spiral_corrugation()
        elif corr_type == 'annular':
            self._process_annular_corrugation()
        else:
            raise ValueError(f"Unknown corrugation type: {corr_type}. "
                           f"Must be 'none', 'spiral', or 'annular'")

    def _process_spiral_corrugation(self):
        """Convert spiral corrugation geometry into equivalent roughness."""
        from .corrugation_effects import (
            SpiralCorrugationProperties,
            compute_spiral_corrugation_effects
        )

        corr = self.corrugation

        depth_m = corr.get('depth_mm', 3.0) / 1000.0
        pitch_m = corr.get('pitch_mm', 180.0) / 1000.0
        num_starts = corr.get('num_starts', 1)
        base_roughness_m = corr.get('base_roughness_mm', 0.001) / 1000.0

        has_lock_seam = corr.get('has_lock_seam', True)
        seam_protrusion_m = corr.get('seam_protrusion_mm', 2.0) / 1000.0
        seam_width_m = corr.get('seam_width_mm', 8.0) / 1000.0
        K_seam = corr.get('K_seam_per_crossing', 0.05)
        K_lock = corr.get('K_lock_seam', 0.03)

        enable_swirl = corr.get('enable_spiral_swirl', False)
        swirl_friction_cap = corr.get('swirl_friction_cap', 0.15)
        enable_HT = corr.get('enable_spiral_HT_enhancement', False)
        HT_enhancement_cap = corr.get('HT_enhancement_cap', 0.25)

        spiral_props = SpiralCorrugationProperties(
            depth_m=depth_m,
            pitch_m=pitch_m,
            num_starts=num_starts,
            base_roughness_m=base_roughness_m,
            has_lock_seam=has_lock_seam,
            seam_protrusion_m=seam_protrusion_m,
            seam_width_m=seam_width_m,
            K_seam_per_crossing=K_seam,
            K_lock_seam=K_lock
        )

        segment_length = self.height_m / self.num_segments

        effects = compute_spiral_corrugation_effects(
            diameter_m=self.inner_diameter_m,
            length_m=segment_length,
            spiral_props=spiral_props,
            representative_Re=10000,  # Typical turbulent Re for flues
            enable_swirl=enable_swirl,
            swirl_friction_cap=swirl_friction_cap,
            enable_HT=enable_HT,
            HT_enhancement_cap=HT_enhancement_cap
        )

        self.corrugation_effects = effects
        self.roughness_m = effects['equivalent_roughness_m']

        version_str = " (v2.0)" if (enable_swirl or enable_HT) else ""
        print(f"\n{'='*70}")
        print(f"SPIRAL CORRUGATION PREPROCESSING{version_str} - Part with D={self.inner_diameter_m*1000:.1f}mm")
        print(f"{'='*70}")
        print(f"  Corrugation depth:        {depth_m*1000:.2f} mm")
        print(f"  Helical pitch:            {pitch_m*1000:.1f} mm")
        print(f"  Number of starts:         {num_starts}")
        print(f"  Helix angle:              {effects['helix_angle_deg']:.1f}°")
        print(f"  Swirl index S:            {effects['swirl_index']:.3f}")
        print(f"  Spiral factor c_spiral:   {effects['spiral_factor']:.3f}")
        print(f"  Base roughness:           {base_roughness_m*1e6:.2f} µm")
        print(f"  Effective roughness:      {effects['effective_roughness_m']*1e6:.1f} µm")
        print(f"  Seam crossings/segment:   {effects['n_seams']:.2f}")
        print(f"  Total K from seams:       {effects['K_total_seams']:.4f}")
        print(f"  Lock seam K:              {effects['K_lock_seam']:.4f}")

        if enable_swirl:
            print(f"  Swirl friction mult m_f:  {effects['friction_multiplier']:.3f} (enabled)")
        else:
            print(f"  Swirl friction mult m_f:  1.000 (disabled)")

        if enable_HT:
            print(f"  Heat transfer mult M_h:   {effects['Mh']:.3f} (enabled)")
        else:
            print(f"  Heat transfer mult M_h:   1.000 (disabled)")

        print(f"  Equivalent roughness:     {effects['equivalent_roughness_m']*1e6:.1f} µm")
        print(f"  Friction factor (base):   {effects['friction_factor_corrugation']:.5f}")
        print(f"  Friction factor (final):  {effects['friction_factor_total']:.5f}")
        print(f"{'='*70}\n")

    def _process_annular_corrugation(self):
        """Convert annular corrugation geometry into equivalent roughness."""
        from .corrugation_effects import (
            CorrugationProperties,
            CorrugatedPipeFriction
        )

        corr = self.corrugation

        depth_m = corr.get('depth_mm', 3.0) / 1000.0
        pitch_m = corr.get('pitch_mm', 25.0) / 1000.0
        base_roughness_m = corr.get('base_roughness_mm', 0.001) / 1000.0

        corr_props = CorrugationProperties(
            depth_m=depth_m,
            pitch_m=pitch_m,
            base_roughness_m=base_roughness_m
        )

        friction_calc = CorrugatedPipeFriction()

        representative_Re = 10000  # Typical turbulent Re for flues
        f_corrugated = friction_calc.calculate_corrugated_friction_factor(
            Re=representative_Re,
            D=self.inner_diameter_m,
            corrugation=corr_props
        )

        f_smooth = friction_calc._colebrook_white(representative_Re, base_roughness_m / self.inner_diameter_m)

        # Invert Colebrook-White to find equivalent roughness (Newton-Raphson)
        eps_eq = base_roughness_m
        for iteration in range(15):
            f_current = friction_calc._colebrook_white(representative_Re, eps_eq / self.inner_diameter_m)

            if abs(f_current - f_corrugated) < 1e-6:
                break

            deps = eps_eq * 0.001 + 1e-9
            f_perturbed = friction_calc._colebrook_white(representative_Re, (eps_eq + deps) / self.inner_diameter_m)
            df_deps = (f_perturbed - f_current) / deps

            if abs(df_deps) > 1e-12:
                eps_eq = eps_eq + (f_corrugated - f_current) / df_deps

            eps_eq = max(eps_eq, base_roughness_m)
            eps_eq = min(eps_eq, self.inner_diameter_m * 0.05)

        self.roughness_m = eps_eq

        self.corrugation_effects = {
            'type': 'annular',
            'depth_m': depth_m,
            'pitch_m': pitch_m,
            'depth_to_pitch_ratio': corr_props.depth_to_pitch_ratio,
            'base_roughness_m': base_roughness_m,
            'equivalent_roughness_m': eps_eq,
            'friction_factor_smooth': f_smooth,
            'friction_factor_corrugated': f_corrugated,
            'enhancement_factor': f_corrugated / f_smooth if f_smooth > 0 else 1.0
        }

        print(f"\n{'='*70}")
        print(f"ANNULAR CORRUGATION PREPROCESSING - Part with D={self.inner_diameter_m*1000:.1f}mm")
        print(f"{'='*70}")
        print(f"  Corrugation depth:        {depth_m*1000:.2f} mm")
        print(f"  Corrugation pitch:        {pitch_m*1000:.1f} mm")
        print(f"  Depth/pitch ratio:        {corr_props.depth_to_pitch_ratio:.3f}")
        print(f"  Base roughness:           {base_roughness_m*1e6:.2f} µm")
        print(f"  Equivalent roughness:     {eps_eq*1e6:.1f} µm")
        print(f"  Friction factor (smooth): {f_smooth:.5f}")
        print(f"  Friction factor (corr):   {f_corrugated:.5f}")
        print(f"  Enhancement factor:       {f_corrugated/f_smooth:.2f}x")
        print(f"{'='*70}\n")

def _smoothstep(t, t_start, t_end):
    """Hermite smoothstep interpolation 0->1 over [t_start, t_end], C1 continuous.

    Returns 0 for t <= t_start, 1 for t >= t_end, smooth transition between.
    """
    if t_end <= t_start:
        return 1.0 if t >= t_start else 0.0
    x = (t - t_start) / (t_end - t_start)
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def make_smooth_door_schedule(events, ramp_open_to_cracked_s=5.0,
                              ramp_cracked_to_closed_s=1.0):
    """Create a smooth door-opening function from a list of (time, percent) events.

    Instead of instantaneous step changes, transitions use Hermite smoothstep
    ramps. Large transitions (|delta| > 50%) use ``ramp_open_to_cracked_s``,
    small ones use ``ramp_cracked_to_closed_s``.

    Args:
        events: List of (time_s, opening_percent) tuples, sorted by time.
        ramp_open_to_cracked_s: Ramp duration for large transitions (default 5s).
        ramp_cracked_to_closed_s: Ramp duration for small transitions (default 1s).

    Returns:
        Callable ``door_opening_func(t)`` returning opening percent at time *t*.
    """
    events = sorted(events, key=lambda e: e[0])
    if not events:
        raise ValueError("events list must not be empty")

    def door_opening_func(t):
        # Before first event: hold first value
        if t <= events[0][0]:
            return events[0][1]

        # Walk through events to find the active ramp
        for i in range(1, len(events)):
            t_event = events[i][0]
            prev_val = events[i - 1][1]
            new_val = events[i][1]
            delta_pct = abs(new_val - prev_val)

            # Choose ramp duration based on transition size
            if delta_pct > 50.0:
                ramp_dur = ramp_open_to_cracked_s
            else:
                ramp_dur = ramp_cracked_to_closed_s

            # Clamp ramp end to not exceed next event
            if i < len(events) - 1:
                ramp_end = min(t_event + ramp_dur, events[i + 1][0])
            else:
                ramp_end = t_event + ramp_dur

            if t < t_event:
                # Between events[i-1] and events[i], but before the event
                # fires -- hold previous value
                return prev_val

            if t <= ramp_end:
                # Inside the ramp region
                s = _smoothstep(t, t_event, ramp_end)
                return prev_val + s * (new_val - prev_val)

            # Past this ramp; if no more events, fall through to hold last value

        # After last event + ramp: hold last value
        return events[-1][1]

    return door_opening_func


class Chimney:
    """Main chimney simulation class."""
    
    def __init__(self, parts: List[Dict[str, Any]], dt: float = 0.5, total_time: float = 60.0,
                 use_coupled_solver: bool = True, elbows: List[Dict[str, Any]] = None,
                 elbow_method: str = "K_sum", top_boundary: Optional[Dict[str, Any]] = None,
                 inlet_restriction: Optional[Dict[str, Any]] = None,
                 catalytic_bypass: Optional[Dict[str, Any]] = None,
                 door_opening_func: Optional[callable] = None,
                 inlet_temp_csv: Optional[str] = None,
                 inlet_temp_func: Optional[callable] = None,
                 door_temp_dilution: Optional[Dict[str, Any]] = None,
                 room_pressure_offset_Pa = 0.0,  # float or callable(t_s) -> Pa
                 leakage_config: Optional[Dict[str, Any]] = None,
                 enable_diagnostics: bool = False,
                 momentum_config: Optional[Dict[str, Any]] = None,
                 room_plenum: Optional[Dict[str, Any]] = None,
                 fittings_thermal: Optional[Dict[str, Any]] = None,
                 solver_relaxation: Optional[float] = None,
                 solver_alpha_cap: Optional[float] = None,
                 elevation_ft: Optional[float] = 0.0,
                 h_inner_multiplier: float = 1.0,
                 h_outer_multiplier: float = 1.0,
                 external_ht_config: Optional[Dict[str, Any]] = None,
                 verbose: bool = False):
        """Initialize chimney simulation.

        Args:
            parts: List of dicts defining chimney parts
            dt: Time step [s]
            total_time: Simulation duration [s]
            use_coupled_solver: Iterative flow-thermal coupling (more accurate)
            elbows: List of fitting dicts with 'between_segments', 'angle_deg', etc.
            elbow_method: "K_sum" or "equivalent_length"
            inlet_restriction: Dict with door geometry/restriction params
            catalytic_bypass: Dict with bypass damper params
            door_opening_func: callable(t) -> opening_percent for dynamic door
            inlet_temp_csv: CSV path or dict for inlet temperature data
            inlet_temp_func: callable(t) -> T_K for inlet temperature
            enable_diagnostics: Record per-timestep momentum/energy budgets
            momentum_config: Dict or MomentumConfig for momentum physics
            room_plenum: Dict or RoomPlenum for room depressurization
            elevation_ft: Site elevation above sea level [ft]. Sets the single
                ambient pressure the flow solver uses.
                * 0.0 -> ISA sea-level pressure (101325 Pa), applied to all parts.
                * A number (e.g. 787 ~= 98476 Pa) -> ISA pressure at that elevation.
                * None -> use the exact Pa supplied in the per-part dicts (e.g. a
                  measured barometric pressure).
            fittings_thermal: Dict for fitting thermal model config
        """
        self.inlet_temp_csv = inlet_temp_csv
        self.inlet_temp_func = inlet_temp_func
        self.inlet_temp_df = None
        self.inlet_temp_interp = None
        self.dt = dt
        self.total_time = total_time
        self.time_steps = int(total_time / dt)

        # Site ambient pressure -> one site pressure (self.Pa_ambient) for all parts.
        # elevation_ft=0.0 -> ISA sea-level pressure (101325 Pa); a number -> ISA
        # pressure at that elevation; None -> use the per-part Pa values directly.
        self.elevation_ft = elevation_ft
        if elevation_ft is None:
            self.elevation_m = 0.0
            _part_pas = [p.get('Pa') for p in parts if p.get('Pa') is not None]
            self.Pa_ambient = float(_part_pas[0]) if _part_pas else 101325.0
        else:
            self.elevation_m = elevation_ft * 0.3048
            self.Pa_ambient = PhysicalProperties.pressure_from_elevation(self.elevation_m)
        parts = [{**part, 'Pa': self.Pa_ambient} for part in parts]

        self.parts = [ChimneyPart(**part) for part in parts]

        self.elbows = elbows if elbows is not None else []
        self.elbow_method = elbow_method

        self._process_elbows()

        self.total_segments = sum(part.num_segments for part in self.parts)

        self._print_geometry_summary()

        has_triple_wall = any(part.get('type', 'single') == 'triple' for part in parts)
        if has_triple_wall and dt > 1.0:
            print(f"Warning: Large time step ({dt}s) detected with triple wall parts.")
            print("Consider using a smaller time step if numerical instabilities occur.")

        self.physics = PhysicalProperties()
        _ext = external_ht_config or {}
        self.comprehensive_ht = ComprehensiveHeatTransfer(
            h_inner_multiplier=h_inner_multiplier,
            h_outer_multiplier=h_outer_multiplier,
            exact_radiation=_ext.get('exact_radiation', False),
            cylinder_nc_correction=_ext.get('cylinder_nc_correction', False),
            local_nc_length=_ext.get('local_nc_length', False),
            external_air_speed_m_s=_ext.get('external_air_speed_m_s', 0.0))

        self.top_boundary = top_boundary if top_boundary is not None else {}
        self._wind_cfg = self._parse_wind_cap(self.top_boundary)
        self._deltaP_wind_cap_static = None
        self._last_deltaP_wind_cap = 0.0
        self.K_exit = float(self.top_boundary.get('K_exit', 0.0))

        if self._wind_cfg.enable and self._wind_cfg.update_mode == "static":
            V = self._wind_cfg.wind_speed_m_s
            if isinstance(V, (list, tuple)):
                V0 = float(V[0]) if len(V) > 0 else 0.0
            else:
                V0 = float(V)

            Pa_outlet = self.parts[-1].Pa if len(self.parts) > 0 else 101325.0
            self._deltaP_wind_cap_static = wind_cap_deltaP(
                Pa=Pa_outlet,
                T_out_K=self._wind_cfg.air_temp_K,
                V_wind=V0,
                cap_type=self._wind_cfg.cap_type,
                exposure=self._wind_cfg.roof_exposure,
                theta_deg=self._wind_cfg.wind_direction_deg,
                gust_factor=self._wind_cfg.gust_factor,
                Cp_override=self._wind_cfg.Cp_override,
            )
            self._last_deltaP_wind_cap = self._deltaP_wind_cap_static


        self._initialize_inlet_restriction(inlet_restriction)

        # Priority: explicit door_opening_func > door_cracked_percent > static
        self.door_opening_func = door_opening_func

        if (self.inlet_restriction.enable and
            self.inlet_restriction.door_cracked_percent is not None and
            door_opening_func is None):
            cracked_pct = self.inlet_restriction.door_cracked_percent
            wide_open_time = self.inlet_restriction.door_wide_open_time_s
            closed_time = self.inlet_restriction.door_closed_time_s

            events = [(0, 100.0), (wide_open_time, cracked_pct)]
            if closed_time is not None and closed_time > wide_open_time:
                events.append((closed_time, 0.0))

            self.door_opening_func = make_smooth_door_schedule(
                events=events,
                ramp_open_to_cracked_s=5.0,
            )
            if closed_time is not None and closed_time > wide_open_time:
                print(f"[DOOR] Dynamic door schedule: 100% open for {wide_open_time:.0f}s, "
                      f"then {cracked_pct:.0f}% cracked, then closed at {closed_time:.0f}s")
            else:
                print(f"[DOOR] Dynamic door schedule: 100% open for {wide_open_time:.0f}s, "
                      f"then {cracked_pct:.0f}% cracked")

        elif door_opening_func is not None and self.inlet_restriction.enable:
            print(f"[DOOR] Custom door opening function enabled")

        self._initialize_catalytic_bypass(catalytic_bypass)

        self.diagnostics_recorder = DiagnosticsRecorder(enable=enable_diagnostics)

        if room_plenum is not None and isinstance(room_plenum, dict):
            self.room_plenum = RoomPlenum(RoomPlenumConfig(**room_plenum))
        elif room_plenum is not None:
            self.room_plenum = room_plenum
        else:
            self.room_plenum = RoomPlenum(RoomPlenumConfig(enable=False))

        from .coupled_solver import MomentumConfig
        if momentum_config is not None and isinstance(momentum_config, dict):
            self._momentum_config = MomentumConfig(**momentum_config)
        elif momentum_config is not None:
            self._momentum_config = momentum_config
        else:
            self._momentum_config = None

        self.use_coupled_solver = use_coupled_solver
        if use_coupled_solver:
            solver_kwargs = dict(
                diagnostics=self.diagnostics_recorder,
                momentum_config=self._momentum_config,
            )
            if solver_relaxation is not None:
                solver_kwargs['relaxation_factor'] = solver_relaxation
            if solver_alpha_cap is not None:
                solver_kwargs['alpha_cap'] = solver_alpha_cap
            self.coupled_solver = CoupledFlowThermalSolver(**solver_kwargs)
            self.convergence_history = []

        self.diagnostics = EarlyTimeDiagnostics(ambient_temp_K=self.parts[0].T_ambient_K)

        if inlet_temp_csv is not None:
            self._load_inlet_temp_csv(inlet_temp_csv)

        # Door temperature dilution (Jan 27 feature)
        self.door_temp_dilution_enable = False
        self.door_temp_dilution_strength = 0.8  # 80% ambient at 100% door open
        if door_temp_dilution is not None:
            self.door_temp_dilution_enable = door_temp_dilution.get('enable', False)
            self.door_temp_dilution_strength = door_temp_dilution.get('strength', 0.8)
            if self.door_temp_dilution_enable:
                print(f"[DOOR DILUTION] Enabled with strength={self.door_temp_dilution_strength:.2f}")

        self._initialize_leakage_config(leakage_config)

        if self.use_coupled_solver and hasattr(self, 'leakage_config') and self.leakage_config.enable:
            self.coupled_solver.momentum_config.use_distributed_leakage = True

        self.verbose = verbose
        if callable(room_pressure_offset_Pa):
            self._room_pressure_func = room_pressure_offset_Pa
            self.room_pressure_offset_Pa = 0.0
        else:
            self._room_pressure_func = None
            self.room_pressure_offset_Pa = room_pressure_offset_Pa

        self.segment_lengths = np.zeros(self.total_segments)
        seg_idx = 0
        for part in self.parts:
            seg_length = part.height_m / part.num_segments
            for i in range(part.num_segments):
                self.segment_lengths[seg_idx] = seg_length
                seg_idx += 1

        self._compute_segment_positions()

        self._init_fittings_thermal(fittings_thermal)

        self._initialize_arrays()

        self.current_time = 0.0
        self.time_step = 0

        self.backdraft_detected = False
        self.backdraft_start_time = None
        self.backdraft_end_time = None
        self.flow_reversal_count = 0

    def run_simulation(self, save_interval: int = 1):
        """Run the time-stepping simulation loop."""
        print(f"Starting comprehensive heat transfer simulation:")
        print(f"  {self.total_segments} segments, {self.time_steps} time steps")
        print(f"  Time step: {self.dt}s, Total time: {self.total_time}s")

        # TEMP: profiling -- reset counters and start wall-clock timer
        _pc_reset()
        _t0 = _time.perf_counter()

        for step in range(1, self.time_steps + 1):
            self.time_step = step
            self.current_time = step * self.dt
            _PC['timesteps'] += 1  # TEMP: profiling

            # === DOOR OPENING UPDATE ===
            if self.door_opening_func is not None and self.inlet_restriction.enable:
                new_opening = self.door_opening_func(self.current_time)
                old_opening = self.door_opening_history[step - 1]
                self._update_door_opening(new_opening)
                self.door_opening_history[step] = new_opening
                self.A_eff_history[step] = self.inlet_restriction.effective_area_m2

                if abs(new_opening - old_opening) > 5.0:
                    direction = "closing" if new_opening < old_opening else "opening"
                    A_cm2 = self.inlet_restriction.effective_area_m2 * 1e4
                    print(f"  [DOOR] t={self.current_time:.1f}s: Door {direction} "
                          f"{old_opening:.0f}% -> {new_opening:.0f}% (A_eff={A_cm2:.1f} cm2)")
            else:
                self.door_opening_history[step] = self.door_opening_history[step - 1]
                self.A_eff_history[step] = self.A_eff_history[step - 1]

            # === ROOM PLENUM ===
            if self.room_plenum.cfg.enable:
                mdot_prev = self.mass_flow_rate[step - 1]
                Pa_ambient = self.parts[0].Pa
                dp_room = self.room_plenum.update(mdot_prev, Pa_ambient, self.dt)
                self.room_pressure_offset_Pa = dp_room
            elif self._room_pressure_func is not None:
                self.room_pressure_offset_Pa = self._room_pressure_func(self.current_time)

            # === INLET TEMPERATURE ===
            T_ambient_inlet = self.parts[0].T_ambient_K
            T_inlet = self._get_inlet_temp(self.current_time, T_ambient=T_ambient_inlet)

            # === SOLVE PHYSICS ===
            if self.use_coupled_solver:
                metrics = self.coupled_solver.solve_coupled_timestep(self, step, T_inlet, self.dt)
                # self.convergence_history.append(metrics)  # uncomment for debugging
            else:
                self._solve_comprehensive_heat_transfer(step, T_inlet)
                self._update_flow_and_pressure(step)

            # === BACKDRAFT DETECTION ===
            if step > 1:
                current_flow = self.mass_flow_rate[step]
                previous_flow = self.mass_flow_rate[step - 1]

                if previous_flow >= 0 and current_flow < 0:
                    self.backdraft_detected = True
                    self.backdraft_start_time = self.current_time
                    self.flow_reversal_count += 1
                    print(f"\n{'='*60}")
                    print(f"[BACKDRAFT DETECTED] Flow reversal at t={self.current_time:.1f}s")
                    print(f"  Mass flow: {previous_flow:.4f} -> {current_flow:.4f} kg/s")
                    print(f"  Flow direction: UPWARD -> DOWNWARD (backdraft)")
                    print(f"{'='*60}\n")

                elif previous_flow < 0 and current_flow >= 0:
                    self.backdraft_end_time = self.current_time
                    duration = self.backdraft_end_time - (self.backdraft_start_time or 0)
                    print(f"\n{'='*60}")
                    print(f"[DRAFT RESTORED] Normal flow resumed at t={self.current_time:.1f}s")
                    print(f"  Mass flow: {previous_flow:.4f} -> {current_flow:.4f} kg/s")
                    print(f"  Flow direction: DOWNWARD -> UPWARD (normal draft)")
                    print(f"  Backdraft duration: {duration:.1f}s")
                    print(f"{'='*60}\n")

            # === EARLY DIAGNOSTICS ===
            if step <= 30:
                T_liner = self.T_liner[step] if hasattr(self, 'T_liner') else None
                T_middle = self.T_middle[step] if hasattr(self, 'T_middle') else None
                T_outer = self.T_outer[step] if hasattr(self, 'T_outer') else None

                self.diagnostics.check_timestep(
                    step, self.current_time, 
                    self.T_gas[step], self.T_wall[step],
                    T_liner, T_middle, T_outer,
                    self.mass_flow_rate[step], self.dt
                )
            
            if step % max(1, self.time_steps // 20) == 0:
                progress = 100 * step / self.time_steps
                gas_temp_range = (self.T_gas[step].min() - 273.15, self.T_gas[step].max() - 273.15)
                wall_temp_range = (self.T_wall[step].min() - 273.15, self.T_wall[step].max() - 273.15)
                print(f"  {progress:5.1f}% | t={self.current_time:6.1f}s | Gas: {gas_temp_range[0]:5.1f}-{gas_temp_range[1]:5.1f}°C | Wall: {wall_temp_range[0]:5.1f}-{wall_temp_range[1]:5.1f}°C")

        print("Comprehensive simulation completed successfully")

        # TEMP: profiling -- print performance summary
        _elapsed = _time.perf_counter() - _t0
        print(_pc_summary(_elapsed))

        self.check_conservation()

    def _inlet_constant_no_heat(self,T_ambient=293.15):
        return T_ambient
    def _inlet_temp_special(self,t):
        """
        Approximate top-of-chimney temperature [°C]
        given time t [s] since ignition.
        """
        if self.inlet_temp_df is None:
            self.inlet_temp_df=pd.read_csv('./resources/actual_flue_temp.csv')
        
        return self.inlet_temp_df[self.inlet_temp_df['t_s']==t]['Flue Temp 1'].values[0] + 50 + 273.15
    def _default_inlet_temp(self, t, T_ambient=293.15, T_max=773.15, tau=300):
        """Exponential rise inlet temperature model.

        Args:
            t: Time [s]
            T_ambient: Ambient temperature [K]
            T_max: Asymptotic max inlet temperature [K]
            tau: Time constant [s] (63% of max at t=tau)

        Returns:
            Inlet gas temperature [K]
        """
        return T_ambient + (T_max - T_ambient) * (1 - np.exp(-t / tau))
    def _default_inlet_temp_fire(self, t_sec: float) -> float:
        """ISO 834 fire curve scaled to flue gas temperature.

        Returns:
            Temperature [K], capped at 800K
        """
        # ISO 834: T = 20 + 345 * log10(8t + 1), t in minutes; +1.5s avoids log(0)
        T_fire_C = 20 + 345 * np.log10(8 * ((t_sec + 1.5) / 60) + 1)
        T_flue_C = 0.5 * T_fire_C  # empirical scaling
        T_flue_K = T_flue_C + 273.15
        return min(T_flue_K, 800)

    def _solve_comprehensive_heat_transfer(self, step: int, T_inlet: float):
        """Solve heat transfer for all segments (sequential mode).

        Args:
            step: Current time step index
            T_inlet: Inlet gas temperature [K]
        """
        self.T_gas[step, 0] = T_inlet

        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = self.segment_lengths[seg]

            if seg > 0:
                self.T_gas[step, seg] = self.T_gas[step, seg-1]

            if step == 1:
                velocity = 1.0
            else:
                velocity = self.velocity[step-1, seg]
                
            # Cumulative distance from start of this part (for entry-region correction)
            x_from_part_start = (local_seg + 0.5) * segment_length

            segment_data = {
                'gas_temperature': self.T_gas[step, seg],
                'ambient_temperature': part.T_ambient_K,
                'pressure': part.Pa,
                'velocity': velocity,
                'diameter': part.inner_diameter_m,
                'length': segment_length,
                'x_from_part_start': x_from_part_start,
                'orientation': part.orientation,
                'part_height': part.height_m,
                'has_fittings': part.has_fittings,
                'humidity_ratio': part.humidity_ratio,
                'wall_type': part.type,
                'segment_id': seg,
                'mass_flow_rate': self.mass_flow_rate[step-1] if step > 0 else 0.01
            }

            if self.thermal_extra_length[seg] > 0:
                segment_data['thermal_length'] = segment_length + self.thermal_extra_length[seg]
            if self.thermal_extra_wall_Cj[seg] > 0:
                segment_data['extra_wall_Cj'] = self.thermal_extra_wall_Cj[seg]
            
            elbow_key = (part_idx, local_seg)
            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                segment_data['elbows'] = self.segment_elbows[elbow_key]
                segment_data['elbow_method'] = self.elbow_method

            if hasattr(part, 'corrugation') and part.corrugation is not None:
                segment_data['corrugation'] = part.corrugation
                if hasattr(part, 'corrugation_effects') and 'Mh' in part.corrugation_effects:
                    segment_data['Mh'] = part.corrugation_effects['Mh']

            if part.type == 'single':
                prev_T_wall = self.T_wall[step-1, seg]
                if prev_T_wall <= 0:  # -273.1 C issue
                    print(f"WARNING: Uninitialized wall temp {prev_T_wall:.1f}K at step={step}, seg={seg}, using ambient")
                    prev_T_wall = part.T_ambient_K
                    
                segment_data.update({
                    'wall_temperature': prev_T_wall,
                    'wall_properties': {
                        'density': part.density_w_kg_m3,
                        'specific_heat': part.cp_w_J_kg_K,
                        'thickness': part.thickness_w_m,
                        'emissivity': part.epsilon_w,
                        'thermal_conductivity': part.k_w_W_m_K
                    }
                })
            elif part.type == 'triple':
                prev_T_liner = self.T_liner[step-1, seg]
                prev_T_middle = self.T_middle[step-1, seg] 
                prev_T_outer = self.T_outer[step-1, seg]

                if prev_T_liner <= 0:  # -273.1 C issue
                    print(f"WARNING: Uninitialized liner temp {prev_T_liner:.1f}K at step={step}, seg={seg}, using ambient")
                    prev_T_liner = part.T_ambient_K
                    prev_T_middle = part.T_ambient_K
                    prev_T_outer = part.T_ambient_K

                segment_data.update({
                    'liner_temperature': prev_T_liner,
                    'middle_temperature': prev_T_middle,
                    'outer_temperature': prev_T_outer,
                    'wall_properties': {
                        'liner': {
                            'density': part.density_inner_kg_m3,
                            'specific_heat': part.cp_inner_J_kg_K,
                            'thickness': part.thickness_inner_m,
                            'emissivity': part.epsilon_inner,
                            'thermal_conductivity': part.k_inner_W_m_K
                        },
                        'middle': {
                            'density': part.density_middle_kg_m3,
                            'specific_heat': part.cp_middle_J_kg_K,
                            'thickness': part.thickness_middle_m,
                            'thermal_conductivity': part.k_middle_W_m_K
                        },
                        'outer': {
                            'density': part.density_outer_kg_m3,
                            'specific_heat': part.cp_outer_J_kg_K,
                            'thickness': part.thickness_outer_m,
                            'emissivity': part.epsilon_outer,
                            'thermal_conductivity': part.k_outer_W_m_K
                        },
                        'air_gap_inner': part.air_gap_inner_m,
                        'air_gap_outer': part.air_gap_outer_m
                    }
                })
            
            results = self.comprehensive_ht.calculate_segment_heat_transfer(segment_data, self.dt)

            if 'wall_temperature_new' not in results and 'liner_temperature_new' not in results:
                print(f"ERROR: No temperature results at step={step}, seg={seg}")
                print(f"Results keys: {list(results.keys())}")
                # Use ambient as fallback
                results['wall_temperature_new'] = part.T_ambient_K
                results['liner_temperature_new'] = part.T_ambient_K
                results['middle_temperature_new'] = part.T_ambient_K
                results['outer_temperature_new'] = part.T_ambient_K
            
            if part.type == 'single':
                T_wall_new = results['wall_temperature_new']
                if T_wall_new <= 0:
                    print(f"ERROR: Computed wall temp {T_wall_new:.1f}K is unphysical, using ambient")
                    T_wall_new = part.T_ambient_K
                    
                self.T_wall[step, seg] = T_wall_new
                self.T_outer[step, seg] = T_wall_new
                self.T_liner[step, seg] = T_wall_new
                self.T_middle[step, seg] = T_wall_new
            elif part.type == 'triple':
                T_liner_new = results['liner_temperature_new']
                T_middle_new = results['middle_temperature_new']
                T_outer_new = results['outer_temperature_new']
                
                if T_liner_new <= 0:
                    print(f"ERROR: Computed liner temp {T_liner_new:.1f}K is unphysical, using ambient")
                    T_liner_new = part.T_ambient_K
                    T_middle_new = part.T_ambient_K
                    T_outer_new = part.T_ambient_K
                    
                self.T_liner[step, seg] = T_liner_new
                self.T_middle[step, seg] = T_middle_new
                self.T_outer[step, seg] = T_outer_new
                self.T_wall[step, seg] = T_liner_new  # liner for compatibility

            q_loss = results['h_inner'] * (np.pi * part.inner_diameter_m * segment_length) * (self.T_gas[step, seg] - self.T_wall[step, seg])

            gas_props = self.comprehensive_ht.gas_properties(self.T_gas[step, seg], part.Pa)
            gas_volume = np.pi * (part.inner_diameter_m/2)**2 * segment_length
            gas_mass = gas_props.density * gas_volume
            
            # Only update gas T here in sequential mode; coupled solver does it itself
            if gas_mass > 0 and not self.use_coupled_solver and seg > 0:
                mass_flow = self.mass_flow_rate[step-1] if step > 0 else 0.0

                if mass_flow > 0:
                    T_upstream = self.T_gas[step, seg-1]

                    q_advection_net = mass_flow * gas_props.specific_heat * (T_upstream - self.T_gas[step, seg])
                    dT_gas = (q_advection_net - q_loss) * self.dt / (gas_mass * gas_props.specific_heat)
                    
                    # Gas cannot cool below ambient without external work
                    if dT_gas < 0:
                        max_cooling = self.T_gas[step, seg] - part.T_ambient_K
                        if abs(dT_gas) > max_cooling:
                            dT_gas = -max_cooling
                    
                    self.T_gas[step, seg] = self.T_gas[step, seg] + dT_gas
                else:
                    # No flow: gas relaxes toward wall temperature
                    tau_gas_wall = gas_mass * gas_props.specific_heat / (results['h_inner'] * np.pi * part.inner_diameter_m * segment_length)
                    self.T_gas[step, seg] = self.T_gas[step, seg] + (self.T_wall[step, seg] - self.T_gas[step, seg]) * self.dt / tau_gas_wall
            
            self.h_conv_inner[step, seg] = results['h_inner']
            self.h_conv_outer[step, seg] = results['h_outer']
            
            self.density[step, seg] = self.comprehensive_ht.gas_properties(
                self.T_gas[step, seg], part.Pa
            ).density
            
            # No refrigeration in system: clamp at ambient
            self.T_gas[step, seg] = max(self.T_gas[step, seg], part.T_ambient_K)
            self.T_wall[step, seg] = max(self.T_wall[step, seg], part.T_ambient_K)
            if part.type == 'triple':
                self.T_liner[step, seg] = max(self.T_liner[step, seg], part.T_ambient_K)
                self.T_middle[step, seg] = max(self.T_middle[step, seg], part.T_ambient_K)
                self.T_outer[step, seg] = max(self.T_outer[step, seg], part.T_ambient_K)

    def _update_flow_and_pressure(self, step: int):
        """Update mass flow, velocity, pressure from buoyancy and losses (sequential mode)."""
        total_buoyancy = 0.0  # Pa
        total_pressure_loss = 0.0  # Pa

        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments

            rho_gas = self.density[step, seg]
            rho_ambient = self.comprehensive_ht.gas_properties(part.T_ambient_K, part.Pa).density

            dp_buoyancy_seg = self.comprehensive_ht.buoyancy_pressure(
                segment_length, rho_ambient, rho_gas
            )
            total_buoyancy += dp_buoyancy_seg

            if step > 1:
                velocity = self.velocity[step-1, seg]
            else:
                velocity = 1.0

            gas_props = self.comprehensive_ht.gas_properties(self.T_gas[step, seg], part.Pa)
            gas_props.velocity = velocity
            
            elbow_key = (part_idx, local_seg)
            seg_elbows = None
            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                seg_elbows = self.segment_elbows[elbow_key]

            dp_loss_seg = self.comprehensive_ht.pressure_drop(
                gas_props, part.inner_diameter_m, segment_length,
                velocity, part.roughness_m, elbows=seg_elbows
            )
            total_pressure_loss += dp_loss_seg
        
        # === MOMENTUM BALANCE ===
        net_pressure = total_buoyancy - total_pressure_loss - self.room_pressure_offset_Pa

        avg_density = np.mean(self.density[step, :])
        flow_coefficient = 1.0
        avg_area = np.mean([part.inner_area_m2 for part in self.parts])

        velocity_avg = np.sqrt(2 * abs(net_pressure) / (flow_coefficient * avg_density + 1e-10))

        if net_pressure >= 0:
            self.mass_flow_rate[step] = avg_density * velocity_avg * avg_area
        else:
            # Negative net pressure drives downward flow
            self.mass_flow_rate[step] = -avg_density * velocity_avg * avg_area

        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            
            self.velocity[step, seg] = self.mass_flow_rate[step] / (
                self.density[step, seg] * part.inner_area_m2
            )
            
            if seg == 0:
                self.pressure[step, seg] = part.Pa
            else:
                segment_length = part.height_m / part.num_segments
                rho_gas = self.density[step, seg]
                rho_ambient = self.comprehensive_ht.gas_properties(part.T_ambient_K, part.Pa).density
                dp_buoyancy = 9.81 * segment_length * (rho_ambient - rho_gas)

                velocity = self.velocity[step, seg]
                dp_friction = 0.0
                if abs(velocity) > 0.01:
                    gas_props = self.comprehensive_ht.gas_properties(self.T_gas[step, seg], part.Pa)
                    gas_props.velocity = velocity
                    elbow_key = (part_idx, local_seg)
                    seg_elbows = None
                    if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                        seg_elbows = self.segment_elbows[elbow_key]
                    dp_friction = self.comprehensive_ht.pressure_drop(
                        gas_props, part.inner_diameter_m, segment_length,
                        velocity, part.roughness_m, elbows=seg_elbows
                    )

                self.pressure[step, seg] = self.pressure[step, seg-1] + dp_buoyancy - dp_friction

    def get_summary_data(self) -> Dict[str, Any]:
        """Get summary data from the simulation"""
        return {
            'total_segments': self.total_segments,
            'total_time': self.total_time,
            'time_steps': self.time_steps,
            'final_gas_temp_range_C': (
                float(self.T_gas[-1].min() - 273.15), 
                float(self.T_gas[-1].max() - 273.15)
            ),
            'final_wall_temp_range_C': (
                float(self.T_wall[-1].min() - 273.15), 
                float(self.T_wall[-1].max() - 273.15)
            ),
            'final_mass_flow_rate_kg_s': float(self.mass_flow_rate[-1]),
            'parts_config': [
                {
                    'type': part.type,
                    'height_m': part.height_m,
                    'diameter_m': part.inner_diameter_m,
                    'segments': part.num_segments
                }
                for part in self.parts
            ]
        }

    def check_conservation(self):
        """Check mass and energy conservation throughout the simulation"""
        print("\n" + "="*80)
        print("CONSERVATION DIAGNOSTICS")
        print("="*80)
        
        # Energy conservation check
        self._check_energy_conservation()
        
        self._check_mass_conservation()

    def _check_energy_conservation(self):
        """Check energy conservation by comparing input, stored, and lost energy."""
        print("\nEnergy Conservation Check:")

        energy_input = 0.0
        for step in range(1, self.time_steps + 1):
            T_inlet = self.T_gas[step, 0]
            mdot = self.mass_flow_rate[step]
            cp = self.physics.air_specific_heat(T_inlet)
            T_ref = self.parts[0].T_ambient_K
            energy_input += mdot * cp * (T_inlet - T_ref) * self.dt
        
        energy_stored_gas = 0.0
        energy_stored_wall = 0.0
        
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            
            T_gas_final = self.T_gas[-1, seg]
            T_gas_initial = self.T_gas[0, seg]
            gas_volume = part.inner_area_m2 * segment_length
            rho_final = self.density[-1, seg]
            cp_gas = self.physics.air_specific_heat(T_gas_final)
            energy_stored_gas += rho_final * gas_volume * cp_gas * (T_gas_final - T_gas_initial)
            
            if part.type == 'single':
                T_wall_final = self.T_wall[-1, seg]
                T_wall_initial = self.T_wall[0, seg]
                wall_volume = np.pi * segment_length * ((part.outer_diameter_m)**2 - (part.inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_w_kg_m3 * wall_volume * part.cp_w_J_kg_K * (T_wall_final - T_wall_initial)
            else:
                T_liner_final = self.T_liner[-1, seg]
                T_liner_initial = self.T_liner[0, seg]
                V_liner = np.pi * segment_length * ((part.liner_outer_diameter_m)**2 - (part.inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_inner_kg_m3 * V_liner * part.cp_inner_J_kg_K * (T_liner_final - T_liner_initial)

                T_middle_final = self.T_middle[-1, seg]
                T_middle_initial = self.T_middle[0, seg]
                V_middle = np.pi * segment_length * ((part.middle_outer_diameter_m)**2 - (part.middle_inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_middle_kg_m3 * V_middle * part.cp_middle_J_kg_K * (T_middle_final - T_middle_initial)

                T_outer_final = self.T_outer[-1, seg]
                T_outer_initial = self.T_outer[0, seg]
                V_outer = np.pi * segment_length * ((part.outer_diameter_m)**2 - (part.outer_inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_outer_kg_m3 * V_outer * part.cp_outer_J_kg_K * (T_outer_final - T_outer_initial)
        
        energy_lost = energy_input - energy_stored_gas - energy_stored_wall

        if energy_input > 0:
            pct_stored_gas = 100 * energy_stored_gas / energy_input
            pct_stored_wall = 100 * energy_stored_wall / energy_input
            pct_lost = 100 * energy_lost / energy_input
            
            print(f"  Total energy input: {energy_input/1e6:.2f} MJ")
            print(f"  Energy stored in gas: {energy_stored_gas/1e6:.2f} MJ ({pct_stored_gas:.1f}%)")
            print(f"  Energy stored in walls: {energy_stored_wall/1e6:.2f} MJ ({pct_stored_wall:.1f}%)")
            print(f"  Energy lost to ambient: {energy_lost/1e6:.2f} MJ ({pct_lost:.1f}%)")
            print(f"  Energy balance closure: {100 - abs(pct_stored_gas + pct_stored_wall + pct_lost - 100):.1f}%")

    def _check_mass_conservation(self):
        """Check water vapor mass conservation with condensation."""
        print("\nMass Conservation Check (Water Vapor):")
        
        if not hasattr(self.comprehensive_ht, 'condensation_states') or not self.comprehensive_ht.condensation_states:
            print("  No condensation tracking available")
            return
        
        total_condensed = 0.0
        for seg_id, state in self.comprehensive_ht.condensation_states.items():
            total_condensed += state.film_mass
        
        print(f"  Total water condensed: {total_condensed*1000:.3f} g")
        
        initial_water = 0.0
        final_water = 0.0
        
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            gas_volume = part.inner_area_m2 * segment_length
            
            rho_initial = self.density[0, seg]
            humidity_initial = part.humidity_ratio
            dry_air_mass_initial = gas_volume * rho_initial / (1 + humidity_initial)
            initial_water += dry_air_mass_initial * humidity_initial

            rho_final = self.density[-1, seg]
            humidity_final = part.humidity_ratio  # This is simplified
            dry_air_mass_final = gas_volume * rho_final / (1 + humidity_final)
            final_water += dry_air_mass_final * humidity_final
        
        water_removed = initial_water - final_water
        
        print(f"  Water vapor removed from gas: {water_removed*1000:.3f} g")
        print(f"  Mass balance error: {abs(water_removed - total_condensed)*1000:.3f} g")
        
        if initial_water > 0:
            error_pct = 100 * abs(water_removed - total_condensed) / initial_water
            print(f"  Mass balance closure: {100 - error_pct:.1f}%")
    def _process_elbows(self):
        """Process elbow definitions and map them to segments."""
        from .flow.elbows import ElbowSpec

        self.segment_elbows = {}  # key: (part_idx, local_seg_idx), value: List[ElbowSpec]

        n_parts = len(self.parts)
        max_fittings = n_parts  # inlet fitting + all junctions between parts

        if len(self.elbows) > max_fittings:
            raise ValueError(
                f"Geometry error: {len(self.elbows)} fittings specified but only "
                f"{max_fittings} locations exist ({n_parts} parts allow 1 inlet + {n_parts-1} junctions). "
                f"Each fitting must be at inlet (-1,0) or between two adjacent parts."
            )
        
        used_junctions = set()
        
        for i, elbow_dict in enumerate(self.elbows):
            between = elbow_dict.get('between_segments', elbow_dict.get('between_parts', []))
            if len(between) != 2:
                raise ValueError(
                    f"Elbow {i} missing or invalid 'between_segments'/'between_parts' specification. "
                    f"Expected [from_part, to_part], got {between}"
                )
                
            part_from, part_to = between[0], between[1]

            is_inlet_fitting = (part_from == -1 and part_to == 0)

            if not is_inlet_fitting and part_from >= n_parts:
                raise ValueError(
                    f"Elbow {i}: Invalid 'from' part index {part_from}. "
                    f"Only {n_parts} parts exist (indices 0-{n_parts-1})"
                )

            if part_to >= n_parts:
                raise ValueError(
                    f"Elbow {i}: Invalid 'to' part index {part_to}. "
                    f"Only {n_parts} parts exist (indices 0-{n_parts-1})"
                )

            if not is_inlet_fitting and abs(part_to - part_from) != 1:
                raise ValueError(
                    f"Elbow {i}: Parts {part_from} and {part_to} are not adjacent. "
                    f"Elbows can only connect adjacent parts."
                )

            junction = (min(part_from, part_to), max(part_from, part_to))
            if junction in used_junctions:
                raise ValueError(
                    f"Elbow {i}: Junction between parts {part_from} and {part_to} "
                    f"already has an elbow defined. Each junction can have only one elbow."
                )
            used_junctions.add(junction)

            if not is_inlet_fitting and part_from >= 0 and part_from < n_parts and part_to < n_parts:
                orient_from = self.parts[part_from].orientation
                orient_to = self.parts[part_to].orientation

                if orient_from == orient_to == "horizontal":
                    print(f"Warning: Elbow {i} connects two horizontal sections. "
                          f"Ensure the angle_deg reflects the actual geometry.")

                diam_from = self.parts[part_from].inner_diameter_m
                diam_to = self.parts[part_to].inner_diameter_m

                if abs(diam_from - diam_to) / min(diam_from, diam_to) > 0.1:
                    print(f"Warning: Elbow {i} connects parts with >10% diameter difference "
                          f"({diam_from*1000:.1f}mm to {diam_to*1000:.1f}mm). "
                          f"Consider adding a reducer/expander.")

            if is_inlet_fitting:
                diameter = self.parts[0].inner_diameter_m
            else:
                diameter = self.parts[part_from].inner_diameter_m

            # Accept both 'elbow_type' and 'type' keys
            elbow_type = elbow_dict.get('elbow_type', elbow_dict.get('type', 'long_radius'))
            if elbow_type == 'tee':
                angle_deg = elbow_dict.get('angle_deg', 90)
                R_over_D = 1.0  # fixed for tees
            else:
                angle_deg = elbow_dict.get('angle_deg', 90)
                R_over_D = elbow_dict.get('R_over_D', 1.5)

            elbow_spec = ElbowSpec(
                angle_deg=angle_deg,
                diameter_m=diameter,
                R_over_D=R_over_D,
                elbow_type=elbow_type,
                smoothness=elbow_dict.get('smoothness', 'commercial_steel'),
                vane_factor=elbow_dict.get('vane_factor', 1.0),
                K_override=elbow_dict.get('K_override', None),
                Le_override_m=elbow_dict.get('Le_override_m', None)
            )
            
            # Inlet fitting: first seg of part 0; otherwise: last seg of upstream part
            if is_inlet_fitting:
                key = (0, 0)  # First segment of first part
            else:
                last_seg_idx = self.parts[part_from].num_segments - 1
                key = (part_from, last_seg_idx)

            if key not in self.segment_elbows:
                self.segment_elbows[key] = []
            self.segment_elbows[key].append(elbow_spec)

    # ------------------------------------------------------------------
    # Auto-computed elevation / path-position arrays
    # ------------------------------------------------------------------

    @staticmethod
    def _fitting_vertical_contribution(fitting_dict):
        """Return the vertical rise (metres) contributed by a fitting.

        Tee:   centre of horizontal branch to top  = vertical_length_ft / 2
        Elbow: bottom to centreline of horiz. exit = height_ft - diameter_ft/2
        """
        ftype = fitting_dict.get('type', fitting_dict.get('elbow_type', ''))
        if ftype == 'tee':
            vl_ft = fitting_dict.get('vertical_length_ft', 0.0)
            return vl_ft * 0.3048 / 2.0
        elif ftype == 'elbow':
            h_ft = fitting_dict.get('height_ft', 0.0)
            d_ft = fitting_dict.get('diameter_ft', 0.0)
            return max((h_ft - d_ft / 2.0) * 0.3048, 0.0)
        return 0.0

    @staticmethod
    def _fitting_path_contribution(fitting_dict):
        """Return the centreline path length (metres) contributed by a fitting.

        Tee:   (horizontal_length + D/2) + vertical_length/2
               horizontal body + transition to vertical centreline + rise to top
        Elbow: (height - D/2) + (width - D/2)
               vertical centreline offset + horizontal centreline offset
        """
        ftype = fitting_dict.get('type', fitting_dict.get('elbow_type', ''))
        d_ft = fitting_dict.get('diameter_ft', 0.0)
        if ftype == 'tee':
            vl_ft = fitting_dict.get('vertical_length_ft', 0.0)
            hl_ft = fitting_dict.get('horizontal_length_ft', 0.0)
            return ((hl_ft + d_ft / 2.0) + vl_ft / 2.0) * 0.3048
        elif ftype == 'elbow':
            h_ft = fitting_dict.get('height_ft', 0.0)
            w_ft = fitting_dict.get('width_ft', 0.0)
            return max(h_ft - d_ft / 2.0, 0.0) * 0.3048 + max(w_ft - d_ft / 2.0, 0.0) * 0.3048
        return 0.0

    def _compute_segment_positions(self):
        """Walk the assembly and compute per-segment elevation and path arrays.

        Produces:
            self.segment_elevations     – shape (total_segments,)  [m above ref]
            self.segment_path_positions – shape (total_segments,)  [m along flow path]
            self.segment_vertical_heights – shape (total_segments,) [m, vertical rise per segment]
            self.part_base_elevations   – shape (n_parts,)         [m, bottom of each part]
        """
        n_seg = self.total_segments
        n_parts = len(self.parts)

        self.segment_elevations = np.zeros(n_seg)
        self.segment_path_positions = np.zeros(n_seg)
        self.segment_vertical_heights = np.zeros(n_seg)
        self.part_base_elevations = np.zeros(n_parts)

        elev = 0.0   # running elevation accumulator (m)
        path = 0.0   # running path-length accumulator (m)

        # --- helper: find raw fitting dict by (from, to) ---
        def _find_fitting(part_from, part_to):
            for fd in self.elbows:
                between = fd.get('between_segments', fd.get('between_parts', []))
                if len(between) == 2 and between[0] == part_from and between[1] == part_to:
                    return fd
            return None

        # 1. Inlet fitting at (-1, 0)
        inlet = _find_fitting(-1, 0)
        if inlet is not None:
            elev += self._fitting_vertical_contribution(inlet)
            path += self._fitting_path_contribution(inlet)

        seg_idx = 0
        for pi, part in enumerate(self.parts):
            self.part_base_elevations[pi] = elev
            seg_len = part.height_m / part.num_segments
            is_vert = part.orientation.lower() == 'vertical'

            for ls in range(part.num_segments):
                # segment centre elevation
                if is_vert:
                    self.segment_elevations[seg_idx] = elev + (ls + 0.5) * seg_len
                    self.segment_vertical_heights[seg_idx] = seg_len
                else:
                    self.segment_elevations[seg_idx] = elev  # no vertical rise
                    self.segment_vertical_heights[seg_idx] = 0.0

                self.segment_path_positions[seg_idx] = path + (ls + 0.5) * seg_len
                seg_idx += 1

            # advance accumulators past the part
            if is_vert:
                elev += part.height_m
            path += part.height_m

            # 2. Junction fitting between part pi and pi+1
            if pi < n_parts - 1:
                jf = _find_fitting(pi, pi + 1)
                if jf is not None:
                    elev += self._fitting_vertical_contribution(jf)
                    path += self._fitting_path_contribution(jf)

    # ------------------------------------------------------------------
    # Fittings thermal model
    # ------------------------------------------------------------------

    def _init_fittings_thermal(self, config):
        """Initialise the fittings thermal model.

        Computes two per-segment arrays:
            self.thermal_extra_length[total_segments]  – extra HT length (m)
            self.thermal_extra_wall_Cj[total_segments] – extra wall heat capacity (J/K)

        These are zero when model='none' (default / legacy behaviour).
        """
        if config is None:
            config = {}
        if isinstance(config, str):
            config = {'model': config}
        self._ftm_model = config.get('model', 'none')
        self._ftm_allocation = config.get('allocation', 'weighted')
        self._ftm_weights = tuple(config.get('weights', (0.8, 0.2)))

        n = self.total_segments
        self.thermal_extra_length = np.zeros(n)
        self.thermal_extra_wall_Cj = np.zeros(n)

        if self._ftm_model == 'none':
            return

        # --- Build per-fitting thermal data ---
        alloc_table = []  # for debug printing
        for fi, fd in enumerate(self.elbows):
            between = fd.get('between_segments', fd.get('between_parts', []))
            if len(between) != 2:
                continue
            part_from, part_to = between

            L_eq = self._fitting_path_contribution(fd)
            if L_eq <= 0:
                continue

            # Determine receiving segments (upstream last seg / downstream first seg)
            is_inlet = (part_from == -1)
            if is_inlet:
                seg_up = None
                seg_down = self._part_start_seg(0)
            else:
                seg_up = self._part_end_seg(part_from)
                seg_down = self._part_start_seg(part_to) if part_to < len(self.parts) else None

            # Allocation weights
            w_up, w_down = self._alloc_weights(seg_up, seg_down)

            # Distribute length
            if seg_up is not None:
                self.thermal_extra_length[seg_up] += L_eq * w_up
            if seg_down is not None:
                self.thermal_extra_length[seg_down] += L_eq * w_down

            # Wall thermal mass (absorb_length_and_mass only)
            if self._ftm_model == 'absorb_length_and_mass':
                Cj = self._fitting_wall_heat_capacity(fd, L_eq)
                if seg_up is not None:
                    self.thermal_extra_wall_Cj[seg_up] += Cj * w_up
                if seg_down is not None:
                    self.thermal_extra_wall_Cj[seg_down] += Cj * w_down

            alloc_table.append({
                'fitting': fi,
                'type': fd.get('type', fd.get('elbow_type', '?')),
                'between': f"{part_from}->{part_to}",
                'L_eq_in': L_eq / 0.0254,
                'seg_up': seg_up,
                'seg_down': seg_down,
                'w_up': w_up,
                'w_down': w_down,
            })

        # --- Debug table ---
        total_L = np.sum(self.thermal_extra_length)
        print(f"\n[FITTINGS THERMAL] model={self._ftm_model}, "
              f"allocation={self._ftm_allocation}, weights={self._ftm_weights}")
        print(f"{'Fit':>3} {'Type':>6} {'Junc':>6} {'L_eq':>7} "
              f"{'SegUp':>5} {'SegDn':>5} {'wUp':>5} {'wDn':>5}")
        for r in alloc_table:
            print(f"{r['fitting']:3d} {r['type']:>6} {r['between']:>6} "
                  f"{r['L_eq_in']:6.1f}in "
                  f"{str(r['seg_up']):>5} {str(r['seg_down']):>5} "
                  f"{r['w_up']:5.2f} {r['w_down']:5.2f}")
        print(f"Total absorbed thermal length: {total_L:.4f} m "
              f"({total_L/0.0254:.1f} in)")
        if self._ftm_model == 'absorb_length_and_mass':
            total_Cj = np.sum(self.thermal_extra_wall_Cj)
            print(f"Total absorbed wall heat capacity: {total_Cj:.1f} J/K")
        # Per-segment summary
        for seg in range(self.total_segments):
            if self.thermal_extra_length[seg] > 0:
                pct = self.thermal_extra_length[seg] / self.segment_lengths[seg] * 100
                msg = f"  seg {seg}: +{self.thermal_extra_length[seg]/0.0254:.1f}in "
                msg += f"({pct:.0f}% of pipe seg)"
                if self.thermal_extra_wall_Cj[seg] > 0:
                    msg += f", +{self.thermal_extra_wall_Cj[seg]:.1f} J/K wall mass"
                print(msg)
        print()

    def _part_start_seg(self, part_idx):
        """Return global segment index of the first segment of a part."""
        seg = 0
        for i in range(part_idx):
            seg += self.parts[i].num_segments
        return seg

    def _part_end_seg(self, part_idx):
        """Return global segment index of the last segment of a part."""
        return self._part_start_seg(part_idx) + self.parts[part_idx].num_segments - 1

    def _alloc_weights(self, seg_up, seg_down):
        """Return (w_up, w_down) allocation weights."""
        if seg_up is None:
            return (0.0, 1.0)
        if seg_down is None:
            return (1.0, 0.0)
        mode = self._ftm_allocation
        if mode == 'upstream':
            return (1.0, 0.0)
        elif mode == 'downstream':
            return (0.0, 1.0)
        elif mode == 'split':
            return (0.5, 0.5)
        else:  # weighted
            return self._ftm_weights

    def _fitting_wall_heat_capacity(self, fitting_dict, L_eq):
        """Estimate the steel thermal mass (J/K) of a fitting.

        Uses the pipe inner/outer diameters of the adjacent part to estimate
        the steel shell volume, then C = rho * cp * V.
        """
        between = fitting_dict.get('between_segments', fitting_dict.get('between_parts', []))
        part_from, part_to = between

        # Pick a representative part for wall properties
        if part_from >= 0 and part_from < len(self.parts):
            part = self.parts[part_from]
        else:
            part = self.parts[part_to]

        # For single-wall fittings, use the pipe wall thickness
        # For triple-wall, use only the liner (innermost steel)
        D_inner = part.inner_diameter_m
        if part.type == 'single':
            thickness = part.thickness_w_m
            rho = part.density_w_kg_m3
            cp = part.cp_w_J_kg_K
        else:
            thickness = part.thickness_inner_m
            rho = part.density_inner_kg_m3
            cp = part.cp_inner_J_kg_K

        r_i = D_inner / 2
        r_o = r_i + thickness
        V_steel = np.pi * (r_o**2 - r_i**2) * L_eq
        return rho * cp * V_steel

    def _print_geometry_summary(self):
        """Print a sanity check summary of the chimney geometry."""
        print("\n" + "="*70)
        print("CHIMNEY GEOMETRY SUMMARY (Sanity Check)")
        print("="*70)
        
        components = []
        total_height = 0.0
        total_length = 0.0
        
        def format_length(meters):
            feet = meters / 0.3048
            inches = meters / 0.0254
            if feet > 0.5:
                return f"{meters:.2f}m ({feet:.1f}ft)"
            else:
                return f"{meters*1000:.0f}mm ({inches:.1f}in)"
        
        def format_diameter(meters):
            mm = meters * 1000
            inches = meters / 0.0254
            return f"{mm:.1f}mm ({inches:.1f}in)"
        
        if hasattr(self, 'segment_elbows') and (0, 0) in self.segment_elbows:
            for elbow in self.segment_elbows[(0, 0)]:
                elbow_desc = f"INLET: {elbow.angle_deg:.0f}° {elbow.elbow_type.replace('_', '-')}"
                elbow_desc += f" (R/D={elbow.R_over_D:.1f})"
                components.append(elbow_desc)

        for i, part in enumerate(self.parts):
            length_str = format_length(part.height_m)
            diam_str = format_diameter(part.inner_diameter_m)

            elev_str = ""
            if hasattr(self, 'part_base_elevations'):
                elev_m = self.part_base_elevations[i]
                elev_ft = elev_m / 0.3048
                elev_str = f", elev={elev_m:.2f}m ({elev_ft:.1f}ft)"

            part_desc = f"Part {i}: {part.type.upper()} {part.orientation}"
            part_desc += f", L={length_str}, D={diam_str}{elev_str}"

            if part.type == 'triple':
                part_desc += " (triple-wall)"

            components.append(part_desc)

            if part.orientation == 'vertical':
                total_height += part.height_m
            total_length += part.height_m

            elbow_key = (i, part.num_segments - 1)
            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                for elbow in self.segment_elbows[elbow_key]:
                    elbow_desc = f"       +-- FITTING: {elbow.angle_deg:.0f}°"
                    elbow_desc += f" {elbow.elbow_type.replace('_', '-')}"
                    elbow_desc += f" (R/D={elbow.R_over_D:.1f})"
                    if elbow.vane_factor < 1.0:
                        elbow_desc += f" with vanes (factor={elbow.vane_factor:.1f})"
                    components.append(elbow_desc)
        
        print("\nFlow path from inlet to outlet:")
        print("-" * 40)
        for comp in components:
            print(comp)
        
        total_fittings = sum(len(v) for v in self.segment_elbows.values()) if hasattr(self, 'segment_elbows') else 0
        print("\n" + "-" * 40)
        print("Summary:")
        print(f"  Total parts: {len(self.parts)}")
        print(f"  Total fittings (elbows/tees): {total_fittings}")
        print(f"  Total path length: {format_length(total_length)}")
        print(f"  Total vertical rise: {format_length(total_height)}")
        print(f"  Computational segments: {self.total_segments}")
        print(f"  Elbow loss method: {self.elbow_method}")
        
        warnings = []

        diameters = [p.inner_diameter_m for p in self.parts]
        if len(set(diameters)) > 1:
            warnings.append("Parts have different diameters - check for proper transitions")
        
        avg_segments = self.total_segments / len(self.parts)
        if avg_segments < 2:
            warnings.append("Low segment count - consider increasing for better accuracy")
        elif avg_segments > 20:
            warnings.append("High segment count - may increase computation time")
        
        if len(self.elbows) > len(self.parts):
            warnings.append("More elbows than parts - verify configuration")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  [!] {warning}")
        
        print("="*70)

    def _initialize_arrays(self):
        """Pre-allocate result arrays: shape (time_steps+1, total_segments)."""
        # Temperature arrays [K]
        self.T_gas = np.zeros((self.time_steps + 1, self.total_segments))
        self.T_wall = np.zeros((self.time_steps + 1, self.total_segments))  # mirrors T_liner for triple
        self.T_outer = np.zeros((self.time_steps + 1, self.total_segments))

        # Triple wall layers
        self.T_liner = np.zeros((self.time_steps + 1, self.total_segments))
        self.T_middle = np.zeros((self.time_steps + 1, self.total_segments))

        # Flow properties
        self.mass_flow_rate = np.zeros(self.time_steps + 1)  # kg/s
        self.velocity = np.zeros((self.time_steps + 1, self.total_segments))  # m/s
        self.pressure = np.zeros((self.time_steps + 1, self.total_segments))  # Pa
        self.density = np.zeros((self.time_steps + 1, self.total_segments))  # kg/m³

        self.h_conv_inner = np.zeros((self.time_steps + 1, self.total_segments))  # W/(m²·K)
        self.h_conv_outer = np.zeros((self.time_steps + 1, self.total_segments))  # W/(m²·K)

        self.door_opening_history = np.zeros(self.time_steps + 1)
        initial_opening = self.inlet_restriction.door_opening_percent if self.inlet_restriction.enable else 100.0
        self.door_opening_history[0] = initial_opening

        self.A_eff_history = np.zeros(self.time_steps + 1)
        if self.inlet_restriction.enable:
            self.A_eff_history[0] = self.inlet_restriction.effective_area_m2

        self.time_array = np.linspace(0, self.total_time, self.time_steps + 1)

        self._set_initial_conditions()

    def _set_initial_conditions(self):
        """Set initial temperature and flow conditions."""
        print(f"DEBUG: Initializing conditions for {len(self.parts)} parts")

        for i, part in enumerate(self.parts):
            start_seg = sum(p.num_segments for p in self.parts[:i])
            end_seg = start_seg + part.num_segments

            for seg in range(start_seg, end_seg):
                if seg == 0:
                    if self.inlet_temp_func is not None:
                        initial_gas_temp = float(self.inlet_temp_func(0.0))
                    elif self.inlet_temp_interp is not None:
                        initial_gas_temp = float(self.inlet_temp_interp(0.0))
                    else:
                        initial_gas_temp = part.T_ambient_K + 10.0  # small rise avoids cooling pulse
                else:
                    initial_gas_temp = part.T_ambient_K

                # Initialize ALL timesteps to prevent -273.1 C issue
                self.T_gas[:, seg] = initial_gas_temp
            
            initial_wall_temp = part.T_ambient_K
            
            print(f"DEBUG: Part {i+1} ({part.type}), segments {start_seg}-{end_seg-1}, T_ambient={part.T_ambient_K-273.15:.1f}°C")
            
            if part.type == 'single':
                self.T_wall[:, start_seg:end_seg] = initial_wall_temp
                self.T_outer[:, start_seg:end_seg] = initial_wall_temp
                # Prevents -273.1 C errors in diagnostics for single-wall segments
                self.T_liner[:, start_seg:end_seg] = initial_wall_temp
                self.T_middle[:, start_seg:end_seg] = initial_wall_temp
                print(f"DEBUG: Single wall initialized - T_wall[0,{start_seg}] = {self.T_wall[0,start_seg]-273.15:.1f}°C")
            elif part.type == 'triple':
                self.T_liner[:, start_seg:end_seg] = part.T_ambient_K
                self.T_middle[:, start_seg:end_seg] = part.T_ambient_K
                self.T_outer[:, start_seg:end_seg] = part.T_ambient_K
                self.T_wall[:, start_seg:end_seg] = part.T_ambient_K
                print(f"DEBUG: Triple wall initialized - T_liner[0,{start_seg}] = {self.T_liner[0,start_seg]-273.15:.1f}°C")
            
            for seg in range(start_seg, end_seg):
                seg_gas_temp = self.T_gas[0, seg]
                initial_density = self.physics.air_density(seg_gas_temp, part.Pa)
                self.density[:, seg] = initial_density
                self.pressure[:, seg] = part.Pa
                self.velocity[:, seg] = 0.1  # m/s

        self.mass_flow_rate[:] = 0.01  # kg/s
        
    def get_segment_info(self, segment_idx: int) -> tuple:
        """Get part and local segment index for a global segment index"""
        current_seg = 0
        for part_idx, part in enumerate(self.parts):
            if current_seg + part.num_segments > segment_idx:
                local_seg = segment_idx - current_seg
                return part_idx, local_seg, part
            current_seg += part.num_segments
        raise IndexError(f"Segment index {segment_idx} out of range")

    def _parse_wind_cap(self, tb: dict) -> WindCapConfig:
        """Parse top_boundary dict into WindCapConfig"""
        return WindCapConfig(
            enable=bool(tb.get("enable_wind_cap_effect", False)),
            wind_speed_m_s=tb.get("wind_speed_m_s", 0.0),
            wind_direction_deg=tb.get("wind_direction_deg", 0.0),
            air_temp_K=tb.get("air_temp_K", 294.15),
            cap_type=tb.get("cap_type", "open"),
            roof_exposure=tb.get("roof_exposure", "open"),
            orientation=tb.get("orientation", "vertical"),
            gust_factor=tb.get("gust_factor", 1.0),
            Cp_override=tb.get("Cp_override", None),
            update_mode=tb.get("wind_cap_update", "dynamic"),
        )

    def _initialize_inlet_restriction(self, inlet_restriction: Optional[Dict[str, Any]]):
        """Convert door geometry to effective flow area and K-loss."""
        import math

        if inlet_restriction is None:
            self.inlet_restriction = InletRestrictionConfig()
            return

        if isinstance(inlet_restriction, dict):
            self.inlet_restriction = InletRestrictionConfig(**inlet_restriction)
        else:
            self.inlet_restriction = inlet_restriction

        if not self.inlet_restriction.enable:
            return

        if not (0.0 <= self.inlet_restriction.door_opening_percent <= 100.0):
            raise ValueError(
                f"door_opening_percent must be between 0 and 100, "
                f"got {self.inlet_restriction.door_opening_percent}"
            )

        if self.inlet_restriction.door_width_ft <= 0:
            raise ValueError(
                f"door_width_ft must be positive, got {self.inlet_restriction.door_width_ft}"
            )
        if self.inlet_restriction.door_height_ft <= 0:
            raise ValueError(
                f"door_height_ft must be positive, got {self.inlet_restriction.door_height_ft}"
            )

        FT_TO_M = 0.3048
        door_width_m = self.inlet_restriction.door_width_ft * FT_TO_M
        door_height_m = self.inlet_restriction.door_height_ft * FT_TO_M
        self.inlet_restriction.door_area_m2 = door_width_m * door_height_m

        if self.inlet_restriction.Cd <= 0 or self.inlet_restriction.Cd > 1.0:
            raise ValueError(
                f"Cd must be in range (0, 1.0], got {self.inlet_restriction.Cd}"
            )

        if self.inlet_restriction.A_leak_m2 <= 0:
            raise ValueError(
                f"A_leak_m2 must be positive, got {self.inlet_restriction.A_leak_m2}"
            )

        if self.inlet_restriction.K_loss_override is not None:
            if self.inlet_restriction.K_loss_override < 0:
                raise ValueError("K_loss_override cannot be negative")
            if self.inlet_restriction.K_loss_override > 100:
                print("WARNING: Very large K_loss (>100) may cause numerical issues")

        door_type = self.inlet_restriction.door_type.lower()
        if door_type not in ('swivel', 'sliding'):
            raise ValueError(
                f"door_type must be 'swivel' or 'sliding', got '{self.inlet_restriction.door_type}'"
            )

        opening_pct = self.inlet_restriction.door_opening_percent
        full_door_area = self.inlet_restriction.door_area_m2

        if door_type == 'swivel':
            angle_rad = (opening_pct / 100.0) * (math.pi / 2.0)
            A_eff_raw = full_door_area * math.sin(angle_rad)
        else:  # sliding
            A_eff_raw = full_door_area * (opening_pct / 100.0)

        self.inlet_restriction.A_eff_raw_m2 = A_eff_raw

        A_leak = self.inlet_restriction.A_leak_m2
        area_mode = self.inlet_restriction.inlet_area_mode.lower()
        if area_mode == "quadrature":
            A_eff_used = math.sqrt(A_eff_raw**2 + A_leak**2)
        else:
            A_eff_used = A_eff_raw + A_leak

        self.inlet_restriction.effective_area_uncapped_m2 = A_eff_used

        cap_mode = self.inlet_restriction.inlet_area_cap_mode.lower()
        if cap_mode == "pipe":
            A_pipe = self.parts[0].inner_area_m2
            if A_eff_used > A_pipe:
                A_eff_used = A_pipe
                print(f"  [Note: A_eff capped at pipe area = {A_pipe*1e4:.0f} cm^2]")

        max_area = self.inlet_restriction.max_effective_area_m2
        if max_area is not None and A_eff_used > max_area:
            A_eff_used = max_area
            print(f"  [Note: A_eff capped at max_effective_area = {max_area*1e4:.0f} cm^2]")

        area_min = self.inlet_restriction.inlet_area_min_m2
        if area_min > 0 and A_eff_used < area_min:
            A_eff_used = area_min

        self.inlet_restriction.effective_area_m2 = A_eff_used

        inlet_part = self.parts[0]
        inlet_pipe_area = inlet_part.inner_area_m2

        if self.inlet_restriction.K_loss_override is not None:
            self.inlet_restriction.K_equiv = self.inlet_restriction.K_loss_override
        else:
            Cd = self.inlet_restriction.Cd
            self.inlet_restriction.K_equiv = 1.0 / (Cd ** 2)

        area_ratio = A_eff_used / inlet_pipe_area

        rho_ref = 1.2  # kg/m³, typical inlet
        K = self.inlet_restriction.K_equiv
        R_door_ref = K / (2 * rho_ref * A_eff_used**2)

        M2_TO_SQIN = 1550.0031
        door_area_sqin = full_door_area * M2_TO_SQIN
        A_eff_raw_sqin = A_eff_raw * M2_TO_SQIN
        A_eff_used_sqin = A_eff_used * M2_TO_SQIN

        print("\n" + "="*70)
        print("INLET RESTRICTION CONFIGURATION")
        print("="*70)
        print(f"Status: ENABLED")
        print(f"")
        print(f"Door Configuration ({door_type}):")
        print(f"  Dimensions:         {self.inlet_restriction.door_width_ft:.2f} ft × {self.inlet_restriction.door_height_ft:.2f} ft")
        print(f"  Full door area:     {door_area_sqin:.1f} sq in ({full_door_area:.5f} m²)")
        print(f"  Opening:            {self.inlet_restriction.door_opening_percent:.1f}%")
        print(f"  Discharge coeff:    Cd = {self.inlet_restriction.Cd:.2f}")
        print(f"  Leakage area:       {A_leak * M2_TO_SQIN:.2f} sq in (min area when closed)")
        print(f"")
        if door_type == 'swivel':
            print(f"Swivel Door Geometry:")
            angle_deg = (opening_pct / 100.0) * 90.0
            angle_rad = (opening_pct / 100.0) * (math.pi / 2.0)
            print(f"  Opening angle:      {angle_deg:.1f}°")
            print(f"  sin(angle):         {math.sin(angle_rad):.4f}")
        else:
            print(f"Sliding Door Geometry:")
            print(f"  Opening fraction:   {opening_pct/100:.2f}")
        print(f"")
        print(f"Calculated Values:")
        print(f"  A_eff_raw (geom):   {A_eff_raw_sqin:.2f} sq in ({A_eff_raw:.5f} m²)")
        print(f"  A_eff_used (total): {A_eff_used_sqin:.2f} sq in ({A_eff_used:.5f} m²)")
        print(f"  Inlet pipe area:    {inlet_pipe_area * M2_TO_SQIN:.2f} sq in (reference)")
        print(f"  Area ratio:         {area_ratio:.3f} ({area_ratio*100:.1f}% of pipe)")
        print(f"  K_equiv (=1/Cd²):   {self.inlet_restriction.K_equiv:.3f} "
              f"{'(from Cd)' if self.inlet_restriction.K_loss_override is None else '(manual override)'}")
        print(f"")
        print(f"Resistance Model (added to total_resistance):")
        print(f"  R_door = K / (2 * rho * A_eff^2)")
        print(f"  R_door ~= {R_door_ref:.2f} Pa*s^2/kg^2 (at rho={rho_ref} kg/m^3)")
        print(f"  dP_door = R_door * mdot^2")
        print("="*70 + "\n")

    def _update_door_opening(self, opening_percent: float):
        """Recalculate effective area for a new door opening percentage."""
        import math

        if not self.inlet_restriction.enable:
            return

        opening_percent = max(0.0, min(100.0, opening_percent))
        self.inlet_restriction.door_opening_percent = opening_percent

        full_door_area = self.inlet_restriction.door_area_m2
        door_type = self.inlet_restriction.door_type.lower()

        if door_type == 'swivel':
            angle_rad = (opening_percent / 100.0) * (math.pi / 2.0)
            A_eff_raw = full_door_area * math.sin(angle_rad)
        else:
            A_eff_raw = full_door_area * (opening_percent / 100.0)

        self.inlet_restriction.A_eff_raw_m2 = A_eff_raw
        A_leak = self.inlet_restriction.A_leak_m2
        area_mode = self.inlet_restriction.inlet_area_mode.lower()
        if area_mode == "quadrature":
            A_eff_used = math.sqrt(A_eff_raw**2 + A_leak**2)
        else:
            A_eff_used = A_eff_raw + A_leak

        self.inlet_restriction.effective_area_uncapped_m2 = A_eff_used

        cap_mode = self.inlet_restriction.inlet_area_cap_mode.lower()
        if cap_mode == "pipe":
            A_pipe = self.parts[0].inner_area_m2
            A_eff_used = min(A_eff_used, A_pipe)

        max_area = self.inlet_restriction.max_effective_area_m2
        if max_area is not None and A_eff_used > max_area:
            A_eff_used = max_area

        area_min = self.inlet_restriction.inlet_area_min_m2
        if area_min > 0 and A_eff_used < area_min:
            A_eff_used = area_min

        self.inlet_restriction.effective_area_m2 = A_eff_used

    def _load_inlet_temp_csv(self, csv_config):
        """Load inlet temperature CSV and create interpolation function."""
        from scipy import interpolate

        if isinstance(csv_config, str):
            csv_path = csv_config
            time_col = 'PC_Test Time [min]'
            temp_col = 'Flue Temp'
            time_unit = 'min'
            temp_unit = 'F'
        else:
            csv_path = csv_config.get('path')
            time_col = csv_config.get('time_col', 'PC_Test Time [min]')
            temp_col = csv_config.get('temp_col', 'Flue Temp')
            time_unit = csv_config.get('time_unit', 'min')
            temp_unit = csv_config.get('temp_unit', 'F')

        try:
            df = pd.read_csv(csv_path)
            self.inlet_temp_df = df

            if time_col not in df.columns:
                raise ValueError(f"CSV must have column '{time_col}'. Found: {list(df.columns)}")
            if temp_col not in df.columns:
                raise ValueError(f"CSV must have column '{temp_col}'. Found: {list(df.columns)}")

            time_raw = df[time_col].values
            temp_raw = df[temp_col].values

            valid_mask = ~(np.isnan(time_raw) | np.isnan(temp_raw))
            time_raw = time_raw[valid_mask]
            temp_raw = temp_raw[valid_mask]

            if len(time_raw) < 2:
                raise ValueError("CSV must have at least 2 valid data points")

            if time_unit == 'min':
                time_sec = time_raw * 60.0
            elif time_unit == 's':
                time_sec = time_raw
            else:
                raise ValueError(f"Unknown time_unit: {time_unit}. Use 'min' or 's'")

            if temp_unit == 'F':
                temp_K = (temp_raw - 32) * 5.0 / 9.0 + 273.15
                temp_display_unit = '°F'
            elif temp_unit == 'C':
                temp_K = temp_raw + 273.15
                temp_display_unit = '°C'
            elif temp_unit == 'K':
                temp_K = temp_raw
                temp_display_unit = 'K'
            else:
                raise ValueError(f"Unknown temp_unit: {temp_unit}. Use 'F', 'C', or 'K'")

            self.inlet_temp_interp = interpolate.interp1d(
                time_sec, temp_K,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )

            time_min_display = time_sec / 60.0
            print(f"\n" + "="*70)
            print("INLET TEMPERATURE FROM CSV")
            print("="*70)
            print(f"File: {csv_path}")
            print(f"Columns: time='{time_col}' ({time_unit}), temp='{temp_col}' ({temp_unit})")
            print(f"Data points: {len(time_raw)}")
            print(f"Time range: {time_min_display[0]:.1f} - {time_min_display[-1]:.1f} min "
                  f"({time_sec[0]:.0f} - {time_sec[-1]:.0f} s)")
            print(f"Temp range: {temp_raw.min():.0f} - {temp_raw.max():.0f} {temp_display_unit} "
                  f"({temp_K.min():.0f} - {temp_K.max():.0f} K)")

            time_diffs = np.diff(time_sec)
            avg_interval = np.mean(time_diffs)
            min_interval = np.min(time_diffs)
            print(f"Data interval: avg={avg_interval:.1f}s, min={min_interval:.1f}s")

            if avg_interval > self.dt:
                print(f"Note: CSV data interval ({avg_interval:.1f}s) > simulation dt ({self.dt}s)")
                print(f"      Linear interpolation will be used between data points")

            print("="*70 + "\n")

        except FileNotFoundError:
            raise FileNotFoundError(f"Inlet temperature CSV not found: {csv_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading inlet temperature CSV: {e}")

    def _get_inlet_temp(self, t_sec: float, T_ambient: float = 293.15) -> float:
        """Get inlet temperature at time t. Priority: func > CSV > default model."""
        if self.inlet_temp_func is not None:
            T_inlet = float(self.inlet_temp_func(t_sec))
        elif self.inlet_temp_interp is not None:
            T_inlet = float(self.inlet_temp_interp(t_sec))
        else:
            T_inlet = self._default_inlet_temp(t_sec, T_ambient=T_ambient)

        # Door temperature dilution: open door mixes cold ambient with hot gas
        if self.door_temp_dilution_enable and self.door_opening_func is not None:
            door_pct = self.door_opening_func(t_sec)
            dilution_frac = min(max((door_pct / 100.0) * self.door_temp_dilution_strength, 0.0), 1.0)
            T_inlet = T_inlet * (1.0 - dilution_frac) + T_ambient * dilution_frac

        return T_inlet

    def _initialize_catalytic_bypass(self, catalytic_bypass: Optional[Dict[str, Any]]):
        """Validate and compute effective K-loss for catalytic bypass."""
        if catalytic_bypass is None:
            self.catalytic_bypass = CatalyticBypassConfig()
            return

        if isinstance(catalytic_bypass, dict):
            self.catalytic_bypass = CatalyticBypassConfig(**catalytic_bypass)
        else:
            self.catalytic_bypass = catalytic_bypass

        if not self.catalytic_bypass.enable:
            return

        if not (0 <= self.catalytic_bypass.segment_index < self.total_segments):
            raise ValueError(
                f"segment_index must be between 0 and {self.total_segments - 1}, "
                f"got {self.catalytic_bypass.segment_index}"
            )

        if self.catalytic_bypass.K_bypass_open < 0:
            raise ValueError(
                f"K_bypass_open must be non-negative, got {self.catalytic_bypass.K_bypass_open}"
            )

        if self.catalytic_bypass.K_catalyst_closed < 0:
            raise ValueError(
                f"K_catalyst_closed must be non-negative, got {self.catalytic_bypass.K_catalyst_closed}"
            )

        state = self.catalytic_bypass.bypass_state

        if isinstance(state, str):
            state_lower = state.lower()
            if state_lower == 'open':
                self.catalytic_bypass.bypass_fraction = 1.0
            elif state_lower == 'closed':
                self.catalytic_bypass.bypass_fraction = 0.0
            else:
                raise ValueError(
                    f"bypass_state must be 'open', 'closed', or a number 0-100, "
                    f"got '{state}'"
                )
        elif isinstance(state, (int, float)):
            if not (0 <= state <= 100):
                raise ValueError(
                    f"bypass_state as number must be between 0 and 100, got {state}"
                )
            self.catalytic_bypass.bypass_fraction = state / 100.0
        else:
            raise ValueError(
                f"bypass_state must be 'open', 'closed', or a number 0-100, "
                f"got {type(state).__name__}"
            )

        # Linear interpolation: closed -> K_catalyst_closed, open -> K_bypass_open
        K_open = self.catalytic_bypass.K_bypass_open
        K_closed = self.catalytic_bypass.K_catalyst_closed

        self.catalytic_bypass.K_effective = (
            K_closed + self.catalytic_bypass.bypass_fraction * (K_open - K_closed)
        )

        # --- PRINT SUMMARY ---

        print("\n" + "="*70)
        print("CATALYTIC BYPASS DAMPER CONFIGURATION")
        print("="*70)
        print(f"Status: ENABLED")
        print(f"")
        print(f"Bypass Configuration:")

        # Display state nicely
        if self.catalytic_bypass.bypass_fraction == 1.0:
            state_str = "OPEN (bypass active, gases bypass catalyst)"
        elif self.catalytic_bypass.bypass_fraction == 0.0:
            state_str = "CLOSED (gases flow through catalyst)"
        else:
            state_str = f"{self.catalytic_bypass.bypass_fraction*100:.0f}% open (partial bypass)"

        print(f"  State:              {state_str}")
        print(f"  Segment index:      {self.catalytic_bypass.segment_index}")
        print(f"")
        print(f"K-Loss Coefficients:")
        print(f"  K when open:        {K_open:.3f} (low resistance, bypass)")
        print(f"  K when closed:      {K_closed:.3f} (through catalyst)")
        print(f"  K effective:        {self.catalytic_bypass.K_effective:.3f}")
        print("="*70 + "\n")

    def _initialize_leakage_config(self, leakage_config: Optional[Dict[str, Any]]):
        """Distribute leakage area across segments by pipe type."""
        if leakage_config is None:
            self.leakage_config = LeakageConfig()
            self.segment_leakage_area_m2 = np.zeros(self.total_segments)
            return

        if isinstance(leakage_config, dict):
            self.leakage_config = LeakageConfig(**leakage_config)
        else:
            self.leakage_config = leakage_config

        if not self.leakage_config.enable:
            self.segment_leakage_area_m2 = np.zeros(self.total_segments)
            return

        if self.leakage_config.total_leakage_area_cm2 < 0:
            raise ValueError(
                f"total_leakage_area_cm2 must be non-negative, "
                f"got {self.leakage_config.total_leakage_area_cm2}"
            )

        total_frac = self.leakage_config.black_pipe_fraction + self.leakage_config.insulated_fraction
        if abs(total_frac - 1.0) > 0.01:
            raise ValueError(
                f"black_pipe_fraction + insulated_fraction must equal 1.0, "
                f"got {total_frac:.2f}"
            )

        if not (0.0 < self.leakage_config.Cd_leak <= 1.0):
            raise ValueError(
                f"Cd_leak must be in range (0, 1.0], got {self.leakage_config.Cd_leak}"
            )

        total_area_m2 = self.leakage_config.total_leakage_area_cm2 * 1e-4  # cm² -> m²

        black_pipe_segs = []
        insulated_segs = []

        seg_idx = 0
        for part_idx, part in enumerate(self.parts):
            for local_seg in range(part.num_segments):
                if part.type == 'single':
                    black_pipe_segs.append(seg_idx)
                else:
                    insulated_segs.append(seg_idx)
                seg_idx += 1

        self.segment_leakage_area_m2 = np.zeros(self.total_segments)

        black_pipe_area = total_area_m2 * self.leakage_config.black_pipe_fraction
        insulated_area = total_area_m2 * self.leakage_config.insulated_fraction

        if len(black_pipe_segs) > 0:
            area_per_seg = black_pipe_area / len(black_pipe_segs)
            for seg in black_pipe_segs:
                self.segment_leakage_area_m2[seg] = area_per_seg

        if len(insulated_segs) > 0:
            area_per_seg = insulated_area / len(insulated_segs)
            for seg in insulated_segs:
                self.segment_leakage_area_m2[seg] = area_per_seg

        print("\n" + "="*70)
        print("LEAKAGE CONFIGURATION")
        print("="*70)
        print(f"Status: ENABLED")
        print(f"")
        print(f"Total leakage area: {self.leakage_config.total_leakage_area_cm2:.1f} cm² "
              f"({total_area_m2*1e4:.1f} cm²)")
        print(f"Discharge coeff:    Cd = {self.leakage_config.Cd_leak:.2f}")
        print(f"")
        print(f"Distribution:")
        print(f"  Black stove pipe: {self.leakage_config.black_pipe_fraction*100:.0f}% "
              f"({len(black_pipe_segs)} segments, {black_pipe_area*1e4:.1f} cm² total)")
        print(f"  Insulated pipe:   {self.leakage_config.insulated_fraction*100:.0f}% "
              f"({len(insulated_segs)} segments, {insulated_area*1e4:.1f} cm² total)")
        print("="*70 + "\n")

    def get_wind_cap_deltaP(self, step_index: int, Pa: float, T_out_K: float) -> float:
        """Return wind-induced pressure at chimney cap [Pa]."""
        cfg = self._wind_cfg
        if not cfg.enable:
            return 0.0

        if cfg.update_mode == "static" and self._deltaP_wind_cap_static is not None:
            return self._deltaP_wind_cap_static

        V = cfg.wind_speed_m_s
        if isinstance(V, (list, tuple)):
            idx = max(0, min(step_index, len(V) - 1))
            Vt = float(V[idx])
        else:
            Vt = float(V)

        dP = wind_cap_deltaP(
            Pa=Pa,
            T_out_K=T_out_K,
            V_wind=Vt,
            cap_type=cfg.cap_type,
            exposure=cfg.roof_exposure,
            theta_deg=cfg.wind_direction_deg,
            gust_factor=cfg.gust_factor,
            Cp_override=cfg.Cp_override,
        )
        self._last_deltaP_wind_cap = dP
        return dP

    def get_stack_differential_pressure(self, timestep=-1, measurement_height_m=None):
        """Calculate differential pressure profile along the stack [Pa, inWC]."""
        if timestep == -1:
            timestep = self.time_steps

        n_segs = self.total_segments

        heights = np.zeros(n_segs + 1)
        seg_heights = np.zeros(n_segs)

        cumulative_height = 0.0
        for seg in range(n_segs):
            part_idx, local_seg, part = self.get_segment_info(seg)
            seg_length = self.segment_lengths[seg]

            heights[seg] = cumulative_height
            seg_heights[seg] = cumulative_height + seg_length / 2
            cumulative_height += seg_length
        heights[n_segs] = cumulative_height

        seg_buoyancy = np.zeros(n_segs)
        seg_friction = np.zeros(n_segs)

        for seg in range(n_segs):
            part_idx, local_seg, part = self.get_segment_info(seg)
            seg_length = self.segment_lengths[seg]
            diameter = part.inner_diameter_m

            T_gas = self.T_gas[timestep, seg]
            rho_gas = self.density[timestep, seg]
            rho_ambient = self.physics.air_density(part.T_ambient_K, part.Pa)
            velocity = self.velocity[timestep, seg]

            if part.orientation.lower() == 'vertical':
                seg_buoyancy[seg] = 9.81 * seg_length * (rho_ambient - rho_gas)

            if abs(velocity) > 0.01:
                mu = self.physics.air_viscosity(T_gas)
                Re = rho_gas * abs(velocity) * diameter / mu
                Re = max(Re, 10)

                relative_roughness = part.roughness_m / diameter
                if Re < 2300:
                    f = 64 / Re
                elif relative_roughness < 1e-6:
                    f = 0.316 / Re**0.25
                else:
                    f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2

                seg_friction[seg] = f * (seg_length / diameter) * 0.5 * rho_gas * velocity**2

        # Integrate from top down: exit at atmospheric, ΔP = -buoyancy + friction
        dP_profile = np.zeros(n_segs + 1)
        dP_profile[n_segs] = 0.0

        buoyancy_above = 0.0
        friction_above = 0.0

        for seg in range(n_segs - 1, -1, -1):
            buoyancy_above += seg_buoyancy[seg]
            friction_above += seg_friction[seg]
            dP_profile[seg] = -buoyancy_above + friction_above

        PA_PER_INWC = 249.09  # Pa per inWC
        dP_inWC = dP_profile / PA_PER_INWC

        if measurement_height_m is None:
            measurement_height_m = 0.0

        dP_at_height = np.interp(measurement_height_m, heights, dP_profile)

        return {
            'height_m': heights,
            'dP_Pa': dP_profile,
            'dP_inWC': dP_inWC,
            'dP_inlet_Pa': dP_profile[0],
            'dP_inlet_inWC': dP_profile[0] / PA_PER_INWC,
            'dP_at_height_Pa': dP_at_height,
            'dP_at_height_inWC': dP_at_height / PA_PER_INWC,
            'measurement_height_m': measurement_height_m,
            'seg_buoyancy': seg_buoyancy,
            'seg_friction': seg_friction
        }

    def print_stack_pressure_comparison(self, measured_dP_inWC, measurement_height_m=0.0, timestep=-1):
        """Compare simulated differential pressure to a field measurement."""
        result = self.get_stack_differential_pressure(timestep, measurement_height_m)

        simulated_inWC = result['dP_at_height_inWC']
        simulated_Pa = result['dP_at_height_Pa']
        measured_Pa = measured_dP_inWC * 249.09

        diff_inWC = simulated_inWC - measured_dP_inWC
        diff_Pa = simulated_Pa - measured_Pa

        print("\n" + "="*60)
        print("STACK PRESSURE COMPARISON")
        print("="*60)
        print(f"\nMeasurement height: {measurement_height_m:.2f} m")
        print(f"\n{'':20} {'Simulated':>15} {'Measured':>15} {'Difference':>15}")
        print("-"*65)
        print(f"{'dP (inWC)':20} {simulated_inWC:>15.4f} {measured_dP_inWC:>15.4f} {diff_inWC:>15.4f}")
        print(f"{'dP (Pa)':20} {simulated_Pa:>15.2f} {measured_Pa:>15.2f} {diff_Pa:>15.2f}")

        print("\n" + "-"*65)
        if abs(diff_inWC) < 0.005:
            print("[OK] Excellent agreement (within 0.005 inWC)")
        elif abs(diff_inWC) < 0.01:
            print("[OK] Good agreement (within 0.01 inWC)")
        elif abs(diff_inWC) < 0.02:
            print("~ Fair agreement (within 0.02 inWC)")
        else:
            print("[X] Significant difference - check inputs or measurement location")

        print(f"\nInlet dP: {result['dP_inlet_inWC']:.4f} inWC ({result['dP_inlet_Pa']:.2f} Pa)")

        total_buoyancy = sum(result['seg_buoyancy'])
        total_friction = sum(result['seg_friction'])
        print(f"Total buoyancy: {total_buoyancy:.2f} Pa")
        print(f"Total friction: {total_friction:.2f} Pa")
        print("="*60)

    def get_static_pressure_profile(self, timestep=-1):
        """Calculate static (gauge) pressure profile along the stack."""
        if timestep == -1:
            timestep = self.time_steps

        n_segs = self.total_segments

        heights = np.zeros(n_segs + 1)
        cumulative_height = 0.0

        for seg in range(n_segs):
            part_idx, local_seg, part = self.get_segment_info(seg)
            seg_length = self.segment_lengths[seg]
            heights[seg] = cumulative_height
            cumulative_height += self.segment_vertical_heights[seg]
        heights[n_segs] = cumulative_height

        # Ambient pressure column: per-segment density from each part's T_ambient
        P_ambient = np.zeros(n_segs + 1)
        P_ambient[0] = self.parts[0].Pa
        for seg in range(n_segs):
            part_idx, local_seg, part = self.get_segment_info(seg)
            rho_ambient = self.physics.air_density(part.T_ambient_K, part.Pa)
            P_ambient[seg + 1] = P_ambient[seg] - rho_ambient * 9.81 * self.segment_vertical_heights[seg]

        P_inside = np.zeros(n_segs + 1)
        deltaP_wind = getattr(self, '_last_deltaP_wind_cap', 0.0)

        dp_exit = 0.0
        K_exit = getattr(self, 'K_exit', 0.0)
        if K_exit > 0 and timestep > 0:
            last_seg = n_segs - 1
            rho_exit = self.density[timestep, last_seg]
            v_exit = self.velocity[timestep, last_seg]
            dp_exit = K_exit * 0.5 * rho_exit * v_exit**2

        P_inside[n_segs] = P_ambient[n_segs] + deltaP_wind + dp_exit

        total_dp_buoyancy = 0.0
        total_dp_friction = 0.0
        total_dp_fittings = 0.0

        for seg in range(n_segs - 1, -1, -1):
            part_idx, local_seg, part = self.get_segment_info(seg)
            seg_length = self.segment_lengths[seg]
            diameter = part.inner_diameter_m

            T_gas = self.T_gas[timestep, seg]
            rho_gas = self.density[timestep, seg]
            velocity = self.velocity[timestep, seg]

            dP_hydrostatic = -rho_gas * 9.81 * self.segment_vertical_heights[seg]

            dP_friction = 0.0
            if abs(velocity) > 0.01:
                mu = self.physics.air_viscosity(T_gas)
                Re = rho_gas * abs(velocity) * diameter / mu
                Re = max(Re, 10)

                relative_roughness = part.roughness_m / diameter
                if Re < 2300:
                    f = 64 / Re
                elif relative_roughness < 1e-6:
                    f = 0.316 / Re**0.25
                else:
                    f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2

                dP_friction = f * (seg_length / diameter) * 0.5 * rho_gas * velocity**2

            dP_fitting = 0.0
            elbow_key = (part_idx, local_seg)
            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                for elbow_spec in self.segment_elbows[elbow_key]:
                    K_fitting = base_K(elbow_spec) * elbow_spec.vane_factor
                    dP_fitting += K_fitting * 0.5 * rho_gas * velocity**2

            # Accumulate totals for diagnostics
            # Buoyancy = ambient hydrostatic - gas hydrostatic (positive = upward draft)
            rho_amb_seg = self.physics.air_density(part.T_ambient_K, part.Pa)
            total_dp_buoyancy += (rho_amb_seg - rho_gas) * 9.81 * self.segment_vertical_heights[seg]
            total_dp_friction += dP_friction
            total_dp_fittings += dP_fitting

            # P_bottom = P_top - dP_hydrostatic + dP_friction + dP_fitting
            P_inside[seg] = P_inside[seg + 1] - dP_hydrostatic + dP_friction + dP_fitting

        # --- Inlet restriction (diagnostic only, NOT subtracted) ---
        # The momentum solver already accounts for dp_door/dp_stove in mass flow.
        # The top-down integration reflects this through reduced velocities/friction.
        # Subtracting dp_door uniformly was double-counting (confirmed by 0130 test:
        # TC3 and TOP match experiment perfectly WITHOUT dp_door subtraction).
        _dp_door = 0.0
        _dp_stove = 0.0
        _A_eff = 0.0
        _mdot = 0.0
        if self.inlet_restriction.enable and timestep > 0:
            _A_eff = self.A_eff_history[timestep] if hasattr(self, 'A_eff_history') \
                else self.inlet_restriction.effective_area_m2
            if _A_eff > 0:
                Cd = self.inlet_restriction.Cd
                K_door = 1.0 / (Cd ** 2) if Cd > 0 else 0.0
                rho_0 = self.density[timestep, 0]
                _mdot = self.mass_flow_rate[timestep] if self.mass_flow_rate.ndim == 1 \
                    else self.mass_flow_rate[timestep, 0]
                _dp_door = K_door / (2.0 * rho_0 * _A_eff ** 2) * _mdot ** 2
                K_stove = getattr(self.inlet_restriction, 'K_stove', 0.0)
                if K_stove > 0:
                    A_pipe = self.parts[0].inner_area_m2
                    _dp_stove = K_stove / (2.0 * rho_0 * A_pipe ** 2) * _mdot ** 2
                # NOT subtracted: already in the integration via momentum solver

        # --- Room depressurization ---
        # dp_room IS needed: the top boundary is at P_ambient(outdoor),
        # but the room (chimney inlet) is at P_ambient - dp_room.
        # The integration doesn't know this — it only uses gas properties.
        _dp_room = 0.0
        if True:
            t_s = timestep * self.dt
            if self._room_pressure_func is not None:
                _dp_room = self._room_pressure_func(t_s)
            elif self.room_plenum.cfg.enable:
                _dp_room = self.room_pressure_offset_Pa
            if _dp_room != 0.0:
                P_inside[:] -= _dp_room

        P_gauge = P_inside - P_ambient

        return {
            'height_m': heights,
            'P_gauge_Pa': P_gauge,
            'P_absolute_Pa': P_inside,
            'P_ambient_Pa': P_ambient,
            # Pressure component breakdown (all positive = reduces gauge pressure)
            'dp_buoyancy': total_dp_buoyancy,
            'dp_friction': total_dp_friction,
            'dp_fittings': total_dp_fittings,
            'dp_door': _dp_door,
            'dp_stove': _dp_stove,
            'dp_room': _dp_room,
            'dp_wind': deltaP_wind,
            'dp_exit': dp_exit,
            'A_eff_m2': _A_eff,
            'mdot': _mdot,
        }

    def get_frontend_outputs(self, parts_to_plot=None):
        """Build JSON-serializable payload for the frontend.

        Call after run_simulation(). Reads instance attributes (T_gas, time_array, parts, ...).

        The temperature plot shows the mid-segment of each part listed in `parts_to_plot`
        (1-indexed positions in `self.parts`). Default: first + last part. Each series is
        labeled with the user-visible component number, which counts every fitting in
        `self.elbows` that sits at-or-before the part in the canvas order. So a layout of
        [pipe, elbow, pipe] labels the two series "Part 1" and "Part 3" (the elbow occupies
        position #2). With no fittings, labels match the parts-array index 1:1.

        UL 103 result is PASS if the static pressure ever crosses the required draft
        threshold (0.006 in.w.c./ft of vertical chimney, Section 25.1) during the run.

        Max temperature and max static pressure are computed over the entire simulation,
        not a steady-state window.

        Args:
            parts_to_plot: list of 1-indexed part numbers to include in the temperature
                plot. Default = [1, len(self.parts)] (first + last).

        Returns:
            dict with both the frontend contract names and legacy aliases for backward
            compatibility:
                ul103_risk                  "low" (PASS) | "high" (FAIL)
                ul103_result                "PASS" | "FAIL"  (legacy alias)
                ul103_required_Pa           float (threshold from Section 25.1)
                backdraft_risk              "low" | "medium" | "high"
                max_static_pressure_ss_Pa   float (max over full sim, +up = upward draft)
                max_static_pressure_Pa      float  (legacy alias, identical value)
                max_temperature_ss_C        float (max gas temp at first/last segment over full sim)
                max_temperature_C           float  (legacy alias, identical value)
                static_pressure_plot        {time_min, static_pressure_Pa}
                temperature_plot            {time_min, series: [{label, temperature_C}, ...]}
        """
        n_parts = len(self.parts)

        n_steps = self.T_gas.shape[0]
        time_s = self.time_array
        time_min = time_s / 60.0

        # Static pressure at segment 0. Sign flipped so positive = normal upward draft.
        static_pressure_Pa = np.zeros(n_steps)
        for step in range(n_steps):
            prof = self.get_static_pressure_profile(timestep=step)
            static_pressure_Pa[step] = -prof['P_gauge_Pa'][0]

        T_first = self.T_gas[:, 0]
        T_last = self.T_gas[:, -1]

        max_static_pressure_Pa = float(static_pressure_Pa.max())
        max_temperature_C = float(np.maximum(T_first, T_last).max() - 273.15)

        # UL 103 draft requirement: 0.006 in.w.c. per ft of vertical chimney (Section 25.1).
        # PASS if the static pressure ever crosses the threshold during the run.
        h_vert_m = sum(p.height_m for p in self.parts if p.orientation.lower() == 'vertical')
        ul_required_Pa = 0.006 * (h_vert_m * 3.28084) * 249.089
        if ul_required_Pa > 0 and max_static_pressure_Pa >= ul_required_Pa:
            ul103_result = "PASS"
        else:
            ul103_result = "FAIL"

        # Backdraft risk: high = never established, medium = early dip then recovered, low = clean.
        # "Established" is judged from the second half of the run (proxy for settled draft).
        THRESH_ESTABLISHED_PA = 2.0
        THRESH_BACKDRAFT_PA = 0.5
        EARLY_WINDOW_S = 180.0
        BACKDRAFT_DURATION_S = 30.0

        late_mean_Pa = float(static_pressure_Pa[n_steps // 2:].mean())
        if late_mean_Pa < THRESH_ESTABLISHED_PA:
            backdraft_risk = "high"
        else:
            early_mask = time_s < EARLY_WINDOW_S
            backdraft_duration_s = float((static_pressure_Pa[early_mask] < -THRESH_BACKDRAFT_PA).sum()) * self.dt
            backdraft_risk = "medium" if backdraft_duration_s > BACKDRAFT_DURATION_S else "low"

        # Mid-segment temperature series for each requested part. Default = first + last.
        part_ranges = []
        s0 = 0
        for part in self.parts:
            part_ranges.append((s0, s0 + part.num_segments))
            s0 += part.num_segments

        if parts_to_plot is None:
            parts_to_plot = [1, n_parts] if n_parts > 1 else [1]

        # Display label = user-visible component number, counting fittings in self.elbows
        # that sit at-or-before this part. A fitting "sits before pipe k" iff its
        # between_parts[1] <= k (next-part index points at or before this pipe). Inlet
        # tee with between_parts=[-1, 0] precedes pipe 0; elbow with between_parts=[0, 1]
        # precedes pipe 1; etc. Falls back to plain "Part k" if no elbows are stored.
        def _display_label(part_idx):
            n_before = 0
            for elbow_dict in (self.elbows or []):
                bp = elbow_dict.get('between_parts')
                if not isinstance(bp, (list, tuple)) or len(bp) < 2:
                    continue
                next_p = bp[1]
                if isinstance(next_p, (int, float)) and 0 <= int(next_p) <= part_idx:
                    n_before += 1
            return part_idx + n_before + 1

        series = []
        for part_num in parts_to_plot:
            seg_lo, _seg_hi = part_ranges[part_num - 1]
            bottom_seg = seg_lo
            series.append({
                "label": f"Part {_display_label(part_num - 1)}",
                "temperature_C": (self.T_gas[:, bottom_seg] - 273.15).tolist(),
            })

        # PASS/FAIL -> low/high so the frontend RiskBadge has a level to render.
        ul103_risk = "low" if ul103_result == "PASS" else "high"

        return {
            # Contract names the frontend reads:
            "ul103_risk": ul103_risk,
            "max_static_pressure_ss_Pa": max_static_pressure_Pa,
            "max_temperature_ss_C": max_temperature_C,
            # Legacy aliases (kept so existing notebooks/diagnostics keep working):
            "ul103_result": ul103_result,
            "max_static_pressure_Pa": max_static_pressure_Pa,
            "max_temperature_C": max_temperature_C,
            # Shared:
            "ul103_required_Pa": float(ul_required_Pa),
            "backdraft_risk": backdraft_risk,
            "static_pressure_plot": {
                "time_min": time_min.tolist(),
                "static_pressure_Pa": static_pressure_Pa.tolist(),
            },
            "temperature_plot": {
                "time_min": time_min.tolist(),
                "series": series,
            },
        }


# Bind optional plotting methods (gitignored, dev only)
try:
    from . import chimney_plots as _cp
    for _name in _cp.PLOT_METHODS:
        setattr(Chimney, _name, getattr(_cp, _name))
except ImportError:
    pass
