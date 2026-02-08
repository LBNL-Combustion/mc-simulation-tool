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

@dataclass
class InletRestrictionConfig:
    """Configuration for inlet flow restriction (door, damper, etc.)

    Models restriction at system inlet based on door geometry and opening.
    Supports two door types with different effective area calculations:

    - **swivel** (hinged door): A_eff = A_door × sin(angle), angle = opening% × π/2
    - **sliding** (slide door): A_eff = A_door × opening% (linear)

    The restriction is modeled as a quadratic resistance in mass-flow space:
        ΔP = R_door × ṁ²
    where R_door = K_equiv / (2 × ρ × A_eff²)

    This resistance is added to total_resistance in the momentum solver,
    ensuring consistent pressure-flow relationships.

    To avoid infinite resistance at A_eff→0, a minimum leakage area is added:
        A_eff_used = A_eff_raw + A_leak
    This provides finite resistance even when fully closed.

    Dynamic Door Control:
        Use door_opening_func parameter in Chimney to provide a function
        that returns door opening percentage at any time t.

    Attributes:
        enable: Activate restriction feature
        door_type: 'swivel' (hinged) or 'sliding' - affects A_eff calculation
        door_opening_percent: How much door is open (0-100%)
        door_width_ft: Door width in feet
        door_height_ft: Door height in feet
        door_cracked_percent: If set, enables simple dynamic door schedule
        door_wide_open_time_s: How long door stays wide open (default 90s)
        Cd: Discharge coefficient (default 0.62)
        A_leak_m2: Minimum leakage area when closed (default 50 cm² = 0.005 m²)
        max_effective_area_m2: Maximum effective area (caps at downstream restriction).
            Set to pipe/tee area when door opens into restricted space. If None, no cap.
            For 6" pipe: 0.0182 m² = 182 cm²
        K_loss_override: Optional manual K-loss coefficient override

        # Computed values (set automatically during initialization)
        door_area_m2: Full door area computed from dimensions [m²]
        A_eff_raw_m2: Raw effective area from geometry (before adding leakage)
        effective_area_m2: Total effective area = A_eff_raw + A_leak
        K_equiv: Equivalent loss coefficient = 1/Cd²
    """
    # User inputs
    enable: bool = False
    door_type: str = 'swivel'  # 'swivel' (hinged) or 'sliding'
    door_opening_percent: float = 100.0  # 0-100%
    door_width_ft: float = 1.0  # feet
    door_height_ft: float = 1.0  # feet

    # Dynamic door schedule (simple built-in schedule)
    # For more control, use door_opening_func in Chimney instead
    door_cracked_percent: Optional[float] = None
    door_wide_open_time_s: float = 90.0

    # Discharge coefficient for orifice model
    Cd: float = 0.62

    # Leakage area - minimum flow area when door is fully closed
    # Default 0.005 m² = 50 cm² represents gaps around a closed door
    A_leak_m2: float = 0.0045

    # Maximum effective area - caps A_eff at downstream restriction (tee/pipe)
    # If None, no cap is applied (door geometry determines area)
    # Set to pipe area (e.g., 0.0182 m² for 6" pipe) when door opens into restricted space
    max_effective_area_m2: Optional[float] = None

    # Stove body K-loss coefficient - represents resistance INSIDE the stove
    # that applies regardless of door position:
    #   - Fuel bed resistance (K ≈ 2-4 depending on loading)
    #   - Baffle losses (K ≈ 1-2)
    #   - Flue collar entry (K ≈ 0.5)
    # Total K_stove ≈ 4-6 for typical woodstove
    # This is added as: R_stove = K_stove / (2 × ρ × A_pipe²)
    # Set to 0 to disable (legacy behavior)
    K_stove: float = 0.0

    # --- Effective area combination mode ---
    # "linear" (default): A_eff = A_eff_raw + A_leak  (current behavior)
    # "quadrature": A_eff = sqrt(A_eff_raw^2 + A_leak^2)
    #   Quadrature avoids the non-smooth kink at A_eff_raw=0 and is more
    #   physically motivated (two parallel flow paths).
    inlet_area_mode: str = "linear"

    # --- Effective area cap mode ---
    # "none" (default): no additional cap (only max_effective_area_m2 applies)
    # "pipe": cap A_eff at the first part's pipe cross-section area
    inlet_area_cap_mode: str = "none"

    # Optional override (if set, bypasses Cd calculation)
    K_loss_override: Optional[float] = None

    # Computed values (set during initialization)
    door_area_m2: float = 0.0  # Full door area from dimensions [m²]
    A_eff_raw_m2: float = 0.0  # Raw area from geometry
    effective_area_m2: float = 0.0  # Total area (method depends on inlet_area_mode)
    K_equiv: float = 0.0  # = 1/Cd² unless K_loss_override is set


@dataclass
class CatalyticBypassConfig:
    """Configuration for catalytic bypass damper

    Models a bypass damper that allows flue gases to either:
    - OPEN: Bypass the catalytic combustor (low resistance, startup mode)
    - CLOSED: Flow through the catalyst honeycomb (higher resistance, steady burn)

    Uses a simple variable resistance approach where the effective K-loss
    changes based on bypass state.

    Attributes:
        enable: Activate bypass feature
        bypass_state: 'open', 'closed', or float 0-100 (0=closed, 100=open)
        segment_index: Which segment contains the catalyst (0-indexed)
        K_bypass_open: K-loss coefficient when bypass is open (low, ~0.3-1.0)
        K_catalyst_closed: K-loss coefficient through catalyst (~2-5 typical)

        # Computed values (set automatically during initialization)
        K_effective: The actual K-loss used based on bypass_state
        bypass_fraction: 0.0 (closed) to 1.0 (open), computed from bypass_state
    """
    # User inputs
    enable: bool = False
    bypass_state: Any = 'open'  # 'open', 'closed', or 0-100 float
    segment_index: int = 0  # Which segment has the catalyst
    K_bypass_open: float = 0.5  # Low resistance when bypassing
    K_catalyst_closed: float = 3.0  # Higher resistance through catalyst

    # Computed values (set during initialization)
    K_effective: float = 0.0
    bypass_fraction: float = 1.0  # 0=closed (through catalyst), 1=open (bypass)

@dataclass
class LeakageConfig:
    """Configuration for distributed pipe joint leakage

    Models air infiltration/exfiltration through pipe joints (slip joints, seams).
    Leakage flow depends on local gauge pressure:
    - Positive P_gauge (above ambient) → gas leaks OUT
    - Negative P_gauge (below ambient) → ambient air leaks IN

    Uses VENT-II style orifice model:
        m_leak = Cd × A_leak × sqrt(2 × rho × |ΔP|)

    Leakage area is distributed across segments based on pipe type:
    - Black stove pipe has slip joints → higher leakage
    - Insulated pipe has sealed joints → lower leakage

    Attributes:
        enable: Activate leakage feature
        total_leakage_area_cm2: Total leakage area in cm² (split between parts)
        black_pipe_fraction: Fraction of leakage in black stove pipe (0-1)
        insulated_fraction: Fraction of leakage in insulated pipe (0-1)
        Cd_leak: Discharge coefficient for leaks (default 0.61)
    """
    enable: bool = False
    total_leakage_area_cm2: float = 10.0  # Total leakage area in cm²
    black_pipe_fraction: float = 0.6     # 60% of leakage in black pipe
    insulated_fraction: float = 0.4      # 40% of leakage in insulated pipe
    Cd_leak: float = 0.61                # Discharge coefficient for leaks


@dataclass
class ChimneyPart:
    """Configuration for a single or triple-wall chimney part

    This class stores all the properties needed to define one section of the chimney.
    The chimney can be made up of multiple parts with different properties.
    """
    # BASIC CONFIGURATION
    type: str  # 'single' for simple wall, 'triple' for insulated wall with air gaps
    inner_diameter_ft: float  # Flow area diameter in feet (will be converted to meters)
    height_ft: float  # Height of this chimney section in feet
    z_ft: float  # Elevation above reference point in feet (affects buoyancy)
    orientation: str  # 'vertical' or 'horizontal' - affects natural convection
    T_ambient_K: float  # Ambient temperature outside the chimney in Kelvin
    Pa: float  # Atmospheric pressure in Pascals
    num_segments: int  # Number of computational cells to divide this part into
    
    # SINGLE WALL PROPERTIES (used when type='single')
    # These define the material properties for a simple single-layer wall
    thickness_w_m: float = 0.001  # Wall thickness in meters
    density_w_kg_m3: float = 7850  # Wall material density (kg/m³) - default is steel
    cp_w_J_kg_K: float = 460  # Wall specific heat (J/kg·K) - energy to heat 1kg by 1K
    k_w_W_m_K: float = 50  # Wall thermal conductivity (W/m·K) - heat flow rate
    epsilon_w: float = 0.8  # Wall emissivity (0-1) - ability to radiate heat
    
    # TRIPLE WALL PROPERTIES (used when type='triple')
    # Triple wall chimneys have 3 layers for better insulation and safety
    
    # Inner wall (liner) - contacts the hot flue gases
    thickness_inner_m: float = 0.0008  # Inner wall thickness (m) - thin stainless steel
    density_inner_kg_m3: float = 8000  # Inner wall density (kg/m³) - stainless steel
    cp_inner_J_kg_K: float = 500  # Inner wall specific heat (J/kg·K)
    k_inner_W_m_K: float = 16  # Inner wall thermal conductivity (W/m·K) - lower than regular steel
    epsilon_inner: float = 0.3  # Inner wall emissivity - polished metal radiates less
    
    # Middle layer (insulation) - provides thermal resistance
    thickness_middle_m: float = 0.025  # Middle layer thickness (m) - 25mm insulation
    density_middle_kg_m3: float = 100  # Middle layer density (kg/m³) - lightweight insulation
    cp_middle_J_kg_K: float = 840  # Middle layer specific heat (J/kg·K)
    k_middle_W_m_K: float = 0.04  # Middle layer thermal conductivity (W/m·K) - very low for insulation
    
    # Outer wall (structural) - provides strength and weather protection
    thickness_outer_m: float = 0.001  # Outer wall thickness (m) - structural steel
    density_outer_kg_m3: float = 7850  # Outer wall density (kg/m³) - regular steel
    cp_outer_J_kg_K: float = 460  # Outer wall specific heat (J/kg·K)
    k_outer_W_m_K: float = 50  # Outer wall thermal conductivity (W/m·K)
    epsilon_outer: float = 0.8  # Outer wall emissivity - painted/oxidized surface
    
    # AIR GAP PROPERTIES (for triple wall)
    # Air gaps provide additional insulation through trapped air
    air_gap_inner_m: float = 0.0  # Air gap between inner and middle layers (m)
    air_gap_outer_m: float = 0.0  # Air gap between middle and outer layers (m)
    
    # GENERAL PROPERTIES (apply to all wall types)
    film_w: float = 5e-5  # Max condensate film thickness (m) - for condensation modeling
    has_fittings: bool = False  # True if section has elbows/tees (increases pressure drop)
    roughness_m: float = 1e-6  # Surface roughness in meters - affects friction factor
    humidity_ratio: float = 0.01  # kg water / kg dry air - for condensation calculations

    # CORRUGATION PROPERTIES (optional, for corrugated pipes)
    # If None, pipe is treated as smooth with base roughness_m
    # If dict provided, corrugation effects are computed and folded into equivalent roughness
    corrugation: Optional[Dict[str, Any]] = None  # See SpiralCorrugationProperties for structure
    
    def __post_init__(self):
        """Convert units and validate inputs
        
        This method runs automatically after initialization to:
        1. Convert Imperial units (feet) to SI units (meters)
        2. Calculate derived geometric properties
        3. Validate that inputs make physical sense
        """
        # Convert feet to meters (1 ft = 0.3048 m)
        self.inner_diameter_m = self.inner_diameter_ft * 0.3048
        self.height_m = self.height_ft * 0.3048
        self.z_m = self.z_ft * 0.3048
        
        # Calculate flow area and perimeter for heat transfer calculations
        self.inner_area_m2 = np.pi * (self.inner_diameter_m / 2) ** 2  # Cross-sectional area
        self.inner_perimeter_m = np.pi * self.inner_diameter_m  # Wetted perimeter
        
        # Calculate outer diameter based on wall type
        if self.type == 'single':
            self.outer_diameter_m = self.inner_diameter_m + 2 * self.thickness_w_m
        elif self.type == 'triple':
            # For triple wall: inner liner + gap + middle + gap + outer
            total_thickness = (self.thickness_inner_m + self.air_gap_inner_m + 
                             self.thickness_middle_m + self.air_gap_outer_m + 
                             self.thickness_outer_m)
            self.outer_diameter_m = self.inner_diameter_m + 2 * total_thickness
            
            # Calculate intermediate diameters for triple wall
            self.liner_outer_diameter_m = self.inner_diameter_m + 2 * self.thickness_inner_m
            self.middle_inner_diameter_m = self.liner_outer_diameter_m + 2 * self.air_gap_inner_m
            self.middle_outer_diameter_m = self.middle_inner_diameter_m + 2 * self.thickness_middle_m
            self.outer_inner_diameter_m = self.middle_outer_diameter_m + 2 * self.air_gap_outer_m
        else:
            raise ValueError(f"Unknown wall type: {self.type}. Must be 'single' or 'triple'")
        
        # Validate inputs
        if self.num_segments < 1:
            raise ValueError("num_segments must be at least 1")
        if self.inner_diameter_m <= 0:
            raise ValueError("inner_diameter must be positive")
        if self.height_m <= 0:
            raise ValueError("height must be positive")

        # Process corrugation properties if provided
        self._process_corrugation_properties()

    def _process_corrugation_properties(self):
        """Process corrugation properties and compute equivalent roughness

        This preprocessing function handles:
        1. Annular corrugations (existing code path)
        2. Spiral corrugations (NEW - compute helix effects and fold seam losses)
        3. None (smooth pipe with base roughness)

        The corrugation effects are folded into self.roughness_m so that
        existing solvers work without modification.
        """
        if self.corrugation is None:
            # No corrugation - use base roughness as-is
            # self.roughness_m already set from input or default
            return

        corr_type = self.corrugation.get('type', 'none').lower()

        if corr_type == 'none':
            # Explicitly no corrugation
            return

        elif corr_type == 'spiral':
            # NEW: Spiral corrugation preprocessing
            self._process_spiral_corrugation()

        elif corr_type == 'annular':
            # Annular corrugations - use existing model
            self._process_annular_corrugation()

        else:
            raise ValueError(f"Unknown corrugation type: {corr_type}. "
                           f"Must be 'none', 'spiral', or 'annular'")

    def _process_spiral_corrugation(self):
        """Process spiral corrugation and compute equivalent roughness

        Converts spiral corrugation geometry into equivalent roughness that
        captures both form drag and seam losses. This allows existing solvers
        to work without modification.
        """
        from .corrugation_effects import (
            SpiralCorrugationProperties,
            compute_spiral_corrugation_effects
        )

        corr = self.corrugation

        # Extract dimensions (convert mm to m)
        depth_m = corr.get('depth_mm', 3.0) / 1000.0
        pitch_m = corr.get('pitch_mm', 180.0) / 1000.0
        num_starts = corr.get('num_starts', 1)
        base_roughness_m = corr.get('base_roughness_mm', 0.001) / 1000.0

        # Seam properties
        has_lock_seam = corr.get('has_lock_seam', True)
        seam_protrusion_m = corr.get('seam_protrusion_mm', 2.0) / 1000.0
        seam_width_m = corr.get('seam_width_mm', 8.0) / 1000.0
        K_seam = corr.get('K_seam_per_crossing', 0.05)
        K_lock = corr.get('K_lock_seam', 0.03)

        # Version 2.0: Swirl and heat transfer enhancement flags
        enable_swirl = corr.get('enable_spiral_swirl', False)
        swirl_friction_cap = corr.get('swirl_friction_cap', 0.15)
        enable_HT = corr.get('enable_spiral_HT_enhancement', False)
        HT_enhancement_cap = corr.get('HT_enhancement_cap', 0.25)

        # Create SpiralCorrugationProperties object
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

        # Compute equivalent roughness using segment length
        # Use average segment length for this part
        segment_length = self.height_m / self.num_segments

        # Compute effects
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

        # Store results for diagnostics
        self.corrugation_effects = effects

        # Update roughness to equivalent value that includes all effects
        self.roughness_m = effects['equivalent_roughness_m']

        # Print diagnostic information
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
        """Process annular corrugation using full physics model

        Annular corrugations are perpendicular rings that create recirculation
        and form drag. The model accounts for depth/pitch ratio, Reynolds effects,
        and diameter scaling.
        """
        from .corrugation_effects import (
            CorrugationProperties,
            CorrugatedPipeFriction
        )

        corr = self.corrugation

        # Extract properties (convert mm to m)
        depth_m = corr.get('depth_mm', 3.0) / 1000.0
        pitch_m = corr.get('pitch_mm', 25.0) / 1000.0
        base_roughness_m = corr.get('base_roughness_mm', 0.001) / 1000.0

        # Create CorrugationProperties object
        corr_props = CorrugationProperties(
            depth_m=depth_m,
            pitch_m=pitch_m,
            base_roughness_m=base_roughness_m
        )

        # Use full annular corrugation model
        friction_calc = CorrugatedPipeFriction()

        # Calculate corrugated friction factor at representative Re
        representative_Re = 10000  # Typical turbulent Re for flues
        f_corrugated = friction_calc.calculate_corrugated_friction_factor(
            Re=representative_Re,
            D=self.inner_diameter_m,
            corrugation=corr_props
        )

        # Compute smooth pipe friction factor for comparison
        f_smooth = friction_calc._colebrook_white(representative_Re, base_roughness_m / self.inner_diameter_m)

        # Invert to find equivalent roughness that gives f_corrugated
        # Use Newton-Raphson iteration
        eps_eq = base_roughness_m  # Initial guess
        for iteration in range(15):
            f_current = friction_calc._colebrook_white(representative_Re, eps_eq / self.inner_diameter_m)

            if abs(f_current - f_corrugated) < 1e-6:
                break

            # Numerical derivative
            deps = eps_eq * 0.001 + 1e-9
            f_perturbed = friction_calc._colebrook_white(representative_Re, (eps_eq + deps) / self.inner_diameter_m)
            df_deps = (f_perturbed - f_current) / deps

            if abs(df_deps) > 1e-12:
                eps_eq = eps_eq + (f_corrugated - f_current) / df_deps

            # Bounds
            eps_eq = max(eps_eq, base_roughness_m)
            eps_eq = min(eps_eq, self.inner_diameter_m * 0.05)

        # Store results
        self.roughness_m = eps_eq

        # Store diagnostics
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

        # Print diagnostic information
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

class Chimney:
    """Main chimney simulation class
    
    This is the core class that manages the entire simulation. It:
    1. Stores the chimney configuration (geometry, materials, conditions)
    2. Manages the simulation time stepping
    3. Coordinates the physics calculations (heat transfer, fluid flow)
    4. Stores and provides access to all results
    """
    
    def __init__(self, parts: List[Dict[str, Any]], dt: float = 0.5, total_time: float = 60.0,
                 use_coupled_solver: bool = True, elbows: List[Dict[str, Any]] = None,
                 elbow_method: str = "K_sum", top_boundary: Optional[Dict[str, Any]] = None,
                 inlet_restriction: Optional[Dict[str, Any]] = None,
                 catalytic_bypass: Optional[Dict[str, Any]] = None,
                 door_opening_func: Optional[callable] = None,
                 inlet_temp_csv: Optional[str] = None,
                 inlet_temp_func: Optional[callable] = None,
                 door_temp_dilution: Optional[Dict[str, Any]] = None,
                 room_pressure_offset_Pa: float = 0.0,
                 leakage_config: Optional[Dict[str, Any]] = None,
                 enable_diagnostics: bool = False,
                 momentum_config: Optional[Dict[str, Any]] = None,
                 room_plenum: Optional[Dict[str, Any]] = None,
                 verbose: bool = False):
        """Initialize chimney simulation

        Args:
            parts: List of dictionaries defining chimney parts. Each dict contains
                   properties like 'type', 'inner_diameter_ft', 'height_ft', etc.
            dt: Time step in seconds. Smaller = more accurate but slower.
                For triple walls, may need dt < 0.1s for stability.
            total_time: Total simulation time in seconds. How long to simulate
                        after ignition.
            use_coupled_solver: If True, use iterative coupling between flow and thermal
                               equations for better accuracy. If False, use faster
                               sequential approach.
            elbows: List of dictionaries defining elbows between parts. Each dict contains
                   'between_segments', 'angle_deg', 'R_over_D', 'elbow_type', etc.
            elbow_method: Method for calculating elbow losses: "K_sum" or "equivalent_length"
            inlet_restriction: Optional dict with inlet restriction config:
                   - enable: bool (activate feature)
                   - door_width_ft: float (door width in feet)
                   - door_height_ft: float (door height in feet)
                   - door_cracked_percent: float (% open after loading, enables dynamic door)
                   - door_wide_open_time_s: float (how long door stays open, default 90s)
                   - door_opening_percent: float (0-100, for static mode only)
                   - Cd: float (discharge coefficient, default 0.62)
                   Example: {'enable': True, 'door_width_ft': 1.29, 'door_height_ft': 0.92,
                            'door_cracked_percent': 10}  # 10% cracked after 90s
            catalytic_bypass: Optional dict with catalytic bypass damper config:
                   - enable: bool (activate feature)
                   - bypass_state: 'open', 'closed', or 0-100 float (0=closed, 100=open)
                   - segment_index: int (which segment has the catalyst, 0-indexed)
                   - K_bypass_open: float (K-loss when bypass open, default 0.5)
                   - K_catalyst_closed: float (K-loss through catalyst, default 3.0)
            door_opening_func: (Advanced) Optional callable(t) -> float for custom door
                   schedules. Most users should use door_cracked_percent instead.
            inlet_temp_csv: Optional inlet temperature data from CSV. Can be:
                   - String path: Uses default columns 'PC_Test Time [min]' (min) and 'Flue Temp' (°F)
                   - Dict with custom config:
                     {'path': 'file.csv', 'time_col': 'Time', 'temp_col': 'Temp',
                      'time_unit': 'min' or 's', 'temp_unit': 'F', 'C', or 'K'}
                   Linear interpolation is used between data points.
                   If not provided, uses default exponential ramp-up model.
            inlet_temp_func: Optional callable(t) -> T_K for inlet temperature.
                   Takes time in seconds, returns temperature in Kelvin.
                   This allows directly passing experimental data (e.g., TC1).
                   If both inlet_temp_csv and inlet_temp_func are provided,
                   inlet_temp_func takes precedence.
        """
        self.inlet_temp_csv = inlet_temp_csv
        self.inlet_temp_func = inlet_temp_func
        self.inlet_temp_df = None
        self.inlet_temp_interp = None  # Interpolation function for CSV data
        # Store simulation parameters
        self.dt = dt  # Time step size
        self.total_time = total_time  # How long to simulate
        self.time_steps = int(total_time / dt)  # Number of time steps
        
        # Convert part dictionaries to ChimneyPart objects
        # This validates inputs and converts units
        self.parts = [ChimneyPart(**part) for part in parts]
        
        # Store elbow configuration
        self.elbows = elbows if elbows is not None else []
        self.elbow_method = elbow_method
        
        # Process elbows and create ElbowSpec objects for each segment junction
        self._process_elbows()
        
        # Calculate total number of computational segments
        # Each part is divided into segments for numerical solution
        self.total_segments = sum(part.num_segments for part in self.parts)
        
        # Print sanity check summary of the geometry
        self._print_geometry_summary()
        
        # Check for triple wall parts and warn if time step is large
        # Triple walls have thin metal layers that require small time steps
        # for numerical stability (explicit time integration limit)
        has_triple_wall = any(part.get('type', 'single') == 'triple' for part in parts)
        if has_triple_wall and dt > 1.0:
            print(f"Warning: Large time step ({dt}s) detected with triple wall parts.")
            print("Consider using a smaller time step if numerical instabilities occur.")
            # We don't automatically reduce dt - let user decide based on their needs
        
        # Initialize physics modules
        # Each module handles a specific aspect of the physics
        self.physics = PhysicalProperties()  # Gas properties (density, viscosity, etc.)
        self.comprehensive_ht = ComprehensiveHeatTransfer()  # Advanced heat transfer with all modes

        # Initialize wind cap boundary effect (optional)
        self.top_boundary = top_boundary if top_boundary is not None else {}
        self._wind_cfg = self._parse_wind_cap(self.top_boundary)
        self._deltaP_wind_cap_static = None
        self._last_deltaP_wind_cap = 0.0  # For diagnostics

        # If static mode, compute once and cache
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

            # Print initialization diagnostic
            if self._wind_cfg.enable:
                sign_str = "suction" if self._deltaP_wind_cap_static < 0 else "pressurization"
                effect_str = "aids draft" if self._deltaP_wind_cap_static < 0 else "hurts draft"
                print(f"[WIND CAP] Static mode: V={V0:.1f} m/s, cap={self._wind_cfg.cap_type}, "
                      f"ΔP={self._deltaP_wind_cap_static:+.2f} Pa ({sign_str}, {effect_str})")

        # Initialize inlet restriction
        self._initialize_inlet_restriction(inlet_restriction)

        # Set up door opening function for dynamic door control
        # Priority: explicit door_opening_func > door_cracked_percent > static
        self.door_opening_func = door_opening_func

        # If door_cracked_percent is set, create the built-in door schedule
        if (self.inlet_restriction.enable and
            self.inlet_restriction.door_cracked_percent is not None and
            door_opening_func is None):
            # Create the standard door schedule function
            cracked_pct = self.inlet_restriction.door_cracked_percent
            wide_open_time = self.inlet_restriction.door_wide_open_time_s

            def _door_schedule(t, wide_open_time=wide_open_time, cracked_pct=cracked_pct):
                """Built-in door schedule: wide open, then cracked"""
                if t < wide_open_time:
                    return 100.0  # Wide open during loading
                else:
                    return cracked_pct  # Cracked open for burn

            self.door_opening_func = _door_schedule
            print(f"[DOOR] Dynamic door schedule: 100% open for {wide_open_time:.0f}s, "
                  f"then {cracked_pct:.0f}% cracked")

        elif door_opening_func is not None and self.inlet_restriction.enable:
            print(f"[DOOR] Custom door opening function enabled")

        # Initialize catalytic bypass damper
        self._initialize_catalytic_bypass(catalytic_bypass)

        # Initialize diagnostics recorder (default off)
        self.diagnostics_recorder = DiagnosticsRecorder(enable=enable_diagnostics)

        # Initialize room plenum (default off)
        if room_plenum is not None and isinstance(room_plenum, dict):
            self.room_plenum = RoomPlenum(RoomPlenumConfig(**room_plenum))
        elif room_plenum is not None:
            self.room_plenum = room_plenum
        else:
            self.room_plenum = RoomPlenum(RoomPlenumConfig(enable=False))

        # Build MomentumConfig from dict if provided
        from .coupled_solver import MomentumConfig
        if momentum_config is not None and isinstance(momentum_config, dict):
            self._momentum_config = MomentumConfig(**momentum_config)
        elif momentum_config is not None:
            self._momentum_config = momentum_config
        else:
            self._momentum_config = None

        # Initialize coupled solver if requested
        # Coupled solver iterates between flow and thermal equations for consistency
        self.use_coupled_solver = use_coupled_solver
        if use_coupled_solver:
            self.coupled_solver = CoupledFlowThermalSolver(
                diagnostics=self.diagnostics_recorder,
                momentum_config=self._momentum_config,
            )
            self.convergence_history = []  # Track convergence metrics for debugging
            
        # Initialize early diagnostics
        # This checks for numerical problems in the first few time steps
        self.diagnostics = EarlyTimeDiagnostics(ambient_temp_K=self.parts[0].T_ambient_K)

        # Load inlet temperature from CSV if provided
        if inlet_temp_csv is not None:
            self._load_inlet_temp_csv(inlet_temp_csv)

        # Initialize door temperature dilution (Jan 27 feature)
        # When door is open, cold ambient air mixes with hot combustion gas
        self.door_temp_dilution_enable = False
        self.door_temp_dilution_strength = 0.8  # Default: 80% ambient at 100% door open
        if door_temp_dilution is not None:
            self.door_temp_dilution_enable = door_temp_dilution.get('enable', False)
            self.door_temp_dilution_strength = door_temp_dilution.get('strength', 0.8)
            if self.door_temp_dilution_enable:
                print(f"[DOOR DILUTION] Enabled with strength={self.door_temp_dilution_strength:.2f}")

        # Initialize leakage configuration (pipe joint air infiltration)
        self._initialize_leakage_config(leakage_config)

        # Update coupled solver momentum config based on leakage settings
        if self.use_coupled_solver and hasattr(self, 'leakage_config') and self.leakage_config.enable:
            self.coupled_solver.momentum_config.use_distributed_leakage = True

        # Store verbose flag and room pressure offset
        self.verbose = verbose
        self.room_pressure_offset_Pa = room_pressure_offset_Pa

        # Pre-compute segment lengths to avoid repeated division (performance optimization)
        self.segment_lengths = np.zeros(self.total_segments)
        seg_idx = 0
        for part in self.parts:
            seg_length = part.height_m / part.num_segments
            for i in range(part.num_segments):
                self.segment_lengths[seg_idx] = seg_length
                seg_idx += 1
        
        # Initialize all the arrays to store results
        # This allocates memory for temperature, flow, pressure histories
        self._initialize_arrays()
        
        # Simulation state tracking
        self.current_time = 0.0  # Current simulation time in seconds
        self.time_step = 0  # Current time step index

        # Backdraft detection and tracking
        self.backdraft_detected = False  # True if backdraft has occurred at any point
        self.backdraft_start_time = None  # Time when backdraft first started (seconds)
        self.backdraft_end_time = None  # Time when normal draft was restored (seconds)
        self.flow_reversal_count = 0  # Number of times flow has reversed direction
    #Done
    def run_simulation(self, save_interval: int = 1):
        """Run the complete simulation using comprehensive heat transfer model
        
        This is the main simulation loop that advances through time, calculating
        temperatures, flow rates, and pressures at each time step.
        
        Args:
            save_interval: Save results every N time steps (not currently used)
        """
        print(f"Starting comprehensive heat transfer simulation:")
        print(f"  {self.total_segments} segments, {self.time_steps} time steps")
        print(f"  Time step: {self.dt}s, Total time: {self.total_time}s")
        
        # MAIN TIME LOOP - advance from t=dt to t=total_time
        for step in range(1, self.time_steps + 1):
            # Update simulation state
            self.time_step = step
            self.current_time = step * self.dt

            # STEP 0: UPDATE DOOR OPENING (if dynamic door control enabled)
            # This updates effective area before physics calculations
            if self.door_opening_func is not None and self.inlet_restriction.enable:
                new_opening = self.door_opening_func(self.current_time)
                old_opening = self.door_opening_history[step - 1]

                # Detect significant change in door opening (threshold: 5%)
                if abs(new_opening - old_opening) > 5.0:
                    direction = "closing" if new_opening < old_opening else "opening"
                    print(f"  [DOOR] t={self.current_time:.1f}s: Door {direction} "
                          f"{old_opening:.0f}% -> {new_opening:.0f}%")

                self._update_door_opening(new_opening)
                self.door_opening_history[step] = new_opening
            else:
                # Static door - copy previous value
                self.door_opening_history[step] = self.door_opening_history[step - 1]

            # STEP 0b: UPDATE ROOM PLENUM PRESSURE (if enabled)
            # Uses previous-timestep chimney mass flow to compute new room pressure
            if self.room_plenum.cfg.enable:
                mdot_prev = self.mass_flow_rate[step - 1]
                Pa_ambient = self.parts[0].Pa
                dp_room = self.room_plenum.update(mdot_prev, Pa_ambient, self.dt)
                self.room_pressure_offset_Pa = dp_room

            # STEP 1: CALCULATE INLET TEMPERATURE
            # This is the temperature of gas entering the bottom of the chimney
            # Uses CSV data if provided, otherwise default exponential ramp-up
            T_ambient_inlet = self.parts[0].T_ambient_K
            T_inlet = self._get_inlet_temp(self.current_time, T_ambient=T_ambient_inlet)

            # STEP 2: SOLVE PHYSICS EQUATIONS
            if self.use_coupled_solver:
                # COUPLED APPROACH: Iterate between flow and thermal equations
                # This ensures consistency between buoyancy (depends on T) and
                # heat transfer (depends on flow). More accurate but slower.
                metrics = self.coupled_solver.solve_coupled_timestep(self, step, T_inlet, self.dt)
                # Don't store convergence history to save memory (uncomment if needed for debugging)
                # self.convergence_history.append(metrics)
                
                # Warn if iteration didn't converge (but continue anyway) - disabled for speed
                # if step % max(1, self.time_steps // 20) == 0 and not metrics.converged:
                #     print(f"  Warning: Coupling did not converge at t={self.current_time:.1f}s")
            else:
                # SEQUENTIAL APPROACH: First heat transfer, then flow
                # Faster but less accurate since T and flow aren't fully consistent
                
                # Solve heat transfer for all segments with current flow
                self._solve_comprehensive_heat_transfer(step, T_inlet)
                
                # Update flow based on new temperatures (buoyancy)
                self._update_flow_and_pressure(step)

            # BACKDRAFT DETECTION: Track flow direction changes
            # This monitors for flow reversal (backdraft) events
            if step > 1:  # Need at least 2 timesteps to detect change
                current_flow = self.mass_flow_rate[step]
                previous_flow = self.mass_flow_rate[step - 1]

                # Detect transition from normal draft to backdraft
                if previous_flow >= 0 and current_flow < 0:
                    self.backdraft_detected = True
                    self.backdraft_start_time = self.current_time
                    self.flow_reversal_count += 1
                    print(f"\n{'='*60}")
                    print(f"[BACKDRAFT DETECTED] Flow reversal at t={self.current_time:.1f}s")
                    print(f"  Mass flow: {previous_flow:.4f} -> {current_flow:.4f} kg/s")
                    print(f"  Flow direction: UPWARD -> DOWNWARD (backdraft)")
                    print(f"{'='*60}\n")

                # Detect transition from backdraft to normal draft (recovery)
                elif previous_flow < 0 and current_flow >= 0:
                    self.backdraft_end_time = self.current_time
                    duration = self.backdraft_end_time - (self.backdraft_start_time or 0)
                    print(f"\n{'='*60}")
                    print(f"[DRAFT RESTORED] Normal flow resumed at t={self.current_time:.1f}s")
                    print(f"  Mass flow: {previous_flow:.4f} -> {current_flow:.4f} kg/s")
                    print(f"  Flow direction: DOWNWARD -> UPWARD (normal draft)")
                    print(f"  Backdraft duration: {duration:.1f}s")
                    print(f"{'='*60}\n")

            # STEP 3: DIAGNOSTICS AND STABILITY CHECKS
            # Check for unphysical behavior in early timesteps when transients are strong
            if step <= 30:  # First 30 timesteps are most prone to instability
                # Prepare temperature arrays based on what exists
                T_liner = self.T_liner[step] if hasattr(self, 'T_liner') else None
                T_middle = self.T_middle[step] if hasattr(self, 'T_middle') else None
                T_outer = self.T_outer[step] if hasattr(self, 'T_outer') else None
                
                # Run diagnostic checks for numerical problems
                self.diagnostics.check_timestep(
                    step, self.current_time, 
                    self.T_gas[step], self.T_wall[step],
                    T_liner, T_middle, T_outer,
                    self.mass_flow_rate[step], self.dt
                )
            
            # STEP 4: PROGRESS REPORTING
            # Print status every 5% of simulation to show it's running
            if step % max(1, self.time_steps // 20) == 0:
                progress = 100 * step / self.time_steps
                gas_temp_range = (self.T_gas[step].min() - 273.15, self.T_gas[step].max() - 273.15)
                wall_temp_range = (self.T_wall[step].min() - 273.15, self.T_wall[step].max() - 273.15)
                print(f"  {progress:5.1f}% | t={self.current_time:6.1f}s | Gas: {gas_temp_range[0]:5.1f}-{gas_temp_range[1]:5.1f}°C | Wall: {wall_temp_range[0]:5.1f}-{wall_temp_range[1]:5.1f}°C")

                # Wind cap diagnostic (if enabled and in dynamic mode)
                if self._wind_cfg.enable and self._wind_cfg.update_mode == "dynamic":
                    dP = self._last_deltaP_wind_cap
                    V = self._wind_cfg.wind_speed_m_s
                    if isinstance(V, (list, tuple)):
                        V_current = V[min(step, len(V) - 1)]
                    else:
                        V_current = V
                    sign_str = "suction" if dP < 0 else "pressurization"
                    print(f"  [WIND CAP] V={V_current:.1f} m/s, cap={self._wind_cfg.cap_type}, dP={dP:+.2f} Pa ({sign_str})")
                
        print("Comprehensive simulation completed successfully")
        
        # Run conservation diagnostics
        self.check_conservation()
    #Done
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
        """
        Models the inlet gas temperature entering the chimney as a function of time.
        This is the temperature of the flue gas entering the bottom of the chimney,
        NOT the fire temperature itself.
        
        Uses an exponential rise model that simulates:
        - Cold start at ambient temperature
        - Gradual heating as fire establishes
        - Asymptotic approach to maximum temperature
        
        Parameters:
            t (float or array): Time in seconds since ignition
            T_ambient (float): Ambient temperature in Kelvin (default is 20°C = 293.15 K)
            T_max (float): Maximum inlet gas temperature in Kelvin (default is 500°C = 773.15 K)
            tau (float): Characteristic time constant in seconds. Controls how fast it heats up.
                        Larger tau = slower heating. tau=300s means 63% of max temp at 5 min.
        
        Returns:
            float or array: Inlet gas temperature in Kelvin at time t
        """
        # Exponential rise: starts at T_ambient, approaches T_max
        # At t=tau, temperature is 63% of the way from ambient to max
        # At t=3*tau, temperature is 95% of the way to max
        return T_ambient + (T_max - T_ambient) * (1 - np.exp(-t / tau))
    def _default_inlet_temp_fire(self, t_sec: float) -> float:
        """Calculate inlet gas temperature based on standard fire curve.
        
        This uses the ISO 834 standard fire curve (common in fire testing)
        to model how fire temperature evolves. The flue gas temperature
        is empirically scaled to be about 50% of the fire temperature.
        
        Args:
            t_sec: Time in seconds since ignition
            
        Returns:
            Temperature in Kelvin
        """
        # ISO 834 fire curve: T = 20 + 345 * log10(8t + 1) where t is in minutes
        # We add 1.5s to avoid log(0) at t=0
        T_fire_C = 20 + 345 * np.log10(8 * ((t_sec + 1.5) / 60) + 1)
        
        # Flue gases are cooler than fire due to heat loss and air dilution
        T_flue_C = 0.5 * T_fire_C   # Empirical scaling factor
        
        # Convert to Kelvin and cap at reasonable maximum
        T_flue_K = T_flue_C + 273.15
        return min(T_flue_K, 800)    # Cap at 800K (527°C) for safety
    #Done
    def _solve_comprehensive_heat_transfer(self, step: int, T_inlet: float):
        """Solve comprehensive heat transfer for all segments
        
        This method calculates heat transfer for each segment of the chimney,
        updating gas and wall temperatures based on convection, conduction,
        and radiation.
        
        Args:
            step: Current time step index
            T_inlet: Temperature of gas entering bottom of chimney (K)
        """
        
        # Set inlet boundary condition - hot gas enters at bottom
        self.T_gas[step, 0] = T_inlet
        
        # SOLVE EACH SEGMENT FROM BOTTOM TO TOP
        # Gas flows upward, so we solve in flow direction
        for seg in range(self.total_segments):
            # Get info about which part this segment belongs to
            part_idx, local_seg, part = self.get_segment_info(seg)
            
            # Length of this computational segment (use pre-computed value)
            segment_length = self.segment_lengths[seg]
            
            # INLET CONDITION FOR THIS SEGMENT
            if seg > 0:
                # Gas enters this segment from the previous segment
                # Temperature is continuous (no sudden jumps)
                self.T_gas[step, seg] = self.T_gas[step, seg-1]
                
                # Debug output disabled for performance
                # if seg == 5 and step == self.time_steps:
                #     print(f"DEBUG Gas flow from single to triple wall:")
                #     print(f"  Single wall final segment gas temp: {self.T_gas[step, seg-1] - 273.15:.1f}°C")
                #     print(f"  Triple wall first segment gas temp: {self.T_gas[step, seg] - 273.15:.1f}°C")
                
            # Estimate velocity for this segment
            if step == 1:
                velocity = 1.0  # Initial guess
            else:
                velocity = self.velocity[step-1, seg]
                
            # PREPARE INPUT DATA FOR HEAT TRANSFER CALCULATION
            # Package all the information needed to compute heat transfer
            segment_data = {
                'gas_temperature': self.T_gas[step, seg],      # Current gas temp in segment
                'ambient_temperature': part.T_ambient_K,        # Outside air temperature
                'pressure': part.Pa,                            # Gas pressure (affects properties)
                'velocity': velocity,                           # Gas velocity (affects convection)
                'diameter': part.inner_diameter_m,              # Flow area diameter
                'length': segment_length,                       # Length of this segment
                'orientation': part.orientation,                # Vertical/horizontal (affects natural conv)
                'has_fittings': part.has_fittings,            # Elbows/tees increase turbulence
                'humidity_ratio': part.humidity_ratio,          # For condensation calculations
                'wall_type': part.type,                        # 'single' or 'triple'
                'segment_id': seg,                             # For debugging
                'mass_flow_rate': self.mass_flow_rate[step-1] if step > 0 else 0.01  # Flow rate
            }
            
            # Add elbow specifications if present at this segment junction
            elbow_key = (part_idx, local_seg)
            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                segment_data['elbows'] = self.segment_elbows[elbow_key]
                segment_data['elbow_method'] = self.elbow_method

            # Add corrugation information and heat transfer multiplier (v2.0)
            if hasattr(part, 'corrugation') and part.corrugation is not None:
                segment_data['corrugation'] = part.corrugation
                if hasattr(part, 'corrugation_effects') and 'Mh' in part.corrugation_effects:
                    segment_data['Mh'] = part.corrugation_effects['Mh']

            # Add wall-specific data based on configuration
            if part.type == 'single':
                # Get previous wall temperature 
                prev_T_wall = self.T_wall[step-1, seg]
                # Ensure temperature is physical (check for uninitialized arrays)
                if prev_T_wall <= 0:  # Uninitialized or absolute zero
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
                # Get previous temperatures 
                prev_T_liner = self.T_liner[step-1, seg]
                prev_T_middle = self.T_middle[step-1, seg] 
                prev_T_outer = self.T_outer[step-1, seg]
                
                # Ensure temperatures are physical (check for uninitialized arrays)
                if prev_T_liner <= 0:  # Uninitialized or absolute zero
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
            
            # Calculate comprehensive heat transfer
            results = self.comprehensive_ht.calculate_segment_heat_transfer(segment_data, self.dt)
            
            # Debug heat transfer coefficients and temperatures
            # Debug disabled for performance
            # if step == self.time_steps:
            #     print(f"DEBUG Segment {seg+1} (Part {part_idx+1} {part.type.title()} Wall):")
            
            # Validate results
            if 'wall_temperature_new' not in results and 'liner_temperature_new' not in results:
                print(f"ERROR: No temperature results at step={step}, seg={seg}")
                print(f"Results keys: {list(results.keys())}")
                # Use ambient as fallback
                results['wall_temperature_new'] = part.T_ambient_K
                results['liner_temperature_new'] = part.T_ambient_K
                results['middle_temperature_new'] = part.T_ambient_K
                results['outer_temperature_new'] = part.T_ambient_K
            
            # Update temperatures and properties based on wall type
            if part.type == 'single':
                # Ensure result is physical
                T_wall_new = results['wall_temperature_new']
                if T_wall_new <= 0:
                    print(f"ERROR: Computed wall temp {T_wall_new:.1f}K is unphysical, using ambient")
                    T_wall_new = part.T_ambient_K
                    
                self.T_wall[step, seg] = T_wall_new
                self.T_outer[step, seg] = T_wall_new  # Single wall
                # Set triple wall arrays for consistency
                self.T_liner[step, seg] = T_wall_new
                self.T_middle[step, seg] = T_wall_new
            elif part.type == 'triple':
                # Ensure results are physical
                T_liner_new = results['liner_temperature_new']
                T_middle_new = results['middle_temperature_new']
                T_outer_new = results['outer_temperature_new']
                
                if T_liner_new <= 0:
                    print(f"ERROR: Computed liner temp {T_liner_new:.1f}K is unphysical, using ambient")
                    T_liner_new = part.T_ambient_K
                    T_middle_new = part.T_ambient_K
                    T_outer_new = part.T_ambient_K
                    
                # Debug: show what we're storing
                # Debug disabled for performance
                # if step <= 3 and seg <= 2:
                #     print(f"DEBUG: Step {step}, seg {seg} - storing new temps: liner={T_liner_new-273.15:.1f}°C, middle={T_middle_new-273.15:.1f}°C, outer={T_outer_new-273.15:.1f}°C")
                    
                self.T_liner[step, seg] = T_liner_new
                self.T_middle[step, seg] = T_middle_new
                self.T_outer[step, seg] = T_outer_new
                # Set wall temperature to liner for compatibility
                self.T_wall[step, seg] = T_liner_new
            
            # Update gas temperature based on heat loss to walls
            # Calculate heat loss rate from gas
            q_loss = results['h_inner'] * (np.pi * part.inner_diameter_m * segment_length) * (self.T_gas[step, seg] - self.T_wall[step, seg])
            
            # Gas properties for heat capacity
            gas_props = self.comprehensive_ht.gas_properties(self.T_gas[step, seg], part.Pa)
            gas_volume = np.pi * (part.inner_diameter_m/2)**2 * segment_length
            gas_mass = gas_props.density * gas_volume
            
            # Calculate gas temperature change including advection
            # NOTE: Only update gas temperature here if NOT using coupled solver
            # The coupled solver handles gas temperature updates itself
            if gas_mass > 0 and not self.use_coupled_solver and seg > 0:
                # For segments after the first, include advection from upstream
                mass_flow = self.mass_flow_rate[step-1] if step > 0 else 0.0
                
                # CRITICAL PHYSICS: If no flow, gas temperature approaches wall temperature
                # which approaches ambient. Gas CANNOT cool below ambient without flow!
                if mass_flow > 0:
                    T_upstream = self.T_gas[step, seg-1]
                    
                    # Energy balance: accumulation = advection_in - advection_out - heat_loss
                    # mass * cp * dT/dt = mdot * cp * (T_in - T_out) - q_loss
                    q_advection_net = mass_flow * gas_props.specific_heat * (T_upstream - self.T_gas[step, seg])
                    dT_gas = (q_advection_net - q_loss) * self.dt / (gas_mass * gas_props.specific_heat)
                    
                    # CRITICAL: Limit cooling to available thermal energy
                    # Gas cannot cool below ambient without external work
                    if dT_gas < 0:
                        max_cooling = self.T_gas[step, seg] - part.T_ambient_K
                        if abs(dT_gas) > max_cooling:
                            dT_gas = -max_cooling
                    
                    self.T_gas[step, seg] = self.T_gas[step, seg] + dT_gas
                else:
                    # No flow: gas temperature approaches wall temperature
                    # Wall temperature approaches ambient (see wall heat transfer)
                    # This prevents gas from cooling below ambient
                    tau_gas_wall = gas_mass * gas_props.specific_heat / (results['h_inner'] * np.pi * part.inner_diameter_m * segment_length)
                    self.T_gas[step, seg] = self.T_gas[step, seg] + (self.T_wall[step, seg] - self.T_gas[step, seg]) * self.dt / tau_gas_wall
            
            # Store heat transfer coefficients
            self.h_conv_inner[step, seg] = results['h_inner']
            self.h_conv_outer[step, seg] = results['h_outer']
            
            # Update density based on new gas temperature
            self.density[step, seg] = self.comprehensive_ht.gas_properties(
                self.T_gas[step, seg], part.Pa
            ).density
            
            # Apply physical temperature bounds - nothing below ambient
            # This is fundamental thermodynamics: no refrigeration in system
            self.T_gas[step, seg] = max(self.T_gas[step, seg], part.T_ambient_K)
            self.T_wall[step, seg] = max(self.T_wall[step, seg], part.T_ambient_K)
            if part.type == 'triple':
                self.T_liner[step, seg] = max(self.T_liner[step, seg], part.T_ambient_K)
                self.T_middle[step, seg] = max(self.T_middle[step, seg], part.T_ambient_K)
                self.T_outer[step, seg] = max(self.T_outer[step, seg], part.T_ambient_K)
    #Done
    def _update_flow_and_pressure(self, step: int):
        """Update mass flow rate, velocity, and pressure using buoyancy and pressure losses
        
        This implements the momentum equation for the chimney:
        - Buoyancy forces (from density differences) drive the flow upward
        - Friction and fitting losses oppose the flow
        - Net pressure determines the flow rate
        
        Args:
            step: Current time step index
        """
        
        # STEP 1: CALCULATE DRIVING FORCES (BUOYANCY)
        # Hot gas is less dense than cold ambient air, creating upward force
        total_buoyancy = 0.0  # Total buoyancy pressure driving flow (Pa)
        total_pressure_loss = 0.0  # Total friction losses resisting flow (Pa)
        
        # Sum contributions from each segment
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            
            # Density of hot gas in chimney
            rho_gas = self.density[step, seg]
            
            # Density of cold ambient air outside
            rho_ambient = self.comprehensive_ht.gas_properties(part.T_ambient_K, part.Pa).density
            
            # Buoyancy pressure: dp = g * height * (rho_ambient - rho_gas)
            # Positive when gas is hotter (less dense) than ambient
            dp_buoyancy_seg = self.comprehensive_ht.buoyancy_pressure(
                segment_length, rho_ambient, rho_gas
            )
            total_buoyancy += dp_buoyancy_seg
            
            # STEP 2: CALCULATE RESISTANCE FORCES (FRICTION)
            # Get velocity from previous time step (or initial guess)
            if step > 1:
                velocity = self.velocity[step-1, seg]
            else:
                velocity = 1.0  # Initial guess for first time step
                
            # Get gas properties at current temperature
            gas_props = self.comprehensive_ht.gas_properties(self.T_gas[step, seg], part.Pa)
            gas_props.velocity = velocity
            
            # Look up actual elbows for this segment
            elbow_key = (part_idx, local_seg)
            seg_elbows = None
            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                seg_elbows = self.segment_elbows[elbow_key]

            # Calculate pressure drop from friction (Darcy-Weisbach equation)
            dp_loss_seg = self.comprehensive_ht.pressure_drop(
                gas_props, part.inner_diameter_m, segment_length,
                velocity, part.roughness_m, elbows=seg_elbows
            )
            total_pressure_loss += dp_loss_seg
        
        # STEP 3: SOLVE FOR MASS FLOW RATE
        # Balance equation: Buoyancy pressure - Friction losses - Room depressurization = Net pressure
        # Room depressurization reduces available draft and can cause backdraft
        net_pressure = total_buoyancy - total_pressure_loss - self.room_pressure_offset_Pa

        # Calculate mass flow from net pressure (allows backdraft)
        avg_density = np.mean(self.density[step, :])  # Average gas density
        flow_coefficient = 1.0  # Simplified - could include discharge coefficient
        avg_area = np.mean([part.inner_area_m2 for part in self.parts])

        # Use Bernoulli equation: dp = K * rho * v^2 / 2
        # Solving for velocity: v = sqrt(2 * |dp| / (K * rho))
        velocity_avg = np.sqrt(2 * abs(net_pressure) / (flow_coefficient * avg_density + 1e-10))

        # Mass flow rate = density * velocity * area
        # Sign matches net_pressure: positive = upward draft, negative = backdraft
        if net_pressure >= 0:
            self.mass_flow_rate[step] = avg_density * velocity_avg * avg_area
        else:
            # BACKDRAFT: Negative net pressure drives downward flow
            # This happens when room depressurization exceeds available draft
            self.mass_flow_rate[step] = -avg_density * velocity_avg * avg_area
        
        # STEP 4: UPDATE VELOCITIES IN EACH SEGMENT
        # Velocity varies with diameter and density changes
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            
            # Velocity from continuity equation
            self.velocity[step, seg] = self.mass_flow_rate[step] / (
                self.density[step, seg] * part.inner_area_m2
            )
            
            # Pressure profile: integrate buoyancy minus friction per segment
            if seg == 0:
                self.pressure[step, seg] = part.Pa  # Reference pressure
            else:
                segment_length = part.height_m / part.num_segments
                rho_gas = self.density[step, seg]
                rho_ambient = self.comprehensive_ht.gas_properties(part.T_ambient_K, part.Pa).density
                dp_buoyancy = 9.81 * segment_length * (rho_ambient - rho_gas)

                # Friction loss for this segment
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
    #Done
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
    #Done
    def check_conservation(self):
        """Check mass and energy conservation throughout the simulation"""
        print("\n" + "="*80)
        print("CONSERVATION DIAGNOSTICS")
        print("="*80)
        
        # Energy conservation check
        self._check_energy_conservation()
        
        # Mass conservation check (if condensation is present)
        self._check_mass_conservation()
    #Done
    def _check_energy_conservation(self):
        """Check energy conservation by comparing input, stored, and lost energy"""
        print("\nEnergy Conservation Check:")
        
        # Calculate total energy input at inlet
        energy_input = 0.0
        for step in range(1, self.time_steps + 1):
            T_inlet = self.T_gas[step, 0]
            mdot = self.mass_flow_rate[step]
            cp = self.physics.air_specific_heat(T_inlet)
            T_ref = self.parts[0].T_ambient_K
            energy_input += mdot * cp * (T_inlet - T_ref) * self.dt
        
        # Calculate energy stored in gas and walls
        energy_stored_gas = 0.0
        energy_stored_wall = 0.0
        
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            
            # Gas energy
            T_gas_final = self.T_gas[-1, seg]
            T_gas_initial = self.T_gas[0, seg]
            gas_volume = part.inner_area_m2 * segment_length
            rho_final = self.density[-1, seg]
            cp_gas = self.physics.air_specific_heat(T_gas_final)
            energy_stored_gas += rho_final * gas_volume * cp_gas * (T_gas_final - T_gas_initial)
            
            # Wall energy
            if part.type == 'single':
                T_wall_final = self.T_wall[-1, seg]
                T_wall_initial = self.T_wall[0, seg]
                wall_volume = np.pi * segment_length * ((part.outer_diameter_m)**2 - (part.inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_w_kg_m3 * wall_volume * part.cp_w_J_kg_K * (T_wall_final - T_wall_initial)
            else:  # triple wall
                # Liner
                T_liner_final = self.T_liner[-1, seg]
                T_liner_initial = self.T_liner[0, seg]
                V_liner = np.pi * segment_length * ((part.liner_outer_diameter_m)**2 - (part.inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_inner_kg_m3 * V_liner * part.cp_inner_J_kg_K * (T_liner_final - T_liner_initial)
                
                # Middle
                T_middle_final = self.T_middle[-1, seg]
                T_middle_initial = self.T_middle[0, seg]
                V_middle = np.pi * segment_length * ((part.middle_outer_diameter_m)**2 - (part.middle_inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_middle_kg_m3 * V_middle * part.cp_middle_J_kg_K * (T_middle_final - T_middle_initial)
                
                # Outer
                T_outer_final = self.T_outer[-1, seg]
                T_outer_initial = self.T_outer[0, seg]
                V_outer = np.pi * segment_length * ((part.outer_diameter_m)**2 - (part.outer_inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_outer_kg_m3 * V_outer * part.cp_outer_J_kg_K * (T_outer_final - T_outer_initial)
        
        # Estimate energy lost to ambient (simplified)
        energy_lost = energy_input - energy_stored_gas - energy_stored_wall
        
        # Calculate percentages
        if energy_input > 0:
            pct_stored_gas = 100 * energy_stored_gas / energy_input
            pct_stored_wall = 100 * energy_stored_wall / energy_input
            pct_lost = 100 * energy_lost / energy_input
            
            print(f"  Total energy input: {energy_input/1e6:.2f} MJ")
            print(f"  Energy stored in gas: {energy_stored_gas/1e6:.2f} MJ ({pct_stored_gas:.1f}%)")
            print(f"  Energy stored in walls: {energy_stored_wall/1e6:.2f} MJ ({pct_stored_wall:.1f}%)")
            print(f"  Energy lost to ambient: {energy_lost/1e6:.2f} MJ ({pct_lost:.1f}%)")
            print(f"  Energy balance closure: {100 - abs(pct_stored_gas + pct_stored_wall + pct_lost - 100):.1f}%")
    #Done
    def _check_mass_conservation(self):
        """Check water vapor mass conservation with condensation"""
        print("\nMass Conservation Check (Water Vapor):")
        
        # Check if we have condensation tracking
        if not hasattr(self.comprehensive_ht, 'condensation_states') or not self.comprehensive_ht.condensation_states:
            print("  No condensation tracking available")
            return
        
        # Calculate total condensed mass
        total_condensed = 0.0
        for seg_id, state in self.comprehensive_ht.condensation_states.items():
            total_condensed += state.film_mass
        
        print(f"  Total water condensed: {total_condensed*1000:.3f} g")
        
        # Estimate initial vs final water vapor in gas
        initial_water = 0.0
        final_water = 0.0
        
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            gas_volume = part.inner_area_m2 * segment_length
            
            # Initial
            rho_initial = self.density[0, seg]
            humidity_initial = part.humidity_ratio
            dry_air_mass_initial = gas_volume * rho_initial / (1 + humidity_initial)
            initial_water += dry_air_mass_initial * humidity_initial
            
            # Final (simplified - would need to track humidity ratio evolution)
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
        
        # Create a mapping of segment boundaries to elbows
        self.segment_elbows = {}  # key: (part_idx, local_seg_idx), value: List[ElbowSpec]
        
        # Validate geometry consistency
        n_parts = len(self.parts)
        # Maximum fittings = n_parts - 1 (between parts) + 1 (inlet fitting before part 0)
        max_fittings = n_parts  # Allow inlet fitting + all junctions between parts

        if len(self.elbows) > max_fittings:
            raise ValueError(
                f"Geometry error: {len(self.elbows)} fittings specified but only "
                f"{max_fittings} locations exist ({n_parts} parts allow 1 inlet + {n_parts-1} junctions). "
                f"Each fitting must be at inlet (-1,0) or between two adjacent parts."
            )
        
        # Track which junctions have been used
        used_junctions = set()
        
        for i, elbow_dict in enumerate(self.elbows):
            # Extract elbow location - accept both 'between_segments' and 'between_parts'
            between = elbow_dict.get('between_segments', elbow_dict.get('between_parts', []))
            if len(between) != 2:
                raise ValueError(
                    f"Elbow {i} missing or invalid 'between_segments'/'between_parts' specification. "
                    f"Expected [from_part, to_part], got {between}"
                )
                
            part_from, part_to = between[0], between[1]

            # Handle special case: part_from == -1 means inlet fitting (before part 0)
            is_inlet_fitting = (part_from == -1 and part_to == 0)

            # Validate part indices
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

            # Check for adjacent parts (skip for inlet fitting)
            if not is_inlet_fitting and abs(part_to - part_from) != 1:
                raise ValueError(
                    f"Elbow {i}: Parts {part_from} and {part_to} are not adjacent. "
                    f"Elbows can only connect adjacent parts."
                )

            # Check for duplicate junction specification
            junction = (min(part_from, part_to), max(part_from, part_to))
            if junction in used_junctions:
                raise ValueError(
                    f"Elbow {i}: Junction between parts {part_from} and {part_to} "
                    f"already has an elbow defined. Each junction can have only one elbow."
                )
            used_junctions.add(junction)

            # Validate geometric compatibility (skip for inlet fitting)
            if not is_inlet_fitting and part_from >= 0 and part_from < n_parts and part_to < n_parts:
                # Check if orientations make sense for an elbow
                orient_from = self.parts[part_from].orientation
                orient_to = self.parts[part_to].orientation

                # Warn about unusual configurations
                if orient_from == orient_to == "horizontal":
                    print(f"Warning: Elbow {i} connects two horizontal sections. "
                          f"Ensure the angle_deg reflects the actual geometry.")

                # Check diameter compatibility
                diam_from = self.parts[part_from].inner_diameter_m
                diam_to = self.parts[part_to].inner_diameter_m

                if abs(diam_from - diam_to) / min(diam_from, diam_to) > 0.1:
                    print(f"Warning: Elbow {i} connects parts with >10% diameter difference "
                          f"({diam_from*1000:.1f}mm to {diam_to*1000:.1f}mm). "
                          f"Consider adding a reducer/expander.")

            # Get diameter from the appropriate part
            if is_inlet_fitting:
                diameter = self.parts[0].inner_diameter_m  # Use first part for inlet
            else:
                diameter = self.parts[part_from].inner_diameter_m

            # Handle tee special case: auto-set to 90° if not specified
            # Accept both 'elbow_type' and 'type' keys for compatibility
            elbow_type = elbow_dict.get('elbow_type', elbow_dict.get('type', 'long_radius'))
            if elbow_type == 'tee':
                # Tees are treated as 90° branch flow
                angle_deg = elbow_dict.get('angle_deg', 90)
                R_over_D = 1.0  # Fixed for tees
            else:
                angle_deg = elbow_dict.get('angle_deg', 90)
                R_over_D = elbow_dict.get('R_over_D', 1.5)

            # Create ElbowSpec object
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
            
            # Store the elbow at the junction between parts
            # For inlet fitting (part_from == -1), store at first segment of part 0
            # Otherwise, associate with the last segment of the upstream part
            if is_inlet_fitting:
                key = (0, 0)  # First segment of first part
            else:
                last_seg_idx = self.parts[part_from].num_segments - 1
                key = (part_from, last_seg_idx)

            if key not in self.segment_elbows:
                self.segment_elbows[key] = []
            self.segment_elbows[key].append(elbow_spec)
            
            # Print confirmation (removed - will print summary instead)
    
    def _print_geometry_summary(self):
        """Print a sanity check summary of the chimney geometry."""
        print("\n" + "="*70)
        print("CHIMNEY GEOMETRY SUMMARY (Sanity Check)")
        print("="*70)
        
        # Build a list of all components in order
        components = []
        total_height = 0.0
        total_length = 0.0
        
        # Convert feet/inches for display if values suggest imperial units
        def format_length(meters):
            """Format length with both metric and imperial units."""
            feet = meters / 0.3048
            inches = meters / 0.0254
            if feet > 0.5:  # Show feet for longer sections
                return f"{meters:.2f}m ({feet:.1f}ft)"
            else:  # Show inches for short sections
                return f"{meters*1000:.0f}mm ({inches:.1f}in)"
        
        def format_diameter(meters):
            """Format diameter with both metric and imperial units."""
            mm = meters * 1000
            inches = meters / 0.0254
            return f"{mm:.1f}mm ({inches:.1f}in)"
        
        # Check for inlet fitting at (0, 0) - before Part 0
        if hasattr(self, 'segment_elbows') and (0, 0) in self.segment_elbows:
            for elbow in self.segment_elbows[(0, 0)]:
                elbow_desc = f"INLET: {elbow.angle_deg:.0f}° {elbow.elbow_type.replace('_', '-')}"
                elbow_desc += f" (R/D={elbow.R_over_D:.1f})"
                components.append(elbow_desc)

        # Process each part and its associated elbows
        for i, part in enumerate(self.parts):
            # Add the part info
            length_str = format_length(part.height_m)
            diam_str = format_diameter(part.inner_diameter_m)

            part_desc = f"Part {i}: {part.type.upper()} {part.orientation}"
            part_desc += f", L={length_str}, D={diam_str}"

            if part.type == 'triple':
                part_desc += " (triple-wall)"

            components.append(part_desc)

            # Track total dimensions
            if part.orientation == 'vertical':
                total_height += part.height_m
            total_length += part.height_m

            # Check for elbow after this part (at last segment)
            elbow_key = (i, part.num_segments - 1)
            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                for elbow in self.segment_elbows[elbow_key]:
                    elbow_desc = f"       +-- FITTING: {elbow.angle_deg:.0f}°"
                    elbow_desc += f" {elbow.elbow_type.replace('_', '-')}"
                    elbow_desc += f" (R/D={elbow.R_over_D:.1f})"
                    if elbow.vane_factor < 1.0:
                        elbow_desc += f" with vanes (factor={elbow.vane_factor:.1f})"
                    components.append(elbow_desc)
        
        # Print the component list
        print("\nFlow path from inlet to outlet:")
        print("-" * 40)
        for comp in components:
            print(comp)
        
        # Print summary statistics
        # Count registered fittings (elbows/tees) from segment_elbows
        total_fittings = sum(len(v) for v in self.segment_elbows.values()) if hasattr(self, 'segment_elbows') else 0
        print("\n" + "-" * 40)
        print("Summary:")
        print(f"  Total parts: {len(self.parts)}")
        print(f"  Total fittings (elbows/tees): {total_fittings}")
        print(f"  Total path length: {format_length(total_length)}")
        print(f"  Total vertical rise: {format_length(total_height)}")
        print(f"  Computational segments: {self.total_segments}")
        print(f"  Elbow loss method: {self.elbow_method}")
        
        # Check for common issues
        warnings = []
        
        # Check if all parts have same diameter
        diameters = [p.inner_diameter_m for p in self.parts]
        if len(set(diameters)) > 1:
            warnings.append("Parts have different diameters - check for proper transitions")
        
        # Check for reasonable number of segments
        avg_segments = self.total_segments / len(self.parts)
        if avg_segments < 2:
            warnings.append("Low segment count - consider increasing for better accuracy")
        elif avg_segments > 20:
            warnings.append("High segment count - may increase computation time")
        
        # Check elbow-to-part ratio
        if len(self.elbows) > len(self.parts):
            warnings.append("More elbows than parts - verify configuration")
        
        # Print warnings if any
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  [!] {warning}")
        
        print("="*70)
    
    #Done
    def _initialize_arrays(self):
        """Initialize arrays for storing simulation results
        
        This pre-allocates all the numpy arrays we'll need to store the
        simulation history. Arrays are 2D: [time_step, segment]
        """
        # TEMPERATURE ARRAYS (all in Kelvin)
        # Gas temperature at each segment over time
        self.T_gas = np.zeros((self.time_steps + 1, self.total_segments))
        # Wall temperature - for single wall this is the only wall temp
        # For triple wall, this mirrors T_liner for compatibility
        self.T_wall = np.zeros((self.time_steps + 1, self.total_segments))
        # Outer surface temperature (same as T_wall for single wall)
        self.T_outer = np.zeros((self.time_steps + 1, self.total_segments))
        
        # TRIPLE WALL TEMPERATURE ARRAYS
        # These store the temperature of each layer in triple wall sections
        self.T_liner = np.zeros((self.time_steps + 1, self.total_segments))  # Inner liner (contacts gas)
        self.T_middle = np.zeros((self.time_steps + 1, self.total_segments))  # Insulation layer
        # Note: T_outer is used for outer layer of triple wall
        
        # FLOW PROPERTIES
        self.mass_flow_rate = np.zeros(self.time_steps + 1)  # Mass flow rate in kg/s (scalar per timestep)
        self.velocity = np.zeros((self.time_steps + 1, self.total_segments))  # Gas velocity in m/s
        self.pressure = np.zeros((self.time_steps + 1, self.total_segments))  # Pressure in Pa
        self.density = np.zeros((self.time_steps + 1, self.total_segments))  # Gas density in kg/m³
        
        # Heat transfer coefficients
        self.h_conv_inner = np.zeros((self.time_steps + 1, self.total_segments))  # W/(m²·K)
        self.h_conv_outer = np.zeros((self.time_steps + 1, self.total_segments))  # W/(m²·K)

        # Door opening history (for dynamic door control)
        # Stores opening percentage (0-100) at each timestep
        self.door_opening_history = np.zeros(self.time_steps + 1)
        # Initialize with current opening percent (or 100 if no restriction)
        initial_opening = self.inlet_restriction.door_opening_percent if self.inlet_restriction.enable else 100.0
        self.door_opening_history[0] = initial_opening

        # Time array
        self.time_array = np.linspace(0, self.total_time, self.time_steps + 1)
        
        # Set initial conditions
        self._set_initial_conditions()
    #Done    
    def _set_initial_conditions(self):
        """Set initial temperature and flow conditions with consistent initialization"""
        # Initialize with small temperature rise to avoid nonphysical cooling pulse
        # This ensures initial heat flux is near zero
        print(f"DEBUG: Initializing conditions for {len(self.parts)} parts")
        
        for i, part in enumerate(self.parts):
            start_seg = sum(p.num_segments for p in self.parts[:i])
            end_seg = start_seg + part.num_segments
            
            # Gas initialization: bottom segment gets inlet temp, others start at local ambient
            # Hot gas from the fire propagates up over time, heating each segment
            for seg in range(start_seg, end_seg):
                if seg == 0:
                    # Bottom segment gets hot inlet air (from fire)
                    # Use inlet_temp_func or CSV at t=0 if available
                    if self.inlet_temp_func is not None:
                        initial_gas_temp = float(self.inlet_temp_func(0.0))
                    elif self.inlet_temp_interp is not None:
                        initial_gas_temp = float(self.inlet_temp_interp(0.0))
                    else:
                        initial_gas_temp = part.T_ambient_K + 10.0  # Small initial rise
                else:
                    # Other segments start at their local ambient temperature
                    # Indoor sections at T_indoor, outdoor sections at T_outdoor
                    # Hot gas propagates up as simulation runs
                    initial_gas_temp = part.T_ambient_K

                # Initialize ALL timesteps to prevent -273.1°C issue
                self.T_gas[:, seg] = initial_gas_temp
            
            # Initial wall temperatures - start at ambient
            # Walls heat up slowly from ambient according to physics
            initial_wall_temp = part.T_ambient_K  # Start at ambient temperature
            
            print(f"DEBUG: Part {i+1} ({part.type}), segments {start_seg}-{end_seg-1}, T_ambient={part.T_ambient_K-273.15:.1f}°C")
            
            if part.type == 'single':
                self.T_wall[:, start_seg:end_seg] = initial_wall_temp
                self.T_outer[:, start_seg:end_seg] = initial_wall_temp
                # Initialize triple wall arrays to ambient for single wall segments
                # This prevents -273.1°C errors in diagnostics
                self.T_liner[:, start_seg:end_seg] = initial_wall_temp
                self.T_middle[:, start_seg:end_seg] = initial_wall_temp
                print(f"DEBUG: Single wall initialized - T_wall[0,{start_seg}] = {self.T_wall[0,start_seg]-273.15:.1f}°C")
            elif part.type == 'triple':
                # Initialize all triple wall layers at ambient
                # They will heat up according to physics
                self.T_liner[:, start_seg:end_seg] = part.T_ambient_K
                self.T_middle[:, start_seg:end_seg] = part.T_ambient_K
                self.T_outer[:, start_seg:end_seg] = part.T_ambient_K
                # Set wall temperature to liner for compatibility
                self.T_wall[:, start_seg:end_seg] = part.T_ambient_K
                print(f"DEBUG: Triple wall initialized - T_liner[0,{start_seg}] = {self.T_liner[0,start_seg]-273.15:.1f}°C")
            
            # Initial density based on actual temperature for each segment
            for seg in range(start_seg, end_seg):
                # Use the actual gas temperature for this segment
                seg_gas_temp = self.T_gas[0, seg]  # Already set above
                initial_density = self.physics.air_density(seg_gas_temp, part.Pa)
                self.density[:, seg] = initial_density
                
                # Initialize pressure to atmospheric for all timesteps
                self.pressure[:, seg] = part.Pa
                
                # Initialize velocity to small positive value
                self.velocity[:, seg] = 0.1  # m/s
        
        # Initial mass flow rate (kg/s) - small positive value for all timesteps
        self.mass_flow_rate[:] = 0.01
        
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
        """Process and validate inlet restriction configuration

        Converts door geometry (opening %, area) to effective flow area.
        The restriction is modeled as a quadratic resistance in mdot-space:
            R_door = K_equiv / (2 × ρ × A_eff²)
        This resistance is added to total_resistance in the momentum solver.

        Supports two door types:
        - SWIVEL (hinged): A_eff = A_door × sin(θ), where θ = opening% × π/2
        - SLIDING: A_eff = A_door × opening%

        Minimum leakage area ensures finite resistance when door is closed:
            A_eff_used = A_eff_raw + A_leak

        Args:
            inlet_restriction: Dict with restriction parameters or None
        """
        import math

        # Create config object
        if inlet_restriction is None:
            self.inlet_restriction = InletRestrictionConfig()
            return

        if isinstance(inlet_restriction, dict):
            self.inlet_restriction = InletRestrictionConfig(**inlet_restriction)
        else:
            self.inlet_restriction = inlet_restriction

        # Early return if disabled
        if not self.inlet_restriction.enable:
            return

        # --- VALIDATION ---

        # Check door opening percent bounds
        if not (0.0 <= self.inlet_restriction.door_opening_percent <= 100.0):
            raise ValueError(
                f"door_opening_percent must be between 0 and 100, "
                f"got {self.inlet_restriction.door_opening_percent}"
            )

        # Check door dimensions are positive
        if self.inlet_restriction.door_width_ft <= 0:
            raise ValueError(
                f"door_width_ft must be positive, got {self.inlet_restriction.door_width_ft}"
            )
        if self.inlet_restriction.door_height_ft <= 0:
            raise ValueError(
                f"door_height_ft must be positive, got {self.inlet_restriction.door_height_ft}"
            )

        # Calculate door area from dimensions (convert feet to meters)
        FT_TO_M = 0.3048
        door_width_m = self.inlet_restriction.door_width_ft * FT_TO_M
        door_height_m = self.inlet_restriction.door_height_ft * FT_TO_M
        self.inlet_restriction.door_area_m2 = door_width_m * door_height_m

        # Validate discharge coefficient
        if self.inlet_restriction.Cd <= 0 or self.inlet_restriction.Cd > 1.0:
            raise ValueError(
                f"Cd must be in range (0, 1.0], got {self.inlet_restriction.Cd}"
            )

        # Validate leakage area
        if self.inlet_restriction.A_leak_m2 <= 0:
            raise ValueError(
                f"A_leak_m2 must be positive, got {self.inlet_restriction.A_leak_m2}"
            )

        # Validate K_loss override if provided
        if self.inlet_restriction.K_loss_override is not None:
            if self.inlet_restriction.K_loss_override < 0:
                raise ValueError("K_loss_override cannot be negative")
            if self.inlet_restriction.K_loss_override > 100:
                print("WARNING: Very large K_loss (>100) may cause numerical issues")

        # --- VALIDATE DOOR TYPE ---
        door_type = self.inlet_restriction.door_type.lower()
        if door_type not in ('swivel', 'sliding'):
            raise ValueError(
                f"door_type must be 'swivel' or 'sliding', got '{self.inlet_restriction.door_type}'"
            )

        # --- CALCULATE EFFECTIVE AREA ---
        # Two door types with different geometry:
        #
        # SWIVEL (hinged) door:
        #   - Opens by rotating around hinge
        #   - At 100% open = 90 degrees from closed
        #   - A_eff = A_door × sin(angle), angle = opening% × π/2
        #
        # SLIDING door:
        #   - Slides open linearly
        #   - A_eff = A_door × opening% (linear relationship)

        opening_pct = self.inlet_restriction.door_opening_percent
        full_door_area = self.inlet_restriction.door_area_m2

        if door_type == 'swivel':
            # Swivel door: A_eff = A_door × sin(θ), θ = opening% × π/2
            angle_rad = (opening_pct / 100.0) * (math.pi / 2.0)
            A_eff_raw = full_door_area * math.sin(angle_rad)
        else:  # sliding
            # Sliding door: A_eff = A_door × (opening% / 100)
            A_eff_raw = full_door_area * (opening_pct / 100.0)

        self.inlet_restriction.A_eff_raw_m2 = A_eff_raw

        # --- COMBINE RAW AREA WITH LEAKAGE ---
        # Two modes available:
        #   "linear" (default):    A_eff = A_eff_raw + A_leak
        #   "quadrature":          A_eff = sqrt(A_eff_raw^2 + A_leak^2)
        # Quadrature is smoother near A_eff_raw=0 and models two parallel
        # orifice paths whose resistances add in parallel.
        A_leak = self.inlet_restriction.A_leak_m2
        area_mode = self.inlet_restriction.inlet_area_mode.lower()
        if area_mode == "quadrature":
            A_eff_used = math.sqrt(A_eff_raw**2 + A_leak**2)
        else:
            # Default: linear
            A_eff_used = A_eff_raw + A_leak

        # --- CAP AT PIPE AREA (optional) ---
        cap_mode = self.inlet_restriction.inlet_area_cap_mode.lower()
        if cap_mode == "pipe":
            A_pipe = self.parts[0].inner_area_m2
            if A_eff_used > A_pipe:
                A_eff_used = A_pipe
                print(f"  [Note: A_eff capped at pipe area = {A_pipe*1e4:.0f} cm^2]")

        # --- CAP AT MAXIMUM EFFECTIVE AREA ---
        # Even when door is wide open, flow is limited by downstream restriction
        # (tee opening, pipe area). Cap effective area at this maximum.
        max_area = self.inlet_restriction.max_effective_area_m2
        if max_area is not None and A_eff_used > max_area:
            A_eff_used = max_area
            print(f"  [Note: A_eff capped at max_effective_area = {max_area*1e4:.0f} cm^2]")

        self.inlet_restriction.effective_area_m2 = A_eff_used

        # Get inlet pipe geometry (for reference only)
        inlet_part = self.parts[0]
        inlet_pipe_area = inlet_part.inner_area_m2

        # --- CALCULATE K_equiv FROM DISCHARGE COEFFICIENT ---
        # K_equiv = 1/Cd² (used in resistance formula)
        if self.inlet_restriction.K_loss_override is not None:
            # User override takes precedence
            self.inlet_restriction.K_equiv = self.inlet_restriction.K_loss_override
        else:
            # K_equiv = 1/Cd²
            Cd = self.inlet_restriction.Cd
            self.inlet_restriction.K_equiv = 1.0 / (Cd ** 2)

        # --- PRINT SUMMARY ---

        area_ratio = A_eff_used / inlet_pipe_area

        # Calculate example R_door for reference (at inlet density ≈ 1.2 kg/m³)
        rho_ref = 1.2  # kg/m³ (typical inlet)
        K = self.inlet_restriction.K_equiv
        R_door_ref = K / (2 * rho_ref * A_eff_used**2)

        # Convert areas to sq inches for display
        M2_TO_SQIN = 1550.0031  # 1 m² = 1550 sq in
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
        """Update inlet restriction effective area for a new door opening percentage

        This method recalculates the effective flow area when door opening changes.
        Used by dynamic door control (door_opening_func) during simulation.

        The calculation mirrors _initialize_inlet_restriction but only updates
        the opening-dependent values (A_eff_raw, effective_area_m2).

        Args:
            opening_percent: New door opening percentage (0-100)
        """
        import math

        if not self.inlet_restriction.enable:
            return

        # Validate opening percent
        opening_percent = max(0.0, min(100.0, opening_percent))

        # Store new opening percent
        self.inlet_restriction.door_opening_percent = opening_percent

        # Recalculate effective area based on door type
        full_door_area = self.inlet_restriction.door_area_m2
        door_type = self.inlet_restriction.door_type.lower()

        if door_type == 'swivel':
            # Swivel door: A_eff = A_door × sin(θ), θ = opening% × π/2
            angle_rad = (opening_percent / 100.0) * (math.pi / 2.0)
            A_eff_raw = full_door_area * math.sin(angle_rad)
        else:  # sliding
            # Sliding door: A_eff = A_door × (opening% / 100)
            A_eff_raw = full_door_area * (opening_percent / 100.0)

        self.inlet_restriction.A_eff_raw_m2 = A_eff_raw

        # Combine raw area with leakage using configured mode
        A_leak = self.inlet_restriction.A_leak_m2
        area_mode = self.inlet_restriction.inlet_area_mode.lower()
        if area_mode == "quadrature":
            A_eff_used = math.sqrt(A_eff_raw**2 + A_leak**2)
        else:
            A_eff_used = A_eff_raw + A_leak

        # Cap at pipe area if configured
        cap_mode = self.inlet_restriction.inlet_area_cap_mode.lower()
        if cap_mode == "pipe":
            A_pipe = self.parts[0].inner_area_m2
            A_eff_used = min(A_eff_used, A_pipe)

        # Cap at maximum effective area (e.g., tee/pipe opening)
        max_area = self.inlet_restriction.max_effective_area_m2
        if max_area is not None and A_eff_used > max_area:
            A_eff_used = max_area

        self.inlet_restriction.effective_area_m2 = A_eff_used

    def _load_inlet_temp_csv(self, csv_config):
        """Load inlet temperature data from CSV and create interpolation function

        Reads a CSV file with time and temperature columns, then creates an
        interpolation function that returns temperature for any time value.
        Linear interpolation is used between data points.

        Args:
            csv_config: Either a string path to CSV, or a dict with:
                - 'path': Path to CSV file (required)
                - 'time_col': Column name for time (default: 'PC_Test Time [min]')
                - 'temp_col': Column name for temperature (default: 'Flue Temp')
                - 'time_unit': 'min' or 's' (default: 'min')
                - 'temp_unit': 'F', 'C', or 'K' (default: 'F')
        """
        from scipy import interpolate

        # Parse config - can be string path or dict
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
            # Read CSV file
            df = pd.read_csv(csv_path)
            self.inlet_temp_df = df

            # Check for required columns
            if time_col not in df.columns:
                raise ValueError(f"CSV must have column '{time_col}'. Found: {list(df.columns)}")
            if temp_col not in df.columns:
                raise ValueError(f"CSV must have column '{temp_col}'. Found: {list(df.columns)}")

            # Extract and clean data
            time_raw = df[time_col].values
            temp_raw = df[temp_col].values

            # Remove any NaN values
            valid_mask = ~(np.isnan(time_raw) | np.isnan(temp_raw))
            time_raw = time_raw[valid_mask]
            temp_raw = temp_raw[valid_mask]

            if len(time_raw) < 2:
                raise ValueError("CSV must have at least 2 valid data points")

            # Convert time to seconds
            if time_unit == 'min':
                time_sec = time_raw * 60.0
            elif time_unit == 's':
                time_sec = time_raw
            else:
                raise ValueError(f"Unknown time_unit: {time_unit}. Use 'min' or 's'")

            # Convert temperature to Kelvin
            if temp_unit == 'F':
                # T_K = (T_F - 32) * 5/9 + 273.15
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

            # Create linear interpolation function
            # fill_value='extrapolate' allows values outside the data range
            self.inlet_temp_interp = interpolate.interp1d(
                time_sec, temp_K,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )

            # Print summary
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

            # Check data granularity
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
        """Get inlet temperature at a given time

        Priority order:
        1. inlet_temp_func (callable) - if provided
        2. inlet_temp_interp (from CSV) - if CSV was loaded
        3. Default exponential ramp-up model

        Applies door temperature dilution if enabled.

        Args:
            t_sec: Time in seconds
            T_ambient: Ambient temperature in Kelvin (used by default model)

        Returns:
            Inlet temperature in Kelvin
        """
        if self.inlet_temp_func is not None:
            # Use user-provided function (e.g., from experimental TC1 data)
            T_inlet = float(self.inlet_temp_func(t_sec))
        elif self.inlet_temp_interp is not None:
            # Use CSV interpolated data
            T_inlet = float(self.inlet_temp_interp(t_sec))
        else:
            # Use default exponential ramp-up model
            T_inlet = self._default_inlet_temp(t_sec, T_ambient=T_ambient)

        # Apply door temperature dilution (Jan 27 feature)
        # When door is open, cold ambient air mixes with hot combustion gas
        if self.door_temp_dilution_enable and self.door_opening_func is not None:
            door_pct = self.door_opening_func(t_sec)
            # Dilution fraction: at 100% open with strength=0.8, inlet is 80% ambient + 20% combustion
            dilution_frac = min(max((door_pct / 100.0) * self.door_temp_dilution_strength, 0.0), 1.0)
            T_inlet = T_inlet * (1.0 - dilution_frac) + T_ambient * dilution_frac

        return T_inlet

    def _initialize_catalytic_bypass(self, catalytic_bypass: Optional[Dict[str, Any]]):
        """Process and validate catalytic bypass damper configuration

        Converts dict input to CatalyticBypassConfig dataclass,
        validates parameters, calculates effective K-loss based on bypass state,
        and prints summary if enabled.

        Args:
            catalytic_bypass: Dict with bypass parameters or None
        """
        # Create config object
        if catalytic_bypass is None:
            self.catalytic_bypass = CatalyticBypassConfig()
            return

        if isinstance(catalytic_bypass, dict):
            self.catalytic_bypass = CatalyticBypassConfig(**catalytic_bypass)
        else:
            self.catalytic_bypass = catalytic_bypass

        # Early return if disabled
        if not self.catalytic_bypass.enable:
            return

        # --- VALIDATION ---

        # Check segment_index is valid
        if not (0 <= self.catalytic_bypass.segment_index < self.total_segments):
            raise ValueError(
                f"segment_index must be between 0 and {self.total_segments - 1}, "
                f"got {self.catalytic_bypass.segment_index}"
            )

        # Validate K values
        if self.catalytic_bypass.K_bypass_open < 0:
            raise ValueError(
                f"K_bypass_open must be non-negative, got {self.catalytic_bypass.K_bypass_open}"
            )

        if self.catalytic_bypass.K_catalyst_closed < 0:
            raise ValueError(
                f"K_catalyst_closed must be non-negative, got {self.catalytic_bypass.K_catalyst_closed}"
            )

        # Validate and convert bypass_state to bypass_fraction (0-1)
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
            # Numeric value: 0 = closed (through catalyst), 100 = open (bypass)
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

        # --- CALCULATE EFFECTIVE K-LOSS ---

        # Linear interpolation between closed (catalyst) and open (bypass)
        # K_effective = K_closed + bypass_fraction * (K_open - K_closed)
        # When closed (bypass_fraction=0): K_effective = K_catalyst_closed
        # When open (bypass_fraction=1): K_effective = K_bypass_open
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
        """Process and validate leakage configuration for pipe joints

        Distributes total leakage area across segments based on pipe type.
        Black stove pipe has slip joints → higher leakage fraction.
        Insulated pipe has sealed joints → lower leakage fraction.

        Leakage flow at each segment is calculated using VENT-II style orifice model:
            m_leak = Cd × A_leak × sqrt(2 × rho × |ΔP|)

        Args:
            leakage_config: Dict with leakage parameters or None
        """
        # Create config object
        if leakage_config is None:
            self.leakage_config = LeakageConfig()
            self.segment_leakage_area_m2 = np.zeros(self.total_segments)
            return

        if isinstance(leakage_config, dict):
            self.leakage_config = LeakageConfig(**leakage_config)
        else:
            self.leakage_config = leakage_config

        # Early return if disabled
        if not self.leakage_config.enable:
            self.segment_leakage_area_m2 = np.zeros(self.total_segments)
            return

        # --- VALIDATION ---

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

        # --- DISTRIBUTE LEAKAGE AREA ACROSS SEGMENTS ---

        # Convert total area to m²
        total_area_m2 = self.leakage_config.total_leakage_area_cm2 * 1e-4

        # Count segments by pipe type (single = black stove pipe, triple = insulated)
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

        # Distribute area proportionally
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

        # --- PRINT SUMMARY ---

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
        """
        Get wind-induced pressure at chimney cap for a given timestep

        Args:
            step_index: Current simulation timestep index
            Pa: Ambient pressure at outlet (Pa)
            T_out_K: Ambient air temperature at outlet (K)

        Returns:
            Wind pressure effect ΔP_wind (Pa)
            Negative = suction (aids draft), Positive = pressurization (hurts draft)
        """
        cfg = self._wind_cfg
        if not cfg.enable:
            return 0.0

        # Static mode — return cached value
        if cfg.update_mode == "static" and self._deltaP_wind_cap_static is not None:
            return self._deltaP_wind_cap_static

        # Dynamic mode — compute per-step wind
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
        self._last_deltaP_wind_cap = dP  # Store for diagnostics
        return dP

    #following functions needs to be added to docs
    def plot_gas_temperatures_only(self,name=None):
        """Plot only gas temperatures for each segment over time"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create color map for segments
        n_segments = self.total_segments
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_segments))

        current_seg = 0
        for part_idx, part in enumerate(self.parts):
            for local_seg in range(part.num_segments):
                global_seg = current_seg + local_seg

                # Plot gas temperature for this segment
                gas_temp_C = self.T_gas[:, global_seg] - 273.15

                label = f'Part {part_idx+1}, Seg {local_seg+1}'
                ax.plot(self.time_array / 60,  # Convert to minutes
                        gas_temp_C,
                        color=colors[global_seg],
                        label=label,
                        linewidth=1.5)

            current_seg += part.num_segments

        # Format plot
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Gas Temperature (°C)', fontsize=12)
        ax.set_title('Flue Gas Temperature Evolution by Segment', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        plt.tight_layout()
        if name: 
            plt.savefig(name)
        plt.show()

    def plot_draft_simple(self,name=None):
        """Simple plot of total draft over time"""
        fig, ax = plt.subplots(figsize=(10, 5))

        time_min = self.time_array / 60

        # Calculate total draft at each timestep
        draft = np.zeros(len(self.time_array))

        for t in range(len(self.time_array)):
            total_draft = 0
            for part_idx, part in enumerate(self.parts):
                if part.orientation.lower() == 'vertical':
                    # Get segments for this part
                    start_seg = sum(p.num_segments for p in self.parts[:part_idx])
                    end_seg = start_seg + part.num_segments

                    for seg in range(start_seg, end_seg):
                        # Draft = g * h * (ρ_ambient - ρ_gas)
                        seg_height = part.height_m / part.num_segments
                        rho_ambient = self.physics.air_density(part.T_ambient_K, part.Pa)
                        rho_gas = self.density[t, seg]

                        total_draft += 9.81 * seg_height * (rho_ambient - rho_gas)

            draft[t] = total_draft

        # Plot
        ax.plot(time_min, draft, 'b-', linewidth=2)
        ax.fill_between(time_min, 0, draft, alpha=0.3)

        # Add final value annotation
        ax.axhline(draft[-1], color='r', linestyle='--', alpha=0.5)
        ax.text(time_min[-1]*0.98, draft[-1], f'{draft[-1]:.1f} Pa',
                ha='right', va='bottom', fontsize=10)

        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Draft Pressure (Pa)', fontsize=12)
        ax.set_title('Chimney Draft Development Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_min[-1])

        plt.tight_layout()
        if name: 
            plt.savefig(name)
        plt.show()
    #plotting/Analyzing
    #Done
    def plot_temperature_profiles(self, times_to_plot: Optional[List[float]] = None):
        """Plot comprehensive temperature profiles with separate triple wall layers
        
        Args:
            times_to_plot: List of times to plot. If None, plots initial, middle, and final
        """
        if times_to_plot is None:
            times_to_plot = [0, self.total_time/2, self.total_time]
        
        # Check if we have triple wall parts
        has_triple_wall = any(part.type == 'triple' for part in self.parts)
        
        # Create segment positions and part boundaries
        positions = []
        part_boundaries = []
        current_z = 0
        
        for part_idx, part in enumerate(self.parts):
            seg_height = part.height_m / part.num_segments
            part_start_z = current_z
            
            for i in range(part.num_segments):
                positions.append(current_z + (i + 0.5) * seg_height)
            
            current_z += part.height_m
            wall_type = "Triple" if part.type == 'triple' else "Single"
            part_boundaries.append((part_start_z, current_z, f"Part {part_idx+1} ({wall_type})"))
        
        # Create figure with more subplots if triple wall
        if has_triple_wall:
            fig = plt.figure(figsize=(20, 15))
            subplot_rows, subplot_cols = 3, 3
        else:
            fig = plt.figure(figsize=(16, 12))
            subplot_rows, subplot_cols = 2, 3
        
        # 1. Temperature vs height profiles (with triple wall layers)
        ax1 = plt.subplot(subplot_rows, subplot_cols, 1)
        colors = plt.cm.Set1(np.linspace(0, 1, len(times_to_plot)))
        
        for i, time_val in enumerate(times_to_plot):
            time_idx = int(time_val / self.dt)
            time_idx = min(time_idx, self.time_steps)
            
            # Gas temperature
            ax1.plot(self.T_gas[time_idx] - 273.15, positions, 
                    label=f'Gas T at t={time_val:.1f}s', 
                    linestyle='-', color=colors[i], marker='o', markersize=4, linewidth=2)
            
            if has_triple_wall:
                # Show all wall layers for triple wall sections
                ax1.plot(self.T_liner[time_idx] - 273.15, positions, 
                        label=f'Liner T at t={time_val:.1f}s', 
                        linestyle='--', color=colors[i], marker='s', markersize=3)
                ax1.plot(self.T_middle[time_idx] - 273.15, positions, 
                        label=f'Middle T at t={time_val:.1f}s', 
                        linestyle=':', color=colors[i], marker='^', markersize=3)
                ax1.plot(self.T_outer[time_idx] - 273.15, positions, 
                        label=f'Outer T at t={time_val:.1f}s', 
                        linestyle='-.', color=colors[i], marker='d', markersize=3)
            else:
                ax1.plot(self.T_wall[time_idx] - 273.15, positions, 
                        label=f'Wall T at t={time_val:.1f}s', 
                        linestyle='--', color=colors[i], marker='s', markersize=4)
        
        # Add part boundaries
        for start_z, end_z, label in part_boundaries:
            ax1.axhline(y=start_z, color='gray', linestyle=':', alpha=0.7)
            ax1.text(ax1.get_xlim()[0] if ax1.get_xlim()[0] != ax1.get_xlim()[1] else 20, 
                    (start_z + end_z)/2, label, rotation=90, ha='left', va='center', fontsize=8)
        
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Temperature vs Height\n(All Layers)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Temperature vs time by part (with triple wall layers)
        ax2 = plt.subplot(subplot_rows, subplot_cols, 2)
        current_seg = 0
        
        # Define colors for parts (still needed for other plots)
        part_colors = plt.cm.tab10(np.linspace(0, 1, len(self.parts)))
        
        # Define distinct colors for different temperature types
        gas_color = 'red'
        liner_color = 'orange' 
        middle_color = 'green'
        outer_color = 'blue'
        wall_color = 'purple'
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            
            # Average temperatures for this part
            part_gas_temps = np.mean(self.T_gas[:, current_seg:end_seg], axis=1) - 273.15
            
            ax2.plot(self.time_array, part_gas_temps, 
                    label=f'Part {part_idx+1} Gas', color=gas_color, linestyle='-', linewidth=3)
            
            if part.type == 'triple':
                part_liner_temps = np.mean(self.T_liner[:, current_seg:end_seg], axis=1) - 273.15
                part_middle_temps = np.mean(self.T_middle[:, current_seg:end_seg], axis=1) - 273.15
                part_outer_temps = np.mean(self.T_outer[:, current_seg:end_seg], axis=1) - 273.15
                
                ax2.plot(self.time_array, part_liner_temps, 
                        label=f'Part {part_idx+1} Liner', color=liner_color, linestyle='--', linewidth=2)
                ax2.plot(self.time_array, part_middle_temps, 
                        label=f'Part {part_idx+1} Middle', color=middle_color, linestyle=':', linewidth=2)
                ax2.plot(self.time_array, part_outer_temps, 
                        label=f'Part {part_idx+1} Outer', color=outer_color, linestyle='-.', linewidth=2)
            else:
                part_wall_temps = np.mean(self.T_wall[:, current_seg:end_seg], axis=1) - 273.15
                ax2.plot(self.time_array, part_wall_temps, 
                        label=f'Part {part_idx+1} Wall', color=wall_color, linestyle='--', linewidth=2)
            
            current_seg = end_seg
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature Evolution by Part')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Final temperature distribution
        ax3 = plt.subplot(subplot_rows, subplot_cols, 3)
        final_gas_temps = self.T_gas[-1] - 273.15
        segment_numbers = range(1, self.total_segments + 1)
        x_pos = np.arange(len(segment_numbers))
        
        if has_triple_wall:
            bar_width = 0.2
            ax3.bar(x_pos - 1.5*bar_width, final_gas_temps, bar_width, 
                   label='Gas', alpha=0.8, color='red')
            
            # Create arrays that show single wall temp for all layers when it's a single wall
            liner_temps = np.zeros(self.total_segments)
            middle_temps = np.zeros(self.total_segments)
            outer_temps = np.zeros(self.total_segments)
            
            current_seg = 0
            for part in self.parts:
                end_seg = current_seg + part.num_segments
                if part.type == 'single':
                    # For single wall, show wall temperature for all three layers
                    liner_temps[current_seg:end_seg] = self.T_wall[-1, current_seg:end_seg] - 273.15
                    middle_temps[current_seg:end_seg] = self.T_wall[-1, current_seg:end_seg] - 273.15
                    outer_temps[current_seg:end_seg] = self.T_wall[-1, current_seg:end_seg] - 273.15
                else:  # triple
                    # For triple wall, show actual layer temperatures
                    liner_temps[current_seg:end_seg] = self.T_liner[-1, current_seg:end_seg] - 273.15
                    middle_temps[current_seg:end_seg] = self.T_middle[-1, current_seg:end_seg] - 273.15
                    outer_temps[current_seg:end_seg] = self.T_outer[-1, current_seg:end_seg] - 273.15
                current_seg = end_seg
            
            ax3.bar(x_pos - 0.5*bar_width, liner_temps, bar_width, 
                   label='Liner', alpha=0.8, color='orange')
            ax3.bar(x_pos + 0.5*bar_width, middle_temps, bar_width, 
                   label='Middle', alpha=0.8, color='green')
            ax3.bar(x_pos + 1.5*bar_width, outer_temps, bar_width, 
                   label='Outer', alpha=0.8, color='blue')
        else:
            bar_width = 0.35
            ax3.bar(x_pos - bar_width/2, final_gas_temps, bar_width, 
                   label='Gas', alpha=0.8, color='red')
            ax3.bar(x_pos + bar_width/2, self.T_wall[-1] - 273.15, bar_width, 
                   label='Wall', alpha=0.8, color='blue')
        
        # Add part boundaries
        current_seg = 0
        for part_idx, part in enumerate(self.parts):
            if part_idx > 0:
                ax3.axvline(x=current_seg - 0.5, color='gray', linestyle='--', alpha=0.7)
            current_seg += part.num_segments
        
        ax3.set_xlabel('Segment Number')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title(f'Final Temperature Distribution')
        ax3.set_xticks(x_pos[::max(1, len(x_pos)//8)])
        ax3.set_xticklabels([str(i+1) for i in x_pos[::max(1, len(x_pos)//8)]])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Triple wall temperature differences (if applicable)
        if has_triple_wall:
            ax4 = plt.subplot(subplot_rows, subplot_cols, 4)
            
            # Calculate temperature differences in triple wall sections
            liner_gas_diff = self.T_gas[-1] - self.T_liner[-1]
            middle_liner_diff = self.T_liner[-1] - self.T_middle[-1]
            outer_middle_diff = self.T_middle[-1] - self.T_outer[-1]
            outer_ambient_diff = self.T_outer[-1] - np.array([part.T_ambient_K for part in self.parts for _ in range(part.num_segments)])
            
            ax4.plot(liner_gas_diff, positions, label='Gas - Liner', marker='o', linestyle='-')
            ax4.plot(middle_liner_diff, positions, label='Liner - Middle', marker='s', linestyle='-')
            ax4.plot(outer_middle_diff, positions, label='Middle - Outer', marker='^', linestyle='-')
            ax4.plot(outer_ambient_diff, positions, label='Outer - Ambient', marker='d', linestyle='-')
            
            ax4.set_xlabel('Temperature Difference (K)')
            ax4.set_ylabel('Height (m)')
            ax4.set_title('Temperature Differences Across Layers')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Insulation effectiveness
            ax5 = plt.subplot(subplot_rows, subplot_cols, 5)
            current_seg = 0
            
            for part_idx, part in enumerate(self.parts):
                end_seg = current_seg + part.num_segments
                if part.type == 'triple':
                    # Calculate insulation effectiveness: (T_liner - T_outer) / (T_liner - T_ambient)
                    liner_temps = np.mean(self.T_liner[:, current_seg:end_seg], axis=1)
                    outer_temps = np.mean(self.T_outer[:, current_seg:end_seg], axis=1)
                    
                    # Debug: Check temperature values

                    # Avoid division by zero
                    denominator = liner_temps - part.T_ambient_K
                    denominator = np.where(np.abs(denominator) < 1.0, 1.0, denominator)  # Avoid very small denominators
                    effectiveness = (liner_temps - outer_temps) / denominator * 100
                    effectiveness = np.clip(effectiveness, 0, 100)  # Limit to 0-100%
                    
                    ax5.plot(self.time_array, effectiveness, 
                            label=f'Part {part_idx+1} Insulation Effectiveness', 
                            color=part_colors[part_idx], linewidth=2)
                current_seg = end_seg
            
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Insulation Effectiveness (%)')
            ax5.set_title('Triple Wall Insulation Performance')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Summary statistics - use subplot 6 for triple wall (bottom right) or 3 for single wall
        if has_triple_wall:
            stats_subplot_num = 6  # Bottom right position in 3x3 grid (instead of subplot 9)
        else:
            stats_subplot_num = 6  # Bottom right position in 2x3 grid
        ax_stats = plt.subplot(subplot_rows, subplot_cols, stats_subplot_num)
        ax_stats.axis('off')
        
        # Calculate comprehensive statistics
        final_gas_avg = np.mean(final_gas_temps)
        max_gas_temp = np.max(self.T_gas) - 273.15
        
        stats_text = f"""SIMULATION SUMMARY
        
Parts: {len(self.parts)}
Total Segments: {self.total_segments}
Simulation Time: {self.total_time}s
        
FINAL AVERAGE TEMPERATURES (°C):
Gas: {final_gas_avg:.1f}
"""
        
        if has_triple_wall:
            final_liner_avg = np.mean(self.T_liner[-1] - 273.15)
            final_middle_avg = np.mean(self.T_middle[-1] - 273.15)
            final_outer_avg = np.mean(self.T_outer[-1] - 273.15)
            
            stats_text += f"""Liner: {final_liner_avg:.1f}
Middle: {final_middle_avg:.1f}
Outer: {final_outer_avg:.1f}

MAXIMUM TEMPERATURES (°C):
Gas: {max_gas_temp:.1f}
Liner: {np.max(self.T_liner) - 273.15:.1f}
Outer: {np.max(self.T_outer) - 273.15:.1f}"""
        else:
            final_wall_avg = np.mean(self.T_wall[-1] - 273.15)
            stats_text += f"""Wall: {final_wall_avg:.1f}

MAXIMUM TEMPERATURES (°C):
Gas: {max_gas_temp:.1f}
Wall: {np.max(self.T_wall) - 273.15:.1f}"""
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Also show segment-wise diagnostics
        self.plot_segment_diagnostics()
        
        # Print detailed segment information
        self._print_detailed_segment_info()
    #Done
    def analyze_draft(self, plot=True, save_plots=False, plot_dir='./'):
        """
        Analyze chimney draft performance
        
        Args:
            plot: Whether to create plots
            save_plots: Whether to save plots to files
            plot_dir: Directory to save plots
            
        Returns:
            DraftResults object with analysis
        """
        from .draft_analysis import DraftAnalyzer
        
        analyzer = DraftAnalyzer(self)
        results = analyzer.calculate_draft()
        
        # Print summary
        analyzer.print_draft_summary(results)
        
        # Create plots if requested
        if plot:
            import os
            
            # Main draft analysis plot
            fig1 = analyzer.plot_draft_analysis(results)
            if save_plots:
                save_path = os.path.join(plot_dir, 'draft_analysis.png')
                fig1.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved draft analysis to {save_path}")
            
            # Draft vs temperature plot
            fig2 = analyzer.plot_draft_vs_temperature(results)
            if save_plots:
                save_path = os.path.join(plot_dir, 'draft_vs_temperature.png')
                fig2.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved draft vs temperature to {save_path}")
        
        return results
    #Done
    def plot_segment_diagnostics(self):
        """Plot detailed segment-wise temperature evolution for all segments"""
        print("\nGenerating segment-wise diagnostic plots...")
        
        # Create figure with subplots for each part
        n_parts = len(self.parts)
        fig = plt.figure(figsize=(20, 5 * n_parts))
        
        current_seg = 0
        
        for part_idx, part in enumerate(self.parts):
            # Create subplot for this part
            ax = plt.subplot(n_parts, 1, part_idx + 1)
            
            # Plot each segment in this part
            for local_seg in range(part.num_segments):
                global_seg = current_seg + local_seg
                
                # Gas temperature
                ax.plot(self.time_array, self.T_gas[:, global_seg] - 273.15,
                       label=f'Seg {global_seg+1} Gas', 
                       color='red', alpha=0.7 - 0.1*local_seg, linewidth=2)
                
                if part.type == 'single':
                    # Wall temperature
                    ax.plot(self.time_array, self.T_wall[:, global_seg] - 273.15,
                           label=f'Seg {global_seg+1} Wall',
                           color='blue', alpha=0.7 - 0.1*local_seg, linewidth=1.5, linestyle='--')
                else:  # triple wall
                    # Liner temperature
                    ax.plot(self.time_array, self.T_liner[:, global_seg] - 273.15,
                           label=f'Seg {global_seg+1} Liner',
                           color='orange', alpha=0.7 - 0.1*local_seg, linewidth=1.5, linestyle='--')
                    
                    # Middle temperature
                    ax.plot(self.time_array, self.T_middle[:, global_seg] - 273.15,
                           label=f'Seg {global_seg+1} Middle',
                           color='green', alpha=0.7 - 0.1*local_seg, linewidth=1.5, linestyle=':')
                    
                    # Outer temperature
                    ax.plot(self.time_array, self.T_outer[:, global_seg] - 273.15,
                           label=f'Seg {global_seg+1} Outer',
                           color='blue', alpha=0.7 - 0.1*local_seg, linewidth=1.5, linestyle='-.')
            
            # Formatting
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'Part {part_idx+1} ({part.type.title()} Wall) - Segment Temperature Evolution\n'
                        f'Height: {part.height_m:.2f}m, Diameter: {part.inner_diameter_m*1000:.1f}mm, '
                        f'Segments: {part.num_segments}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Add ambient temperature reference
            ax.axhline(y=part.T_ambient_K - 273.15, color='gray', linestyle=':', 
                      alpha=0.5, label='Ambient')
            
            current_seg += part.num_segments
        
        plt.tight_layout()
        plt.show()
        
        # Create additional plots for heat transfer coefficients
        self._plot_heat_transfer_coefficients()
    #Done
    def temp_evolution_by_part(self, ambient_temp=False):
        # Temperature vs time by part with custom color schemes
        fig, ax = plt.subplots(1, 1, figsize=(20, 6.5))
        current_seg = 0
        
        # Convert time to minutes
        time_minutes = self.time_array / 60.0
        
        # First plot the inlet temperature in black
        T_ambient_inlet = self.parts[0].T_ambient_K
        
        # Get inlet temperatures (uses CSV if available, otherwise default model)
        inlet_temps = np.array([self._get_inlet_temp(t, T_ambient=T_ambient_inlet)
                                for t in self.time_array]) - 273.15

        ax.plot(time_minutes, inlet_temps, 
                label='Inlet Gas', color='black', linestyle='-', linewidth=3.5, zorder=10)
        
        # Also plot the gas temperature at the first segment (bottom) to show the difference
        first_seg_gas_temps = self.T_gas[:, 0] - 273.15
        ax.plot(time_minutes, first_seg_gas_temps, 
                label='Bottom Segment Gas', color='gray', linestyle='--', linewidth=2, alpha=0.7)
        
        # Get color maps for different temperature types
        # We'll use different intensities within each colormap for different parts
        n_parts = len(self.parts)
        
        # Define color intensity ranges for each part (0.3 to 0.9 to avoid too light/dark)
        color_intensities = np.linspace(0.3, 0.9, n_parts)
        
        # Plot ambient temperatures if requested
        if ambient_temp:
            # Use light blue shades for ambient temperatures
            ambient_intensities = np.linspace(0.2, 0.5, n_parts)  # Lighter shades
            for part_idx, part in enumerate(self.parts):
                ambient_color = plt.cm.Blues(ambient_intensities[part_idx])
                height_ft = round(part.z_ft, 2)
                ax.axhline(y=part.T_ambient_K - 273.15, 
                          color=ambient_color, 
                          linestyle=':', 
                          linewidth=2.5,
                          alpha=0.7,
                          label=f'Part {part_idx+1} Ambient ({height_ft} ft)')
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            
            # Average temperatures for this part
            part_gas_temps = np.mean(self.T_gas[:, current_seg:end_seg], axis=1) - 273.15
            
            # Gas temperature - use Reds colormap
            gas_color = plt.cm.Reds(color_intensities[part_idx])
            height_ft = round(part.z_ft,2)
            ax.plot(time_minutes, part_gas_temps, 
                    label=f'Part {part_idx+1} Gas ({height_ft} ft)', color=gas_color, linestyle='-', linewidth=3)
            
            if part.type == 'triple':
                part_liner_temps = np.mean(self.T_liner[:, current_seg:end_seg], axis=1) - 273.15
                part_middle_temps = np.mean(self.T_middle[:, current_seg:end_seg], axis=1) - 273.15
                part_outer_temps = np.mean(self.T_outer[:, current_seg:end_seg], axis=1) - 273.15
                
                # Triple wall inner (liner) - use Blues colormap
                liner_color = plt.cm.Blues(color_intensities[part_idx])
                ax.plot(time_minutes, part_liner_temps, 
                        label=f'Part {part_idx+1} Inner ({height_ft} ft)', color=liner_color, linestyle='--', linewidth=2)
                
                # Triple wall middle - use Purples colormap
                middle_color = plt.cm.Purples(color_intensities[part_idx])
                ax.plot(time_minutes, part_middle_temps, 
                        label=f'Part {part_idx+1} Middle ({height_ft} ft)', color=middle_color, linestyle=':', linewidth=2)
                
                # Triple wall outer - use RdPu (Red-Purple/Pink) colormap
                outer_color = plt.cm.RdPu(color_intensities[part_idx])
                ax.plot(time_minutes, part_outer_temps, 
                        label=f'Part {part_idx+1} Outer ({height_ft} ft)', color=outer_color, linestyle='-.', linewidth=2)
            else:
                part_wall_temps = np.mean(self.T_wall[:, current_seg:end_seg], axis=1) - 273.15
                
                # Single wall - use Greens colormap
                wall_color = plt.cm.Greens(color_intensities[part_idx])
                ax.plot(time_minutes, part_wall_temps, 
                        label=f'Part {part_idx+1} Wall ({height_ft} ft)', color=wall_color, linestyle='--', linewidth=2)
            
            current_seg = end_seg
        
        ax.set_xlabel('Time (minutes)', fontsize=15)
        ax.set_ylabel('Temperature (°C)', fontsize=15)
        ax.set_title('Temperature Evolution by Part', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add a text box explaining the color scheme
        color_scheme_text = (
            "Color Scheme:\n"
            "• Inlet: Black\n"
            "• Gas: Reds\n"
            "• Single Wall: Greens\n"
            "• Triple Inner: Blues\n"
            "• Triple Middle: Purples\n"
            "• Triple Outer: Pinks"
        )
        if ambient_temp:
            color_scheme_text += "\n• Ambient: Light Blues"
        ax.text(0.02, 0.98, color_scheme_text, transform=ax.transAxes, 
                fontsize=15, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done
    def temp_evolution_br(self):
        """Plot temperature evolution showing bottom, average, and top segments for each part"""
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        # Convert time to minutes
        time_minutes = self.time_array / 60.0
        
        # First plot the inlet temperature in black
        T_ambient_inlet = self.parts[0].T_ambient_K

        # Get inlet temperatures (uses CSV if available, otherwise default model)
        inlet_temps = np.array([self._get_inlet_temp(t, T_ambient=T_ambient_inlet)
                                 for t in self.time_array]) - 273.15

        ax.plot(time_minutes, inlet_temps,
                label='Inlet Gas', color='black', linestyle='-', linewidth=3.5, zorder=10)

        # Get color maps for different temperature types
        n_parts = len(self.parts)
        color_intensities = np.linspace(0.3, 0.9, n_parts)
        
        current_seg = 0
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            height_ft = round(part.height_ft)
            
            # Get temperatures for bottom, average, and top
            # Gas temperatures
            part_gas_bottom = self.T_gas[:, current_seg] - 273.15  # First segment
            part_gas_avg = np.mean(self.T_gas[:, current_seg:end_seg], axis=1) - 273.15  # Average
            part_gas_top = self.T_gas[:, end_seg-1] - 273.15  # Last segment
            
            # Plot gas temperatures - use Reds colormap
            gas_color = plt.cm.Reds(color_intensities[part_idx])
            ax.plot(time_minutes, part_gas_bottom, 
                    label=f'P{part_idx+1} Gas Bottom ({height_ft} ft)', 
                    color=gas_color, linestyle='-', linewidth=2.5, alpha=1.0)
            ax.plot(time_minutes, part_gas_avg, 
                    label=f'P{part_idx+1} Gas Avg ({height_ft} ft)', 
                    color=gas_color, linestyle='--', linewidth=2.5, alpha=0.7)
            ax.plot(time_minutes, part_gas_top, 
                    label=f'P{part_idx+1} Gas Top ({height_ft} ft)', 
                    color=gas_color, linestyle=':', linewidth=2.5, alpha=0.5)
            
            if part.type == 'triple':
                # Inner/Liner temperatures
                part_liner_bottom = self.T_liner[:, current_seg] - 273.15
                part_liner_avg = np.mean(self.T_liner[:, current_seg:end_seg], axis=1) - 273.15
                part_liner_top = self.T_liner[:, end_seg-1] - 273.15
                
                liner_color = plt.cm.Blues(color_intensities[part_idx])
                ax.plot(time_minutes, part_liner_bottom, 
                        label=f'P{part_idx+1} Inner Bottom', 
                        color=liner_color, linestyle='-', linewidth=2, alpha=1.0)
                ax.plot(time_minutes, part_liner_avg, 
                        label=f'P{part_idx+1} Inner Avg', 
                        color=liner_color, linestyle='--', linewidth=2, alpha=0.7)
                ax.plot(time_minutes, part_liner_top, 
                        label=f'P{part_idx+1} Inner Top', 
                        color=liner_color, linestyle=':', linewidth=2, alpha=0.5)
                
                # Middle layer temperatures
                part_middle_bottom = self.T_middle[:, current_seg] - 273.15
                part_middle_avg = np.mean(self.T_middle[:, current_seg:end_seg], axis=1) - 273.15
                part_middle_top = self.T_middle[:, end_seg-1] - 273.15
                
                middle_color = plt.cm.Purples(color_intensities[part_idx])
                ax.plot(time_minutes, part_middle_bottom, 
                        label=f'P{part_idx+1} Mid Bottom', 
                        color=middle_color, linestyle='-', linewidth=1.8, alpha=1.0)
                ax.plot(time_minutes, part_middle_avg, 
                        label=f'P{part_idx+1} Mid Avg', 
                        color=middle_color, linestyle='--', linewidth=1.8, alpha=0.7)
                ax.plot(time_minutes, part_middle_top, 
                        label=f'P{part_idx+1} Mid Top', 
                        color=middle_color, linestyle=':', linewidth=1.8, alpha=0.5)
                
                # Outer layer temperatures
                part_outer_bottom = self.T_outer[:, current_seg] - 273.15
                part_outer_avg = np.mean(self.T_outer[:, current_seg:end_seg], axis=1) - 273.15
                part_outer_top = self.T_outer[:, end_seg-1] - 273.15
                
                outer_color = plt.cm.RdPu(color_intensities[part_idx])
                ax.plot(time_minutes, part_outer_bottom, 
                        label=f'P{part_idx+1} Out Bottom', 
                        color=outer_color, linestyle='-', linewidth=1.8, alpha=1.0)
                ax.plot(time_minutes, part_outer_avg, 
                        label=f'P{part_idx+1} Out Avg', 
                        color=outer_color, linestyle='--', linewidth=1.8, alpha=0.7)
                ax.plot(time_minutes, part_outer_top, 
                        label=f'P{part_idx+1} Out Top', 
                        color=outer_color, linestyle=':', linewidth=1.8, alpha=0.5)
            else:
                # Single wall temperatures
                part_wall_bottom = self.T_wall[:, current_seg] - 273.15
                part_wall_avg = np.mean(self.T_wall[:, current_seg:end_seg], axis=1) - 273.15
                part_wall_top = self.T_wall[:, end_seg-1] - 273.15
                
                wall_color = plt.cm.Greens(color_intensities[part_idx])
                ax.plot(time_minutes, part_wall_bottom, 
                        label=f'P{part_idx+1} Wall Bottom', 
                        color=wall_color, linestyle='-', linewidth=2, alpha=1.0)
                ax.plot(time_minutes, part_wall_avg, 
                        label=f'P{part_idx+1} Wall Avg', 
                        color=wall_color, linestyle='--', linewidth=2, alpha=0.7)
                ax.plot(time_minutes, part_wall_top, 
                        label=f'P{part_idx+1} Wall Top', 
                        color=wall_color, linestyle=':', linewidth=2, alpha=0.5)
            
            current_seg = end_seg
        
        # Format axes
        ax.set_xlabel('Time (minutes)', fontsize=15)
        ax.set_ylabel('Temperature (°C)', fontsize=15)
        ax.set_title('Temperature Evolution by Part (Bottom/Average/Top)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add ambient temperature reference
        ax.axhline(y=self.parts[0].T_ambient_K - 273.15, color='gray', 
                   linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Legend with multiple columns
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        
        # Add color scheme and line style explanation
        info_text = (
            "Color Scheme:\n"
            "• Inlet: Black\n"
            "• Gas: Reds\n"
            "• Single Wall: Greens\n"
            "• Triple Inner: Blues\n"
            "• Triple Middle: Purples\n"
            "• Triple Outer: Pinks\n"
            "\n"
            "Line Styles:\n"
            "• Solid: Bottom segment\n"
            "• Dashed: Average\n"
            "• Dotted: Top segment\n"
            "\n"
            "Alpha:\n"
            "• 1.0: Bottom (darkest)\n"
            "• 0.7: Average\n"
            "• 0.5: Top (lightest)"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done
    def plot_draft(self):
        """Plot draft-related parameters: buoyancy pressure, mass flow rate, and velocity"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Convert time to minutes
        time_minutes = self.time_array / 60.0
        
        # 1. Calculate and plot total buoyancy pressure over time
        total_buoyancy = np.zeros(self.time_steps + 1)
        
        for step in range(self.time_steps + 1):
            buoyancy_sum = 0.0
            for seg in range(self.total_segments):
                part_idx, local_seg, part = self.get_segment_info(seg)
                segment_length = part.height_m / part.num_segments
                
                # Only vertical segments contribute to buoyancy
                if part.orientation.lower() == 'vertical':
                    rho_gas = self.density[step, seg]
                    rho_ambient = self.physics.air_density(part.T_ambient_K, part.Pa)
                    
                    # Buoyancy pressure: g * height * (rho_ambient - rho_gas)
                    dp_buoyancy = 9.81 * segment_length * (rho_ambient - rho_gas)
                    buoyancy_sum += dp_buoyancy
            
            total_buoyancy[step] = buoyancy_sum
        
        ax1.plot(time_minutes, total_buoyancy, 'b-', linewidth=2.5, label='Total Buoyancy')
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_ylabel('Buoyancy Pressure (Pa)', fontsize=14)
        ax1.set_title('Draft Analysis', fontsize=18)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # 2. Plot mass flow rate
        ax2.plot(time_minutes, self.mass_flow_rate, 'r-', linewidth=2.5, label='Mass Flow Rate')
        ax2.set_ylabel('Mass Flow Rate (kg/s)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # 3. Plot velocity profile for different segments
        # Use color scheme similar to temperature plots
        current_seg = 0
        n_parts = len(self.parts)
        color_intensities = np.linspace(0.3, 0.9, n_parts)
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            
            # Plot velocity for first and last segment of each part
            vel_color = plt.cm.Blues(color_intensities[part_idx])
            
            # Bottom segment
            vel_bottom = self.velocity[:, current_seg]
            ax3.plot(time_minutes, vel_bottom, 
                    label=f'Part {part_idx+1} Bottom', 
                    color=vel_color, linestyle='-', linewidth=2, alpha=1.0)
            
            # Top segment
            if part.num_segments > 1:
                vel_top = self.velocity[:, end_seg-1]
                ax3.plot(time_minutes, vel_top, 
                        label=f'Part {part_idx+1} Top', 
                        color=vel_color, linestyle=':', linewidth=2, alpha=0.6)
            
            current_seg = end_seg
        
        ax3.set_xlabel('Time (minutes)', fontsize=14)
        ax3.set_ylabel('Velocity (m/s)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='both', which='major', labelsize=12)
        
        # Add text box with key insights
        # Calculate some key metrics
        max_buoyancy = np.max(total_buoyancy)
        final_buoyancy = total_buoyancy[-1]
        max_flow = np.max(self.mass_flow_rate)
        final_flow = self.mass_flow_rate[-1]
        
        info_text = (
            f"Key Metrics:\n"
            f"• Max buoyancy: {max_buoyancy:.1f} Pa\n"
            f"• Final buoyancy: {final_buoyancy:.1f} Pa\n"
            f"• Max flow rate: {max_flow:.4f} kg/s\n"
            f"• Final flow rate: {final_flow:.4f} kg/s\n"
            f"• Stack height: {sum(p.height_m for p in self.parts):.1f} m"
        )
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done
    def plot_buoyancy_pressure(self):
        """Plot total buoyancy pressure and mass flow rate over time with detailed analysis"""
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))

        # Convert time to minutes
        time_minutes = self.time_array / 60.0

        # Calculate total buoyancy pressure and per-part contributions
        total_buoyancy = np.zeros(self.time_steps + 1)
        part_buoyancy = np.zeros((self.time_steps + 1, len(self.parts)))

        for step in range(self.time_steps + 1):
            current_seg = 0

            for part_idx, part in enumerate(self.parts):
                end_seg = current_seg + part.num_segments
                part_buoyancy_sum = 0.0

                for seg in range(current_seg, end_seg):
                    segment_length = part.height_m / part.num_segments

                    # Only vertical segments contribute to buoyancy
                    if part.orientation.lower() == 'vertical':
                        rho_gas = self.density[step, seg]
                        rho_ambient = self.physics.air_density(part.T_ambient_K, part.Pa)

                        # Buoyancy pressure: g * height * (rho_ambient - rho_gas)
                        dp_buoyancy = 9.81 * segment_length * (rho_ambient - rho_gas)
                        part_buoyancy_sum += dp_buoyancy

                part_buoyancy[step, part_idx] = part_buoyancy_sum
                total_buoyancy[step] += part_buoyancy_sum
                current_seg = end_seg

        # Plot total buoyancy on primary axis
        line1, = ax1.plot(time_minutes, total_buoyancy, 'k-', linewidth=3.5,
                label='Total Buoyancy', zorder=10)

        # Plot per-part contributions
        n_parts = len(self.parts)
        color_intensities = np.linspace(0.4, 0.9, n_parts)

        for part_idx, part in enumerate(self.parts):
            if part.orientation.lower() == 'vertical':
                color = plt.cm.Blues(color_intensities[part_idx])
                height_ft = round(part.height_ft,2)
                ax1.plot(time_minutes, part_buoyancy[:, part_idx],
                       label=f'Part {part_idx+1} ({height_ft} ft, {part.orientation})',
                       color=color, linestyle='--', linewidth=2, alpha=0.8)

        # Add zero reference line
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

        # Format primary axis (buoyancy)
        ax1.set_xlabel('Time (min)', fontsize=16)
        ax1.set_ylabel('Buoyancy pressure (Pa)', fontsize=16, color='black')
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)

        # Create secondary y-axis for mass flow rate
        ax2 = ax1.twinx()

        # Convert mass flow to g/s for better readability
        mass_flow_gs = self.mass_flow_rate * 1000  # kg/s to g/s

        line2, = ax2.plot(time_minutes, mass_flow_gs, 'r-', linewidth=2.5,
                         label='Mass Flow Rate', alpha=0.8)

        ax2.set_ylabel('Mass Flow Rate (g/s)', fontsize=16, color='red')
        ax2.tick_params(axis='y', labelcolor='red', labelsize=14)

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')

        #ax1.set_title('Buoyancy Pressure & Mass Flow Rate Development', fontsize=20)

        # Add annotations for key points
        # Find time when buoyancy reaches 90% of final value
        final_buoyancy = total_buoyancy[-1]
        if final_buoyancy > 0:
            idx_90 = np.where(total_buoyancy >= 0.9 * final_buoyancy)[0]
            if len(idx_90) > 0:
                time_90 = time_minutes[idx_90[0]]
                ax1.axvline(x=time_90, color='green', linestyle=':', alpha=0.5)
                ax1.text(time_90, ax1.get_ylim()[1]*0.95, f'90% at {time_90:.1f} min',
                       ha='center', va='top', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Add statistics box
        max_buoyancy = np.max(total_buoyancy)
        final_flow_gs = mass_flow_gs[-1]
        max_flow_gs = np.max(mass_flow_gs)

        # Stack information
        total_height = sum(p.height_m for p in self.parts if p.orientation.lower() == 'vertical')

        info_text = (
            f"Buoyancy Statistics:\n"
            f"• Maximum: {max_buoyancy:.1f} Pa\n"
            f"• Final: {final_buoyancy:.1f} Pa\n"
            f"\n"
            f"Flow Rate:\n"
            f"• Maximum: {max_flow_gs:.2f} g/s\n"
            f"• Final: {final_flow_gs:.2f} g/s\n"
            f"\n"
            f"Stack Info:\n"
            f"• Height: {total_height:.1f} m ({total_height/0.3048:.0f} ft)"
        )

        ax1.text(0.98, 0.02, info_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        plt.show()

    def plot_available_draft(self):
        """
        Plot available draft (buoyancy minus losses) over time.

        Uses consistent accounting with the momentum solver:
        - dp_drive (buoyancy) is the driving pressure
        - dp_losses = R_total × ṁ² includes friction, minor, and inlet restriction

        Shows:
        - Total buoyancy (gross draft from stack effect)
        - Pressure losses (friction + minor + inlet restriction)
        - Available draft (net draft = buoyancy - losses)
        - Mass flow rate on secondary axis
        """
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

        # Convert time to minutes
        time_minutes = self.time_array / 60.0
        n_steps = self.time_steps + 1

        # Initialize arrays
        total_buoyancy = np.zeros(n_steps)
        total_friction_loss = np.zeros(n_steps)
        total_minor_loss = np.zeros(n_steps)
        total_inlet_loss = np.zeros(n_steps)
        available_draft = np.zeros(n_steps)

        # Calculate buoyancy and losses for each timestep
        for step in range(n_steps):
            current_seg = 0
            step_buoyancy = 0.0
            step_friction = 0.0
            step_minor = 0.0

            for part_idx, part in enumerate(self.parts):
                end_seg = current_seg + part.num_segments
                segment_length = part.height_m / part.num_segments
                diameter = part.inner_diameter_m

                for seg in range(current_seg, end_seg):
                    local_seg = seg - current_seg

                    # Gas properties
                    T_gas = self.T_gas[step, seg]
                    rho_gas = self.density[step, seg]
                    rho_ambient = self.physics.air_density(part.T_ambient_K, part.Pa)

                    # Buoyancy (only vertical segments contribute)
                    if part.orientation.lower() == 'vertical':
                        dp_buoyancy = 9.81 * segment_length * (rho_ambient - rho_gas)
                        step_buoyancy += dp_buoyancy

                    # Friction and minor losses (need flow)
                    if step > 0:
                        velocity = self.velocity[step, seg]

                        if abs(velocity) > 0.01:
                            # Friction factor
                            mu = self.physics.air_viscosity(T_gas)
                            Re = rho_gas * abs(velocity) * diameter / mu
                            Re = max(Re, 100)

                            relative_roughness = part.roughness_m / diameter
                            if Re < 2300:
                                f = 64 / Re
                            elif relative_roughness < 1e-6:
                                f = 0.316 / Re**0.25
                            else:
                                f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2

                            # Friction loss: f * (L/D) * (rho * v^2 / 2)
                            dp_friction = f * (segment_length / diameter) * 0.5 * rho_gas * velocity**2
                            step_friction += dp_friction

                            # Minor losses (fittings) - use actual elbow K values (Feb 2 fix)
                            elbow_key = (part_idx, local_seg)
                            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                                for elbow_spec in self.segment_elbows[elbow_key]:
                                    K_fitting = base_K(elbow_spec) * elbow_spec.vane_factor
                                    dp_minor = K_fitting * 0.5 * rho_gas * velocity**2
                                    step_minor += dp_minor

                current_seg = end_seg

            # Inlet restriction loss: dp_door = R_door × ṁ²
            # R_door = K / (2 × ρ_inlet × A_eff²)
            step_inlet = 0.0
            if self.inlet_restriction.enable and step > 0:
                rho_inlet = self.density[step, 0]
                A_eff = self.inlet_restriction.effective_area_m2
                K = self.inlet_restriction.K_equiv
                mdot = self.mass_flow_rate[step]
                R_door = K / (2.0 * rho_inlet * A_eff**2)
                step_inlet = R_door * mdot**2

            total_buoyancy[step] = step_buoyancy
            total_friction_loss[step] = step_friction
            total_minor_loss[step] = step_minor
            total_inlet_loss[step] = step_inlet
            available_draft[step] = step_buoyancy - step_friction - step_minor - step_inlet

        # Total losses for plotting
        total_losses = total_friction_loss + total_minor_loss + total_inlet_loss

        # Plot draft components
        line_buoy, = ax1.plot(time_minutes, total_buoyancy, 'b-', linewidth=2.5,
                              label='Buoyancy (Gross Draft)', alpha=0.9)
        line_loss, = ax1.plot(time_minutes, total_losses, 'r--', linewidth=2,
                              label='Total Losses', alpha=0.8)
        line_avail, = ax1.plot(time_minutes, available_draft, 'g-', linewidth=3,
                               label='Available Draft (Net)', zorder=10)

        # Fill between buoyancy and available draft to show losses
        ax1.fill_between(time_minutes, available_draft, total_buoyancy,
                        alpha=0.2, color='red', label='_nolegend_')

        # Zero reference line
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

        # Format primary axis
        ax1.set_xlabel('Time (min)', fontsize=14)
        ax1.set_ylabel('Pressure (Pa)', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(True, alpha=0.3)

        # Secondary axis for mass flow rate
        ax2 = ax1.twinx()
        mass_flow_gs = self.mass_flow_rate * 1000  # kg/s to g/s
        line_flow, = ax2.plot(time_minutes, mass_flow_gs, 'k-', linewidth=1.5,
                              label='Mass Flow Rate', alpha=0.6)
        ax2.set_ylabel('Mass Flow Rate (g/s)', fontsize=14, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=12)

        # Combined legend
        lines = [line_buoy, line_loss, line_avail, line_flow]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=11, loc='upper left')

        ax1.set_title('Available Draft Analysis', fontsize=16)

        # Statistics box
        final_buoyancy = total_buoyancy[-1]
        final_losses = total_losses[-1]
        final_available = available_draft[-1]
        final_friction = total_friction_loss[-1]
        final_minor = total_minor_loss[-1]
        final_inlet = total_inlet_loss[-1]

        # Draft efficiency
        if final_buoyancy > 0:
            efficiency = (final_available / final_buoyancy) * 100
        else:
            efficiency = 0

        # Stack height
        total_height = sum(p.height_m for p in self.parts if p.orientation.lower() == 'vertical')

        # Build info text
        info_lines = [
            f"Final Values:",
            f"• Buoyancy: {final_buoyancy:.1f} Pa",
            f"• Friction Loss: {final_friction:.2f} Pa",
            f"• Minor Loss: {final_minor:.2f} Pa",
        ]
        if self.inlet_restriction.enable:
            info_lines.append(f"• Inlet Loss: {final_inlet:.2f} Pa")
        info_lines.extend([
            f"• Available Draft: {final_available:.1f} Pa",
            f"• Efficiency: {efficiency:.0f}%",
            f"",
            f"Stack Height: {total_height:.2f} m"
        ])
        info_text = "\n".join(info_lines)

        ax1.text(0.98, 0.02, info_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        plt.show()

        # Return key values for programmatic access
        return {
            'final_buoyancy_Pa': final_buoyancy,
            'final_losses_Pa': final_losses,
            'final_available_draft_Pa': final_available,
            'final_inlet_loss_Pa': final_inlet,
            'efficiency_percent': efficiency,
            'time_minutes': time_minutes,
            'buoyancy': total_buoyancy,
            'losses': total_losses,
            'inlet_loss': total_inlet_loss,
            'available_draft': available_draft
        }

    def get_stack_differential_pressure(self, timestep=-1, measurement_height_m=None):
        """
        Calculate differential pressure profile along the stack.

        This is what a field technician measures when they connect a manometer
        between the stack interior and outdoor ambient at ground level.

        ΔP(z) = P_inside(z) - P_ambient_at_reference

        Args:
            timestep: Which timestep to analyze (-1 for final)
            measurement_height_m: Height at which to report ΔP (None = inlet)

        Returns:
            dict with:
                - height_m: array of heights
                - dP_Pa: differential pressure profile in Pa
                - dP_inWC: differential pressure profile in inches water column
                - dP_inlet_Pa: ΔP at stack inlet
                - dP_inlet_inWC: ΔP at inlet in inches WC
                - dP_at_height_Pa: ΔP at specified measurement height
                - dP_at_height_inWC: ΔP at measurement height in inches WC
        """
        if timestep == -1:
            timestep = self.time_steps

        n_segs = self.total_segments

        # Build height array (cumulative from bottom)
        heights = np.zeros(n_segs + 1)  # Include both inlet and exit
        seg_heights = np.zeros(n_segs)  # Height at segment center

        cumulative_height = 0.0
        for seg in range(n_segs):
            part_idx, local_seg, part = self.get_segment_info(seg)
            seg_length = self.segment_lengths[seg]

            heights[seg] = cumulative_height
            seg_heights[seg] = cumulative_height + seg_length / 2
            cumulative_height += seg_length
        heights[n_segs] = cumulative_height  # Top of stack

        # Calculate buoyancy and friction for each segment
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

            # Buoyancy (only vertical segments)
            if part.orientation.lower() == 'vertical':
                seg_buoyancy[seg] = 9.81 * seg_length * (rho_ambient - rho_gas)

            # Friction
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

        # Calculate ΔP profile (integrate from top down)
        # At exit: ΔP ≈ 0 (connects to atmosphere)
        # Moving down: ΔP = -buoyancy_above + friction_above
        dP_profile = np.zeros(n_segs + 1)
        dP_profile[n_segs] = 0.0  # Exit at atmospheric

        buoyancy_above = 0.0
        friction_above = 0.0

        for seg in range(n_segs - 1, -1, -1):
            buoyancy_above += seg_buoyancy[seg]
            friction_above += seg_friction[seg]
            dP_profile[seg] = -buoyancy_above + friction_above

        # Convert to inches water column (1 inWC = 249.09 Pa)
        PA_PER_INWC = 249.09
        dP_inWC = dP_profile / PA_PER_INWC

        # Find ΔP at specified measurement height
        if measurement_height_m is None:
            measurement_height_m = 0.0

        # Interpolate to find ΔP at measurement height
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
        """
        Compare simulated differential pressure to a field measurement.

        Args:
            measured_dP_inWC: Measured differential pressure in inches water column
                             (negative = suction, typical for operating chimney)
            measurement_height_m: Height where measurement was taken (0 = inlet)
            timestep: Which simulation timestep to compare (-1 = final)
        """
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

        # Assessment
        print("\n" + "-"*65)
        if abs(diff_inWC) < 0.005:
            print("[OK] Excellent agreement (within 0.005 inWC)")
        elif abs(diff_inWC) < 0.01:
            print("[OK] Good agreement (within 0.01 inWC)")
        elif abs(diff_inWC) < 0.02:
            print("~ Fair agreement (within 0.02 inWC)")
        else:
            print("[X] Significant difference - check inputs or measurement location")

        # Additional context
        print(f"\nInlet dP: {result['dP_inlet_inWC']:.4f} inWC ({result['dP_inlet_Pa']:.2f} Pa)")

        total_buoyancy = sum(result['seg_buoyancy'])
        total_friction = sum(result['seg_friction'])
        print(f"Total buoyancy: {total_buoyancy:.2f} Pa")
        print(f"Total friction: {total_friction:.2f} Pa")
        print("="*60)

    def get_static_pressure_profile(self, timestep=-1):
        """
        Calculate static pressure profile along the stack.

        Static pressure is the local gas pressure at each height.
        P_gauge(z) = P_inside(z) - P_ambient(z)

        For a hot chimney with good draft, P_gauge is negative (suction)
        throughout the stack.

        Args:
            timestep: Which timestep to analyze (-1 for final)

        Returns:
            dict with:
                - height_m: array of heights at segment boundaries
                - P_gauge_Pa: gauge pressure profile (relative to local ambient)
                - P_absolute_Pa: absolute pressure profile
                - P_ambient_Pa: ambient pressure at each height
        """
        if timestep == -1:
            timestep = self.time_steps

        n_segs = self.total_segments

        # Build height array
        heights = np.zeros(n_segs + 1)
        cumulative_height = 0.0

        for seg in range(n_segs):
            part_idx, local_seg, part = self.get_segment_info(seg)
            seg_length = self.segment_lengths[seg]
            heights[seg] = cumulative_height
            cumulative_height += seg_length
        heights[n_segs] = cumulative_height

        # Get ambient pressure at each height (hydrostatic decrease with altitude)
        # P(z) = P_0 * exp(-g*M*z / (R*T)) ≈ P_0 - rho_air * g * z for small z
        P_ambient_base = self.parts[0].Pa
        T_ambient_avg = np.mean([p.T_ambient_K for p in self.parts])
        rho_ambient_avg = self.physics.air_density(T_ambient_avg, P_ambient_base)

        P_ambient = P_ambient_base - rho_ambient_avg * 9.81 * heights

        # Calculate pressure inside the stack
        # Start from exit (top) where P_inside = P_ambient + wind_cap_effect
        P_inside = np.zeros(n_segs + 1)
        deltaP_wind = getattr(self, '_last_deltaP_wind_cap', 0.0)
        P_inside[n_segs] = P_ambient[n_segs] + deltaP_wind

        # Work down from top
        for seg in range(n_segs - 1, -1, -1):
            part_idx, local_seg, part = self.get_segment_info(seg)
            seg_length = self.segment_lengths[seg]
            diameter = part.inner_diameter_m

            T_gas = self.T_gas[timestep, seg]
            rho_gas = self.density[timestep, seg]
            velocity = self.velocity[timestep, seg]

            # Pressure change across this segment
            # dP = -rho_gas * g * dz (hydrostatic) + friction_loss
            dP_hydrostatic = -rho_gas * 9.81 * seg_length

            # Friction loss
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

            # Fitting losses - add to pressure drop
            dP_fitting = 0.0
            elbow_key = (part_idx, local_seg)
            if hasattr(self, 'segment_elbows') and elbow_key in self.segment_elbows:
                for elbow_spec in self.segment_elbows[elbow_key]:
                    K_fitting = base_K(elbow_spec) * elbow_spec.vane_factor
                    dP_fitting += K_fitting * 0.5 * rho_gas * velocity**2

            # Going from top to bottom: P_bottom = P_top - dP_hydrostatic + dP_friction + dP_fitting
            # (Hydrostatic makes pressure higher at bottom, friction/fitting losses reduce it)
            P_inside[seg] = P_inside[seg + 1] - dP_hydrostatic + dP_friction + dP_fitting

        # Gauge pressure
        P_gauge = P_inside - P_ambient

        return {
            'height_m': heights,
            'P_gauge_Pa': P_gauge,
            'P_absolute_Pa': P_inside,
            'P_ambient_Pa': P_ambient
        }

    def plot_pressure_profile(self, timestep=-1):
        """
        Plot static and differential pressure profiles along the stack.

        Creates a two-panel figure:
        - Left: Static pressure (gauge) vs height
        - Right: Differential pressure vs height

        Args:
            timestep: Which timestep to plot (-1 for final)
        """
        static = self.get_static_pressure_profile(timestep)
        diff = self.get_stack_differential_pressure(timestep)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

        # Left panel: Static pressure
        ax1.plot(static['P_gauge_Pa'], static['height_m'], 'b-', linewidth=2, label='Static Pressure')
        ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Static Pressure (Pa)', fontsize=12)
        ax1.set_ylabel('Height (m)', fontsize=12)
        ax1.set_title('Static Pressure Profile', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add secondary x-axis for inWC
        ax1_top = ax1.twiny()
        x_min, x_max = ax1.get_xlim()
        ax1_top.set_xlim(x_min / 249.09, x_max / 249.09)
        ax1_top.set_xlabel('Static Pressure (inWC)', fontsize=10, color='gray')
        ax1_top.tick_params(axis='x', labelcolor='gray')

        # Right panel: Differential pressure
        ax2.plot(diff['dP_Pa'], diff['height_m'], 'r-', linewidth=2, label='ΔP (vs ambient at ground)')
        ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Differential Pressure (Pa)', fontsize=12)
        ax2.set_ylabel('Height (m)', fontsize=12)
        ax2.set_title('Differential Pressure Profile', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add secondary x-axis for inWC
        ax2_top = ax2.twiny()
        x_min, x_max = ax2.get_xlim()
        ax2_top.set_xlim(x_min / 249.09, x_max / 249.09)
        ax2_top.set_xlabel('Differential Pressure (inWC)', fontsize=10, color='gray')
        ax2_top.tick_params(axis='x', labelcolor='gray')

        # Add annotations
        total_height = static['height_m'][-1]
        inlet_gauge = static['P_gauge_Pa'][0]
        inlet_diff = diff['dP_Pa'][0]

        info_text = (
            f"Stack height: {total_height:.2f} m\n"
            f"Inlet gauge P: {inlet_gauge:.2f} Pa ({inlet_gauge/249.09:.4f} inWC)\n"
            f"Inlet ΔP: {inlet_diff:.2f} Pa ({inlet_diff/249.09:.4f} inWC)"
        )

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()

        return {
            'static': static,
            'differential': diff
        }

    def plot_standard_flow_rate(self, T_std_K=273.15, P_std_Pa=101325.0):
        """
        Plot standard volumetric flow rate over time.

        Standard flow rate is the volumetric flow rate corrected to standard
        conditions (default: 0°C, 1 atm). This is useful for comparing flow
        rates across different operating conditions.

        Q_std = ṁ / ρ_std

        where ρ_std = P_std / (R × T_std)

        Args:
            T_std_K: Standard temperature in Kelvin (default: 273.15 K = 0°C)
            P_std_Pa: Standard pressure in Pascals (default: 101325 Pa = 1 atm)

        Returns:
            dict with time and flow rate arrays
        """
        # Calculate standard density
        R_AIR = 287.0  # J/(kg·K)
        rho_std = P_std_Pa / (R_AIR * T_std_K)

        # Convert mass flow to standard volumetric flow
        # Q_std (m³/s) = ṁ (kg/s) / ρ_std (kg/m³)
        Q_std_m3s = self.mass_flow_rate / rho_std

        # Convert to m³/hr
        Q_std_m3hr = Q_std_m3s * 3600.0

        # Time in minutes
        time_minutes = self.time_array / 60.0

        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot standard flow rate
        ax1.plot(time_minutes, Q_std_m3hr, 'b-', linewidth=2, label='Standard Flow Rate')
        ax1.set_xlabel('Time (min)', fontsize=12)
        ax1.set_ylabel('Standard Flow Rate (m³/hr)', fontsize=12)
        ax1.set_title('Standard Volumetric Flow Rate', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Add secondary y-axis for CFM (cubic feet per minute)
        ax2 = ax1.twinx()
        Q_cfm = Q_std_m3hr / 60.0 * 35.3147  # m³/hr to CFM
        ax2.set_ylabel('Standard Flow Rate (CFM)', fontsize=12, color='gray')
        ax2.set_ylim(ax1.get_ylim()[0] / 60.0 * 35.3147, ax1.get_ylim()[1] / 60.0 * 35.3147)
        ax2.tick_params(axis='y', labelcolor='gray')

        # Statistics box
        final_Q = Q_std_m3hr[-1]
        max_Q = np.max(Q_std_m3hr)
        avg_Q = np.mean(Q_std_m3hr[len(Q_std_m3hr)//2:])  # Average of second half

        info_text = (
            f"Standard Conditions:\n"
            f"  T = {T_std_K - 273.15:.1f}°C ({T_std_K:.2f} K)\n"
            f"  P = {P_std_Pa/1000:.1f} kPa\n"
            f"  ρ = {rho_std:.3f} kg/m³\n"
            f"\n"
            f"Flow Rate:\n"
            f"  Final: {final_Q:.2f} m³/hr\n"
            f"  Max: {max_Q:.2f} m³/hr\n"
            f"  Avg (2nd half): {avg_Q:.2f} m³/hr"
        )

        ax1.text(0.98, 0.02, info_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                family='monospace')

        ax1.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

        return {
            'time_minutes': time_minutes,
            'Q_std_m3hr': Q_std_m3hr,
            'Q_std_m3s': Q_std_m3s,
            'rho_std_kg_m3': rho_std,
            'T_std_K': T_std_K,
            'P_std_Pa': P_std_Pa
        }

    def plot_door_opening_and_flow(self, name=None):
        """Plot door opening percentage and mass flow rate over time

        Shows door opening (%) on left y-axis and mass flow (kg/s) on right y-axis.
        Useful for visualizing how door changes affect flow rate.

        Args:
            name: Optional filename to save the figure
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        time_minutes = self.time_array / 60.0

        # Left axis: Door opening percentage
        color1 = 'tab:blue'
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('Door Opening (%)', fontsize=12, color=color1)
        ax1.plot(time_minutes, self.door_opening_history, color=color1, linewidth=2,
                 label='Door Opening')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(-5, 105)
        ax1.set_xlim(0, time_minutes[-1])

        # Fill region to show door state
        ax1.fill_between(time_minutes, 0, self.door_opening_history,
                         alpha=0.2, color=color1)

        # Right axis: Mass flow rate
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Mass Flow Rate (kg/s)', fontsize=12, color=color2)
        ax2.plot(time_minutes, self.mass_flow_rate, color=color2, linewidth=2,
                 linestyle='--', label='Mass Flow')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Set y-limits for mass flow (add some padding)
        max_flow = np.max(self.mass_flow_rate) * 1.1
        min_flow = min(0, np.min(self.mass_flow_rate) * 1.1)
        ax2.set_ylim(min_flow, max_flow)

        # Title and grid
        title = 'Door Opening and Mass Flow Rate Over Time'
        if self.door_opening_func is not None:
            title += ' (Dynamic Door Control)'
        plt.title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Add statistics box
        final_opening = self.door_opening_history[-1]
        final_flow = self.mass_flow_rate[-1]
        min_opening = np.min(self.door_opening_history)
        max_opening = np.max(self.door_opening_history)

        info_text = (
            f"Door Opening:\n"
            f"  Final: {final_opening:.0f}%\n"
            f"  Range: {min_opening:.0f}% - {max_opening:.0f}%\n"
            f"\n"
            f"Mass Flow:\n"
            f"  Final: {final_flow:.4f} kg/s\n"
            f"  Max: {np.max(self.mass_flow_rate):.4f} kg/s"
        )

        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                family='monospace')

        plt.tight_layout()
        if name:
            plt.savefig(name)
        plt.show()

    def print_flow_rate(self, interval_seconds: float = None):
        """Print mass flow rate over time

        Args:
            interval_seconds: Time interval between printed values (seconds).
                             If None, prints at ~10 evenly spaced points plus start/end.
        """
        print("\n" + "="*70)
        print("MASS FLOW RATE OVER TIME")
        print("="*70)

        # Determine which timesteps to print
        if interval_seconds is None:
            # Print ~10 evenly spaced points
            n_points = min(12, self.time_steps + 1)
            indices = np.linspace(0, self.time_steps, n_points, dtype=int)
        else:
            # Print at specified interval
            steps_per_interval = max(1, int(interval_seconds / self.dt))
            indices = np.arange(0, self.time_steps + 1, steps_per_interval)
            # Always include the last timestep
            if indices[-1] != self.time_steps:
                indices = np.append(indices, self.time_steps)

        # Print header
        print(f"\n{'Time (s)':>10} {'Time (min)':>12} {'Flow (kg/s)':>14} {'Flow (g/s)':>12}")
        print("-" * 50)

        for idx in indices:
            time_s = self.time_array[idx]
            time_min = time_s / 60.0
            flow_kgs = self.mass_flow_rate[idx]
            flow_gs = flow_kgs * 1000

            print(f"{time_s:>10.1f} {time_min:>12.2f} {flow_kgs:>14.6f} {flow_gs:>12.3f}")

        # Print summary statistics
        print("-" * 50)
        print(f"\nSummary Statistics:")
        print(f"  Initial flow rate:  {self.mass_flow_rate[0]*1000:.3f} g/s")
        print(f"  Final flow rate:    {self.mass_flow_rate[-1]*1000:.3f} g/s")
        print(f"  Maximum flow rate:  {np.max(self.mass_flow_rate)*1000:.3f} g/s")
        print(f"  Minimum flow rate:  {np.min(self.mass_flow_rate)*1000:.3f} g/s")
        print(f"  Average flow rate:  {np.mean(self.mass_flow_rate)*1000:.3f} g/s")

        # Check for backdraft (negative flow)
        if np.any(self.mass_flow_rate < 0):
            backdraft_times = self.time_array[self.mass_flow_rate < 0]
            print(f"\n  WARNING: Backdraft detected!")
            print(f"  Negative flow from {backdraft_times[0]:.1f}s to {backdraft_times[-1]:.1f}s")

        print("="*70 + "\n")
    #Done
    def temp_evolution_by_segment(self):
        """Plot temperature evolution for each individual segment"""
        # Create single plot for all temperatures
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        
        # First plot the inlet temperature in black
        T_ambient_inlet = self.parts[0].T_ambient_K
        # Get inlet temperatures (uses CSV if available, otherwise default model)
        inlet_temps = np.array([self._get_inlet_temp(t, T_ambient=T_ambient_inlet)
                                 for t in self.time_array]) - 273.15

        ax.plot(self.time_array, inlet_temps,
                label='Inlet Gas', color='black', linestyle='-', linewidth=3, zorder=10)
        
        current_seg = 0
        
        # Create color intensities for segments
        total_segs = self.total_segments
        seg_intensities = np.linspace(0.3, 0.9, total_segs)
        seg_counter = 0
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            
            for local_seg in range(part.num_segments):
                global_seg = current_seg + local_seg
                intensity = seg_intensities[seg_counter]
                seg_counter += 1
                
                # Plot gas temperature - use Reds colormap
                gas_temps = self.T_gas[:, global_seg] - 273.15
                gas_color = plt.cm.Reds(intensity)
                label_gas = f'P{part_idx+1}S{local_seg+1} Gas'
                ax.plot(self.time_array, gas_temps, 
                       label=label_gas, color=gas_color, 
                       linestyle='-', linewidth=2.5)
                
                # Plot wall temperatures based on type
                if part.type == 'single':
                    wall_temps = self.T_wall[:, global_seg] - 273.15
                    wall_color = plt.cm.Greens(intensity)
                    label_wall = f'P{part_idx+1}S{local_seg+1} Wall'
                    ax.plot(self.time_array, wall_temps,
                           label=label_wall, color=wall_color,
                           linestyle='--', linewidth=2)
                else:  # triple wall
                    # Plot all three layers with appropriate colors
                    liner_temps = self.T_liner[:, global_seg] - 273.15
                    middle_temps = self.T_middle[:, global_seg] - 273.15
                    outer_temps = self.T_outer[:, global_seg] - 273.15
                    
                    # Inner (liner) - Blues
                    liner_color = plt.cm.Blues(intensity)
                    ax.plot(self.time_array, liner_temps,
                           label=f'P{part_idx+1}S{local_seg+1} Inner',
                           color=liner_color, linestyle='--', linewidth=2)
                    
                    # Middle - Purples
                    middle_color = plt.cm.Purples(intensity)
                    ax.plot(self.time_array, middle_temps,
                           label=f'P{part_idx+1}S{local_seg+1} Mid',
                           color=middle_color, linestyle=':', linewidth=1.8)
                    
                    # Outer - RdPu (Pinks)
                    outer_color = plt.cm.RdPu(intensity)
                    ax.plot(self.time_array, outer_temps,
                           label=f'P{part_idx+1}S{local_seg+1} Out',
                           color=outer_color, linestyle='-.', linewidth=1.8)
            
            current_seg = end_seg
        
        # Format plot
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Evolution by Segment (All Components)')
        ax.grid(True, alpha=0.3)
        
        # Add ambient temperature reference
        ax.axhline(y=self.parts[0].T_ambient_K - 273.15, color='gray', 
                   linestyle=':', alpha=0.5, linewidth=1.5)
        ax.text(self.time_array[-1]*0.98, self.parts[0].T_ambient_K - 273.15 + 1, 
                'Ambient', ha='right', va='bottom', color='gray', fontsize=9)
        
        # Create custom legend with multiple columns
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=3)
        
        # Add color scheme explanation
        color_scheme_text = (
            "Color Scheme:\n"
            "• Inlet: Black\n"
            "• Gas: Reds (darker=higher)\n"
            "• Single Wall: Greens\n"
            "• Triple Inner: Blues\n"
            "• Triple Middle: Purples\n"
            "• Triple Outer: Pinks\n"
            "\n"
            "Line Styles:\n"
            "• Solid: Gas\n"
            "• Dashed: Wall/Inner\n"
            "• Dotted: Middle\n"
            "• Dash-dot: Outer"
        )
        ax.text(0.02, 0.98, color_scheme_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Add segment info
        info_text = (
            f"Notation: PxSy = Part x, Segment y\n"
            f"Total segments: {self.total_segments}\n"
            f"Darker colors = higher elevation"
        )
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done    
    def _plot_heat_transfer_coefficients(self):
        """Plot heat transfer coefficient evolution for all segments"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        current_seg = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.parts)))
        
        for part_idx, part in enumerate(self.parts):
            color = colors[part_idx]
            
            for local_seg in range(part.num_segments):
                global_seg = current_seg + local_seg
                
                # Inner heat transfer coefficient
                ax1.plot(self.time_array, self.h_conv_inner[:, global_seg],
                        label=f'Part {part_idx+1} Seg {local_seg+1}',
                        color=color, alpha=0.7 - 0.1*local_seg, linewidth=1.5)
                
                # Outer heat transfer coefficient
                ax2.plot(self.time_array, self.h_conv_outer[:, global_seg],
                        label=f'Part {part_idx+1} Seg {local_seg+1}',
                        color=color, alpha=0.7 - 0.1*local_seg, linewidth=1.5)
            
            current_seg += part.num_segments
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Inner h (W/m²·K)')
        ax1.set_title('Inner Surface Heat Transfer Coefficients')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Outer h (W/m²·K)')
        ax2.set_title('Outer Surface Heat Transfer Coefficients')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    #Done    
    def _print_detailed_segment_info(self):
        """Print detailed temperature information for each segment"""
        print("\n" + "="*80)
        print("DETAILED SEGMENT INFORMATION")
        print("="*80)
        
        current_seg = 0
        for part_idx, part in enumerate(self.parts):
            print(f"\nPart {part_idx+1} ({part.type.upper()} WALL):")
            print(f"  Segments: {current_seg+1} to {current_seg+part.num_segments}")
            print(f"  Height: {part.height_m:.2f}m")
            print(f"  Diameter: {part.inner_diameter_m*1000:.1f}mm")
            
            end_seg = current_seg + part.num_segments
            final_gas_temps = self.T_gas[-1, current_seg:end_seg] - 273.15
            
            print(f"  Final Gas Temperatures (°C): {final_gas_temps}")
            
            if part.type == 'triple':
                final_liner_temps = self.T_liner[-1, current_seg:end_seg] - 273.15
                final_middle_temps = self.T_middle[-1, current_seg:end_seg] - 273.15
                final_outer_temps = self.T_outer[-1, current_seg:end_seg] - 273.15
                
                print(f"  Final Liner Temperatures (°C): {final_liner_temps}")
                print(f"  Final Middle Temperatures (°C): {final_middle_temps}")
                print(f"  Final Outer Temperatures (°C): {final_outer_temps}")
                
                # Calculate and show insulation performance
                avg_liner = np.mean(final_liner_temps)
                avg_outer = np.mean(final_outer_temps)
                temp_reduction = avg_liner - avg_outer
                print(f"  Temperature Reduction (Liner to Outer): {temp_reduction:.1f}°C")
                print(f"  Insulation Effectiveness: {temp_reduction/avg_liner*100:.1f}%")
            else:
                final_wall_temps = self.T_wall[-1, current_seg:end_seg] - 273.15
                print(f"  Final Wall Temperatures (°C): {final_wall_temps}")
            
            current_seg = end_seg
    #Done
    def plot_chimney_snapshot(self, timestep=-1, figsize=(10, 8), wall_scale=5.0, 
                             save_path=None):
        """Plot a single snapshot of the chimney geometry with temperatures
        
        Args:
            timestep: Timestep index to plot (default -1 for final state)
            figsize: Figure size (width, height)
            wall_scale: Factor to scale wall thickness for visibility
            save_path: Optional path to save the figure
        """
        return plot_chimney_snapshot(self, timestep, figsize, wall_scale, save_path)