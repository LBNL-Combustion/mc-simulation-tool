"""
Enhanced friction factor calculations for corrugated pipes and seam effects

MODEL SEPARATION (v3.0):
-----------------------
This module provides two distinct models for different pipe geometries:

1. SPIRAL-SEAM DUCT (SpiralCorrugationProperties):
   - Intended for: rigid metal spiral pipe, spiral-wound connectors
   - Loss model: Seam-crossing minor losses (K_total) + optional swirl multipliers
   - Does NOT modify effective roughness via c_spiral*depth (removed in v3.0)
   - Roughness = base_roughness (optionally + seam_protrusion heuristic)

2. ANNULAR CORRUGATIONS (CorrugationProperties):
   - Intended for: corrugated flex hose, annularly ribbed pipes
   - Loss model: C_form(d/p) friction multiplier via CorrugatedPipeFriction
   - Does NOT apply seam-crossing K_total (no discrete seams)

This separation prevents double-counting between distributed friction
enhancement and discrete minor losses.
"""
import numpy as np
import warnings
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class CorrugationProperties:
    """Properties of annular corrugated pipe (e.g., flex hose)

    Use this for pipes with circumferential (annular) corrugations.
    Loss model: C_form friction multiplier based on depth/pitch ratio.
    Does NOT include seam-crossing losses (use SpiralCorrugationProperties for that).
    """
    depth_m: float  # Corrugation depth (amplitude)
    pitch_m: float  # Corrugation pitch (wavelength)
    base_roughness_m: float  # Base material roughness

    @property
    def depth_to_pitch_ratio(self) -> float:
        """Ratio of depth to pitch (affects flow separation)"""
        return self.depth_m / self.pitch_m if self.pitch_m > 0 else 0

@dataclass
class SpiralCorrugationProperties:
    """Properties of spiral-seam duct (rigid metal spiral pipe)

    Use this for spiral-wound metal duct with discrete helical seams.

    Loss model (v3.0):
    - Primary: Seam-crossing minor losses (K_total = n_seams * K_per_crossing + K_lock)
    - Optional: Mild swirl friction/HT multipliers (heuristic, capped)
    - Roughness: base_roughness only (c_spiral*depth removed - not applicable geometry)

    NOTE: This model does NOT apply CorrugatedPipeFriction's C_form(d/p) model.
    Spiral-seam duct is fundamentally different from annular corrugations.
    """
    depth_m: float  # Corrugation depth (valley-to-crest), meters
    pitch_m: float  # Axial distance per helical revolution, meters
    num_starts: int  # Number of parallel spiral seams (1, 4, 5, 6, etc.)
    base_roughness_m: float  # Base material roughness (sheet steel), meters

    # Seam properties (lock seam where metal strip joins)
    has_lock_seam: bool = True
    seam_protrusion_m: float = 0.002  # Seam height above surface, meters (default 2mm)
    seam_width_m: float = 0.008  # Seam width, meters (default 8mm)
    K_seam_per_crossing: float = 0.05  # Loss coefficient per seam crossing (HEURISTIC - uncited)
    K_lock_seam: float = 0.03  # Loss coefficient for longitudinal lock seam (HEURISTIC - uncited)

    # Guardrails to prevent K_total explosion (v3.0)
    K_seams_max: float = 5.0  # Maximum cumulative seam K before clamping

    # Optional: add seam protrusion to roughness (conservative heuristic)
    include_seam_roughness: bool = False  # If True: eps = base + seam_protrusion_m

    def compute_helix_angle(self, diameter_m: float) -> float:
        """Compute helix angle β (radians) from pitch and diameter

        β = arctan(p / (π*D))

        Args:
            diameter_m: Pipe inner diameter in meters

        Returns:
            Helix angle in radians (0 = annular, π/2 = axial)
        """
        return np.arctan(self.pitch_m / (np.pi * diameter_m))

    def compute_helix_angle_deg(self, diameter_m: float) -> float:
        """Compute helix angle in degrees"""
        return np.degrees(self.compute_helix_angle(diameter_m))

    @property
    def depth_to_pitch_ratio(self) -> float:
        """Ratio of depth to pitch"""
        return self.depth_m / self.pitch_m if self.pitch_m > 0 else 0

@dataclass
class SeamProperties:
    """Properties of pipe seams/joints"""
    K_loss: float  # Loss coefficient per seam
    spacing_m: Optional[float]  # Distance between seams

class CorrugatedPipeFriction:
    """
    Calculate enhanced friction factors for corrugated pipes

    Corrugations increase pressure drop through:
    1. Increased effective roughness
    2. Form drag from flow separation
    3. Secondary flow patterns
    """

    def __init__(self):
        self.smooth_pipe_f_min = 0.008  # Minimum friction factor for smooth pipe

    def calculate_corrugated_friction_factor(self,
                                            Re: float,
                                            D: float,
                                            corrugation: CorrugationProperties) -> float:
        """
        Calculate friction factor for corrugated pipe

        Based on empirical correlations from ASHRAE and manufacturer data

        Args:
            Re: Reynolds number
            D: Pipe diameter (m)
            corrugation: Corrugation properties

        Returns:
            Friction factor (dimensionless)
        """
        # Base smooth pipe friction factor (Colebrook)
        f_smooth = self._colebrook_white(Re, corrugation.base_roughness_m / D)

        # Corrugation enhancement factors

        # 1. Roughness enhancement - corrugations act as large-scale roughness
        effective_roughness = corrugation.base_roughness_m + 0.5 * corrugation.depth_m
        f_rough = self._colebrook_white(Re, effective_roughness / D)

        # 2. Form drag enhancement - depends on depth/pitch ratio
        # Based on empirical data for annular corrugations
        d_p_ratio = corrugation.depth_to_pitch_ratio

        if d_p_ratio < 0.1:
            # Shallow corrugations - minimal form drag
            C_form = 1.0 + 2.5 * d_p_ratio
        elif d_p_ratio < 0.3:
            # Moderate corrugations - increasing form drag
            C_form = 1.25 + 5.0 * (d_p_ratio - 0.1)
        else:
            # Deep corrugations - significant form drag
            C_form = 2.25 + 10.0 * (d_p_ratio - 0.3)

        # 3. Reynolds number effect - form drag more significant at high Re
        if Re < 4000:
            # Laminar/transitional - corrugations have less effect
            Re_factor = 0.5 + 0.5 * (Re / 4000)
        else:
            # Turbulent - full corrugation effect
            Re_factor = 1.0

        # 4. Diameter effect - smaller pipes have relatively larger corrugations
        D_factor = min(2.0, (0.1 / D) ** 0.2)  # Enhanced effect for D < 0.1m

        # Combined friction factor
        f_corrugated = f_smooth * C_form * Re_factor * D_factor

        # Limit to reasonable range (based on experimental data)
        f_corrugated = min(f_corrugated, 0.15)  # Max around 0.15
        f_corrugated = max(f_corrugated, f_rough)  # At least as high as rough pipe

        return f_corrugated

    def calculate_seam_losses(self,
                            length: float,
                            velocity: float,
                            seam_props: SeamProperties) -> float:
        """
        Calculate additional pressure drop from seams/joints

        Args:
            length: Pipe length (m)
            velocity: Flow velocity (m/s)
            seam_props: Seam properties

        Returns:
            Additional pressure drop (Pa)
        """
        if seam_props.spacing_m is None or seam_props.spacing_m <= 0:
            return 0.0

        # Number of seams in this length
        n_seams = int(length / seam_props.spacing_m)

        # Pressure drop per seam: ΔP = K × (ρv²/2)
        # (density will be applied by caller)
        dp_per_seam = seam_props.K_loss * (velocity ** 2 / 2)

        return n_seams * dp_per_seam

    def _colebrook_white(self, Re: float, relative_roughness: float) -> float:
        """
        Colebrook-White equation for friction factor

        Args:
            Re: Reynolds number
            relative_roughness: ε/D ratio

        Returns:
            Darcy friction factor
        """
        if Re < 2300:
            # Laminar flow
            return 64 / Re if Re > 0 else 0.1

        # Turbulent flow - iterative solution
        # Initial guess using Swamee-Jain approximation
        term1 = relative_roughness / 3.7
        term2 = 5.74 / (Re ** 0.9)
        f = 0.25 / (np.log10(term1 + term2) ** 2)

        # Refine with iteration
        for _ in range(3):
            term = relative_roughness / 3.7 + 2.51 / (Re * np.sqrt(f))
            f_new = 0.25 / (np.log10(term) ** 2)
            if abs(f_new - f) < 0.0001:
                break
            f = f_new

        return f

    def get_enhancement_factor(self,
                              corrugation: Optional[CorrugationProperties],
                              smooth_f: float) -> float:
        """
        Get the enhancement factor for corrugated vs smooth pipe

        Args:
            corrugation: Corrugation properties (None for smooth)
            smooth_f: Smooth pipe friction factor

        Returns:
            Enhancement factor (>1 for corrugated)
        """
        if corrugation is None:
            return 1.0

        # Typical enhancement factors based on corrugation type
        d_p_ratio = corrugation.depth_to_pitch_ratio

        if d_p_ratio < 0.1:
            return 1.5  # Light corrugation
        elif d_p_ratio < 0.25:
            return 2.0  # Medium corrugation
        else:
            return 2.5  # Heavy corrugation


def compute_spiral_factor(beta_rad: float) -> float:
    """
    Compute spiral correction factor for effective roughness

    *** DEPRECATED in v3.0 for spiral-seam duct ***

    This correlation was adapted from helically ribbed heat-transfer tubes
    and is NOT appropriate for spiral-seam duct. It remains available for:
    - Diagnostic purposes
    - Legacy code compatibility
    - Potential use with annular corrugation models

    For spiral-seam duct, use seam-crossing K losses (spiral_equivalents)
    instead of roughness blending.

    Original physics (for helically ribbed tubes):
    - β = 0° (annular): Maximum form drag, c_spiral ≈ 0.5
    - β = 30° (typical): Reduced form drag, c_spiral ≈ 0.3
    - β = 90° (axial): Minimal form drag, c_spiral ≈ 0.2

    Args:
        beta_rad: Helix angle in radians

    Returns:
        Spiral correction factor c_spiral (dimensionless)
    """
    # Empirical correlation: c_spiral(β) = 0.5 * [1 - 0.6 * sin²(β)]
    # WARNING: This is uncited and not validated for spiral-seam duct
    c_spiral = 0.5 * (1.0 - 0.6 * np.sin(beta_rad)**2)
    return c_spiral


def spiral_descriptors(diameter_m: float, depth_m: float, pitch_m: float) -> Dict[str, float]:
    """
    Compute geometric and dimensionless descriptors for spiral corrugations

    These descriptors characterize the spiral geometry and are used to
    quantify swirl intensity and its effects on friction and heat transfer.

    Args:
        diameter_m: Pipe inner diameter in meters
        depth_m: Corrugation depth (rib height) in meters
        pitch_m: Helical pitch (axial distance per revolution) in meters

    Returns:
        Dictionary containing:
            'beta_rad': Helix angle in radians
            'beta_deg': Helix angle in degrees
            'swirl_index': S = sin(β)*cos(β), quantifies swirl intensity
            'rel_depth': e/D, relative rib depth
            'rel_pitch': p/D, relative pitch
    """
    # Helix angle: β = arctan(p / π*D)
    beta_rad = np.arctan(pitch_m / (np.pi * diameter_m))
    beta_deg = np.degrees(beta_rad)

    # Swirl index: S = sin(β)*cos(β)
    # Maximum at β = 45°, zero at β = 0° or 90°
    swirl_index = np.sin(beta_rad) * np.cos(beta_rad)

    # Relative geometric parameters
    rel_depth = depth_m / diameter_m
    rel_pitch = pitch_m / diameter_m

    return {
        'beta_rad': beta_rad,
        'beta_deg': beta_deg,
        'swirl_index': swirl_index,
        'rel_depth': rel_depth,
        'rel_pitch': rel_pitch
    }


def swirl_friction_multiplier(swirl_index: float, rel_depth: float, rel_pitch: float,
                              Re: float, swirl_friction_cap: float = 0.15,
                              a1: float = 0.25, a2: float = 0.2, a3: float = 0.1) -> float:
    """
    Compute friction multiplier due to swirl-induced secondary flows

    *** HEURISTIC MODEL - COEFFICIENTS ARE UNCITED ***

    This is a phenomenological model with plausible functional form but
    arbitrary coefficients. Use with caution. The cap (default 15%) limits
    the impact of coefficient uncertainty on overall results.

    Physical basis (qualitative):
    - Swirl creates secondary flows normal to main flow direction
    - Increases near-wall velocity gradients
    - Effect scales with swirl intensity (S²) and rib geometry
    - Diminishes as pitch increases (less swirl per length)
    - Weakens at very high Re (inertia dominates)

    Args:
        swirl_index: S = sin(β)*cos(β), dimensionless swirl intensity
        rel_depth: e/D, relative rib depth
        rel_pitch: p/D, relative pitch
        Re: Reynolds number
        swirl_friction_cap: Maximum fractional increase (default 0.15 = 15%)
        a1: Swirl intensity coefficient (default 0.25, HEURISTIC)
        a2: Pitch effect coefficient (default 0.2, HEURISTIC)
        a3: Reynolds number effect coefficient (default 0.1, HEURISTIC)

    Returns:
        Friction multiplier m_f, where 1 <= m_f <= 1 + cap

    Formula:
        m_f = 1 + a1*S²*min(e/D / 0.02, 1)
                  * (1 - a2*min((p/D - 1)³, 0.5))
                  * (1 - a3 / (1 + Re/20000))
    """
    # Depth factor: normalize by 0.02 (typical rib height), cap at 1.0
    depth_factor = min(rel_depth / 0.02, 1.0)

    # Pitch factor: effect diminishes as pitch increases
    # (p/D - 1)³ captures nonlinear decay, capped at 0.5
    pitch_term = max(rel_pitch - 1.0, 0.0)  # No effect if p/D < 1
    pitch_factor = 1.0 - a2 * min(pitch_term**3, 0.5)

    # Reynolds factor: swirl effect weakens at high Re
    re_factor = 1.0 - a3 / (1.0 + Re / 20000.0)

    # Combined multiplier
    enhancement = a1 * swirl_index**2 * depth_factor * pitch_factor * re_factor

    # Apply cap and ensure >= 1.0
    m_f = 1.0 + min(enhancement, swirl_friction_cap)

    return m_f


def spiral_HT_multiplier(swirl_index: float, rel_depth: float, rel_pitch: float,
                        Re: float, HT_enhancement_cap: float = 0.25,
                        b1: float = 0.35, b2: float = 0.25, b3: float = 0.1) -> float:
    """
    Compute heat transfer enhancement multiplier due to swirl

    *** HEURISTIC MODEL - COEFFICIENTS ARE UNCITED ***

    This is a phenomenological model with plausible functional form but
    arbitrary coefficients. Use with caution. The cap (default 25%) limits
    the impact of coefficient uncertainty on overall results.

    Physical basis (qualitative):
    - Disrupting thermal boundary layer
    - Increasing turbulent mixing
    - Creating helical flow that sweeps wall surface

    The enhancement is typically larger than friction increase because
    heat transfer is more sensitive to flow disruption near the wall.

    Args:
        swirl_index: S = sin(β)*cos(β), dimensionless swirl intensity
        rel_depth: e/D, relative rib depth
        rel_pitch: p/D, relative pitch
        Re: Reynolds number
        HT_enhancement_cap: Maximum fractional increase (default 0.25 = 25%)
        b1: Swirl intensity coefficient (default 0.35, HEURISTIC)
        b2: Pitch effect coefficient (default 0.25, HEURISTIC)
        b3: Reynolds number effect coefficient (default 0.1, HEURISTIC)

    Returns:
        Heat transfer multiplier M_h, where 1 <= M_h <= 1 + cap

    Formula:
        M_h = 1 + min(b1*S²*min(e/D / 0.02, 1)
                      * (1 - b2*min((p/D - 1)³, 0.5))
                      * (1 - b3 / (1 + Re/20000)), cap)
    """
    # Depth factor: normalize by 0.02, cap at 1.0
    depth_factor = min(rel_depth / 0.02, 1.0)

    # Pitch factor: same form as friction multiplier
    pitch_term = max(rel_pitch - 1.0, 0.0)
    pitch_factor = 1.0 - b2 * min(pitch_term**3, 0.5)

    # Reynolds factor: enhancement weakens at high Re
    re_factor = 1.0 - b3 / (1.0 + Re / 20000.0)

    # Combined enhancement
    enhancement = b1 * swirl_index**2 * depth_factor * pitch_factor * re_factor

    # Apply cap
    M_h = 1.0 + min(enhancement, HT_enhancement_cap)

    return M_h


def spiral_equivalents(diameter_m: float, length_m: float, Re_ref: float,
                      depth_m: float, pitch_m: float, num_starts: int,
                      base_roughness_m: float = 1e-6,
                      K_seam_per_crossing: float = 0.05,
                      enable_swirl: bool = False,
                      swirl_friction_cap: float = 0.15,
                      enable_HT: bool = False,
                      HT_enhancement_cap: float = 0.25,
                      has_lock_seam: bool = True,
                      K_lock: float = 0.03,
                      K_seams_max: float = 5.0,
                      include_seam_roughness: bool = False,
                      seam_protrusion_m: float = 0.002) -> Dict[str, float]:
    """
    Compute K-losses and heat transfer multiplier for spiral-seam duct

    VERSION 3.0 - MODEL SEPARATION UPDATE:
    This function is for SPIRAL-SEAM DUCT only. It computes:
    - Seam-crossing minor losses (K_total) as the primary loss mechanism
    - Optional swirl friction/HT multipliers (heuristic, capped)
    - Roughness = base_roughness only (c_spiral*depth REMOVED)

    The c_spiral roughness blending was removed because:
    - It was adapted from helically ribbed heat-transfer tubes, not spiral-seam duct
    - It caused double-counting with seam K losses
    - Spiral-seam duct losses are dominated by discrete seam crossings, not
      distributed roughness enhancement

    For annular corrugations (flex hose), use CorrugatedPipeFriction instead.

    Args:
        diameter_m: Pipe inner diameter (m)
        length_m: Segment length (m)
        Re_ref: Representative Reynolds number for calculations
        depth_m: Corrugation depth (m)
        pitch_m: Helical pitch (m) - must be > 0
        num_starts: Number of parallel spiral seams
        base_roughness_m: Base material roughness (m)
        K_seam_per_crossing: Loss coefficient per seam crossing (HEURISTIC)
        enable_swirl: Enable swirl friction enhancement (HEURISTIC)
        swirl_friction_cap: Maximum swirl friction increase (default 0.15)
        enable_HT: Enable heat transfer enhancement (HEURISTIC)
        HT_enhancement_cap: Maximum heat transfer increase (default 0.25)
        has_lock_seam: Include longitudinal lock seam
        K_lock: Lock seam loss coefficient (HEURISTIC)
        K_seams_max: Maximum cumulative seam K before clamping (default 5.0)
        include_seam_roughness: If True, add seam_protrusion to roughness (conservative)
        seam_protrusion_m: Seam protrusion height for optional roughness increment

    Returns:
        Dictionary with:
            'roughness_m': Base roughness (+ seam_protrusion if enabled)
            'K_total': Total minor loss coefficient (clamped to K_seams_max)
            'Mh': Heat transfer multiplier (1.0 if disabled)
            'K_seams_clamped': True if K_seams was clamped
            'debug': Detailed intermediate results

    Raises:
        ValueError: If pitch_m <= 0
    """
    # Input validation
    if pitch_m <= 0:
        raise ValueError(f"pitch_m must be > 0, got {pitch_m}")

    # Compute geometric descriptors
    descriptors = spiral_descriptors(diameter_m, depth_m, pitch_m)

    # v3.0: Roughness = base only (c_spiral*depth REMOVED)
    # Spiral-seam duct losses are discrete (seam crossings), not distributed roughness
    eps_corr = base_roughness_m
    if include_seam_roughness:
        # Optional conservative heuristic: add seam protrusion to roughness
        eps_corr = base_roughness_m + seam_protrusion_m

    # Seam losses with clamping (v3.0 guardrail)
    n_seams = (length_m * num_starts) / pitch_m
    K_seams_raw = n_seams * K_seam_per_crossing
    K_lock_val = K_lock if has_lock_seam else 0.0

    K_seams_clamped = False
    if K_seams_raw > K_seams_max:
        warnings.warn(
            f"Spiral seam K_seams ({K_seams_raw:.2f}) exceeds K_seams_max ({K_seams_max:.2f}). "
            f"Clamping to {K_seams_max:.2f}. Check pitch_m={pitch_m}, num_starts={num_starts}, "
            f"length_m={length_m} for reasonableness.",
            UserWarning
        )
        K_seams = K_seams_max
        K_seams_clamped = True
    else:
        K_seams = K_seams_raw

    K_total = K_seams + K_lock_val

    # v3.0: NO friction factor inversion for spiral-seam duct
    # Losses are accounted for via K_total (minor losses), not roughness
    # Just use base friction factor for any swirl multiplier calculations
    friction_calc = CorrugatedPipeFriction()
    f_base = friction_calc._colebrook_white(Re_ref, eps_corr / diameter_m)

    # Apply swirl friction multiplier if enabled (HEURISTIC)
    m_f = 1.0
    if enable_swirl:
        m_f = swirl_friction_multiplier(
            descriptors['swirl_index'],
            descriptors['rel_depth'],
            descriptors['rel_pitch'],
            Re_ref,
            swirl_friction_cap
        )

    # Heat transfer multiplier if enabled (HEURISTIC)
    M_h = 1.0
    if enable_HT:
        M_h = spiral_HT_multiplier(
            descriptors['swirl_index'],
            descriptors['rel_depth'],
            descriptors['rel_pitch'],
            Re_ref,
            HT_enhancement_cap
        )

    # Assemble results
    results = {
        'roughness_m': eps_corr,
        'K_total': K_total,
        'Mh': M_h,
        'K_seams_clamped': K_seams_clamped,
        'debug': {
            'helix_angle_deg': descriptors['beta_deg'],
            'helix_angle_rad': descriptors['beta_rad'],
            'swirl_index': descriptors['swirl_index'],
            'rel_depth': descriptors['rel_depth'],
            'rel_pitch': descriptors['rel_pitch'],
            # c_spiral retained for diagnostics but NOT used in roughness
            'spiral_factor_diagnostic': compute_spiral_factor(descriptors['beta_rad']),
            'effective_roughness_m': eps_corr,
            'n_seams': n_seams,
            'K_seams_raw': K_seams_raw,
            'K_seams': K_seams,
            'K_lock': K_lock_val,
            'friction_factor_base': f_base,
            'friction_multiplier': m_f,
            'HT_multiplier': M_h,
            'swirl_enabled': enable_swirl,
            'HT_enabled': enable_HT,
            'include_seam_roughness': include_seam_roughness,
            'model_version': '3.0'
        }
    }

    return results


def compute_spiral_corrugation_effects(
    diameter_m: float,
    length_m: float,
    spiral_props: SpiralCorrugationProperties,
    representative_Re: float = 10000,
    enable_swirl: bool = False,
    swirl_friction_cap: float = 0.15,
    enable_HT: bool = False,
    HT_enhancement_cap: float = 0.25
) -> Dict[str, float]:
    """
    Compute K-losses and heat transfer multiplier for spiral-seam duct

    VERSION 3.0 - MODEL SEPARATION UPDATE:
    This is the main entry point for spiral-seam duct calculations.

    Loss model:
    - Primary: Seam-crossing minor losses (K_total)
    - Optional: Swirl friction/HT multipliers (HEURISTIC, capped)
    - Roughness: base_roughness only (c_spiral*depth REMOVED in v3.0)

    The c_spiral roughness blending was removed because it was adapted from
    helically ribbed heat-transfer tubes and caused double-counting with
    seam K losses.

    For annular corrugations (flex hose), use CorrugatedPipeFriction directly.

    Args:
        diameter_m: Pipe inner diameter (m)
        length_m: Segment length (m)
        spiral_props: Spiral corrugation properties (includes K_seams_max)
        representative_Re: Representative Reynolds number (default 10000)
        enable_swirl: Enable swirl friction enhancement (HEURISTIC, default False)
        swirl_friction_cap: Maximum swirl friction increase (default 0.15)
        enable_HT: Enable heat transfer enhancement (HEURISTIC, default False)
        HT_enhancement_cap: Maximum HT increase (default 0.25)

    Returns:
        Dictionary with:
            'equivalent_roughness_m': Base roughness (+ seam if enabled)
            'K_total_seams': Total minor loss coefficient (clamped)
            'Mh': Heat transfer multiplier (1.0 if disabled)
            'K_seams_clamped': True if K_seams was clamped
            And diagnostic fields

    Raises:
        ValueError: If spiral_props.pitch_m <= 0
    """
    # Use the integrated spiral_equivalents function
    results = spiral_equivalents(
        diameter_m=diameter_m,
        length_m=length_m,
        Re_ref=representative_Re,
        depth_m=spiral_props.depth_m,
        pitch_m=spiral_props.pitch_m,
        num_starts=spiral_props.num_starts,
        base_roughness_m=spiral_props.base_roughness_m,
        K_seam_per_crossing=spiral_props.K_seam_per_crossing,
        enable_swirl=enable_swirl,
        swirl_friction_cap=swirl_friction_cap,
        enable_HT=enable_HT,
        HT_enhancement_cap=HT_enhancement_cap,
        has_lock_seam=spiral_props.has_lock_seam,
        K_lock=spiral_props.K_lock_seam,
        K_seams_max=spiral_props.K_seams_max,
        include_seam_roughness=spiral_props.include_seam_roughness,
        seam_protrusion_m=spiral_props.seam_protrusion_m
    )

    # Flatten debug info to main level for backward compatibility
    output = {
        'equivalent_roughness_m': results['roughness_m'],
        'K_total_seams': results['K_total'],
        'Mh': results['Mh'],
        'K_seams_clamped': results['K_seams_clamped']
    }

    # Add all debug fields to output
    output.update(results['debug'])

    # Rename some fields for backward compatibility
    output['K_lock_seam'] = output.pop('K_lock')
    output['friction_factor_corrugation'] = output['friction_factor_base']
    output['friction_factor_seams'] = 0.0  # v3.0: seams are K losses, not friction
    output['friction_factor_total'] = output['friction_factor_base']  # v3.0: no inflation

    return output


def invert_colebrook_for_roughness(
    Re: float,
    diameter_m: float,
    f_target: float,
    initial_guess: float,
    max_iterations: int = 10,
    tolerance: float = 1e-6
) -> float:
    """
    Invert Colebrook-White equation to find roughness that gives target friction factor

    Uses Newton-Raphson iteration to solve:
    f_target = 0.25 / [log10(ε/(3.7*D) + 5.74/Re^0.9)]²
    for ε given f_target, Re, and D.

    Args:
        Re: Reynolds number
        diameter_m: Pipe diameter (m)
        f_target: Target friction factor
        initial_guess: Initial guess for roughness (m)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Roughness in meters that gives f_target
    """
    eps = initial_guess
    friction_calc = CorrugatedPipeFriction()

    for i in range(max_iterations):
        # Compute friction factor with current roughness
        f_current = friction_calc._colebrook_white(Re, eps / diameter_m)

        # Check convergence
        if abs(f_current - f_target) < tolerance:
            break

        # Numerical derivative: df/dε
        deps = eps * 0.001 + 1e-9  # Small perturbation
        f_perturbed = friction_calc._colebrook_white(Re, (eps + deps) / diameter_m)
        df_deps = (f_perturbed - f_current) / deps

        if abs(df_deps) < 1e-12:
            # Derivative too small, use bisection fallback
            break

        # Newton-Raphson update
        delta_eps = (f_target - f_current) / df_deps
        eps = eps + delta_eps

        # Bounds checking
        eps = max(eps, 1e-7)  # Minimum 0.1 micron
        eps = min(eps, diameter_m * 0.1)  # Maximum 10% of diameter

    return eps

def compare_smooth_vs_corrugated(diameter_m: float = 0.1,
                                velocity_m_s: float = 5.0,
                                temperature_C: float = 200.0,
                                length_m: float = 10.0) -> Dict[str, float]:
    """
    Compare pressure drop for smooth vs corrugated pipes

    Args:
        diameter_m: Pipe diameter
        velocity_m_s: Flow velocity
        temperature_C: Gas temperature
        length_m: Pipe length

    Returns:
        Dictionary with comparison results
    """
    from flued_module.physics import PhysicalProperties

    physics = PhysicalProperties()
    friction_calc = CorrugatedPipeFriction()

    # Gas properties
    T_K = temperature_C + 273.15
    rho = physics.air_density(T_K, 98457)
    mu = physics.dynamic_viscosity_air(T_K)

    # Reynolds number
    Re = rho * velocity_m_s * diameter_m / mu

    # Smooth pipe
    smooth_roughness = 1e-6  # 1 micron
    f_smooth = friction_calc._colebrook_white(Re, smooth_roughness / diameter_m)
    dp_smooth = f_smooth * (length_m / diameter_m) * (rho * velocity_m_s ** 2 / 2)

    # Light corrugation
    light_corr = CorrugationProperties(
        depth_m=0.003,
        pitch_m=0.012,
        base_roughness_m=1e-6
    )
    f_light = friction_calc.calculate_corrugated_friction_factor(Re, diameter_m, light_corr)
    dp_light = f_light * (length_m / diameter_m) * (rho * velocity_m_s ** 2 / 2)

    # Heavy corrugation
    heavy_corr = CorrugationProperties(
        depth_m=0.006,
        pitch_m=0.025,
        base_roughness_m=1e-6
    )
    f_heavy = friction_calc.calculate_corrugated_friction_factor(Re, diameter_m, heavy_corr)
    dp_heavy = f_heavy * (length_m / diameter_m) * (rho * velocity_m_s ** 2 / 2)

    # Add seam losses (snap-lock joints every 1.2m)
    seam = SeamProperties(K_loss=0.05, spacing_m=1.2)
    dp_seams = friction_calc.calculate_seam_losses(length_m, velocity_m_s, seam) * rho

    return {
        'Reynolds_number': Re,
        'f_smooth': f_smooth,
        'f_light_corrugated': f_light,
        'f_heavy_corrugated': f_heavy,
        'dp_smooth_Pa': dp_smooth,
        'dp_light_corrugated_Pa': dp_light,
        'dp_heavy_corrugated_Pa': dp_heavy,
        'dp_seams_Pa': dp_seams,
        'enhancement_light': f_light / f_smooth,
        'enhancement_heavy': f_heavy / f_smooth,
        'draft_reduction_light': (1 - dp_smooth / dp_light) * 100,
        'draft_reduction_heavy': (1 - dp_smooth / dp_heavy) * 100
    }

if __name__ == "__main__":
    print("="*80)
    print("SPIRAL CORRUGATION v3.0 VALIDATION TESTS")
    print("="*80)
    print("\nv3.0 MODEL SEPARATION:")
    print("  - Spiral-seam duct: K_total (seam losses) + optional swirl multipliers")
    print("  - c_spiral roughness blending REMOVED (not applicable to spiral-seam geometry)")
    print("  - Roughness = base_roughness only")

    # Test parameters (from documentation example)
    D = 0.1016  # m (4 inches)
    L = 2.0     # m
    depth = 0.003  # m (3 mm)
    pitch = 0.180  # m (180 mm)
    num_starts = 4
    Re = 20000
    base_roughness = 1e-6  # 1 micron

    print("\nTest Configuration:")
    print(f"  Diameter: {D*1000:.1f} mm")
    print(f"  Length: {L:.1f} m")
    print(f"  Depth: {depth*1000:.1f} mm")
    print(f"  Pitch: {pitch*1000:.1f} mm")
    print(f"  Starts: {num_starts}")
    print(f"  Reynolds: {Re:.0f}")
    print(f"  Base roughness: {base_roughness*1e6:.1f} um")

    # Test 1: Geometric descriptors
    print("\n" + "-"*80)
    print("TEST 1: Geometric Descriptors")
    print("-"*80)

    desc = spiral_descriptors(D, depth, pitch)
    print(f"  Helix angle beta: {desc['beta_deg']:.2f} deg (expected ~28.7 deg)")
    print(f"  Swirl index S: {desc['swirl_index']:.3f} (expected ~0.42)")
    print(f"  Relative depth e/D: {desc['rel_depth']:.4f}")
    print(f"  Relative pitch p/D: {desc['rel_pitch']:.2f}")

    # Validation
    assert 27.0 <= desc['beta_deg'] <= 30.0, "Helix angle out of range"
    assert 0.40 <= desc['swirl_index'] <= 0.45, "Swirl index out of range"
    print("  PASS: Descriptors within expected ranges")

    # Test 2: v3.0 - Roughness equals base_roughness (no c_spiral)
    print("\n" + "-"*80)
    print("TEST 2: v3.0 Roughness = Base Only (c_spiral REMOVED)")
    print("-"*80)

    result = spiral_equivalents(D, L, Re, depth, pitch, num_starts,
                               base_roughness_m=base_roughness,
                               include_seam_roughness=False)
    print(f"  Base roughness: {base_roughness*1e6:.1f} um")
    print(f"  Output roughness: {result['roughness_m']*1e6:.1f} um")
    print(f"  c_spiral (diagnostic only): {result['debug']['spiral_factor_diagnostic']:.3f}")

    assert result['roughness_m'] == base_roughness, \
        f"v3.0: roughness should equal base_roughness, got {result['roughness_m']} != {base_roughness}"
    print("  PASS: Roughness equals base_roughness (c_spiral NOT applied)")

    # Test 2b: Optional seam roughness heuristic
    result_seam = spiral_equivalents(D, L, Re, depth, pitch, num_starts,
                                    base_roughness_m=base_roughness,
                                    include_seam_roughness=True,
                                    seam_protrusion_m=0.002)
    print(f"  With seam roughness heuristic: {result_seam['roughness_m']*1e6:.1f} um")
    assert result_seam['roughness_m'] == base_roughness + 0.002, \
        "Seam roughness heuristic not applied correctly"
    print("  PASS: Optional seam roughness heuristic works")

    # Test 3: K_total grows with length/num_starts, shrinks with pitch
    print("\n" + "-"*80)
    print("TEST 3: K_total Monotonicity")
    print("-"*80)

    # Vary length
    K_L1 = spiral_equivalents(D, 1.0, Re, depth, pitch, num_starts)['K_total']
    K_L2 = spiral_equivalents(D, 2.0, Re, depth, pitch, num_starts)['K_total']
    K_L4 = spiral_equivalents(D, 4.0, Re, depth, pitch, num_starts)['K_total']
    print(f"  Length 1m: K_total = {K_L1:.3f}")
    print(f"  Length 2m: K_total = {K_L2:.3f}")
    print(f"  Length 4m: K_total = {K_L4:.3f}")
    assert K_L1 < K_L2 < K_L4, "K_total should increase with length"

    # Vary num_starts
    K_n1 = spiral_equivalents(D, L, Re, depth, pitch, 1)['K_total']
    K_n4 = spiral_equivalents(D, L, Re, depth, pitch, 4)['K_total']
    K_n6 = spiral_equivalents(D, L, Re, depth, pitch, 6)['K_total']
    print(f"  1 start: K_total = {K_n1:.3f}")
    print(f"  4 starts: K_total = {K_n4:.3f}")
    print(f"  6 starts: K_total = {K_n6:.3f}")
    assert K_n1 < K_n4 < K_n6, "K_total should increase with num_starts"

    # Vary pitch
    K_p1 = spiral_equivalents(D, L, Re, depth, 0.10, num_starts)['K_total']
    K_p2 = spiral_equivalents(D, L, Re, depth, 0.18, num_starts)['K_total']
    K_p3 = spiral_equivalents(D, L, Re, depth, 0.30, num_starts)['K_total']
    print(f"  Pitch 100mm: K_total = {K_p1:.3f}")
    print(f"  Pitch 180mm: K_total = {K_p2:.3f}")
    print(f"  Pitch 300mm: K_total = {K_p3:.3f}")
    assert K_p1 > K_p2 > K_p3, "K_total should decrease with pitch"

    print("  PASS: K_total monotonicity correct")

    # Test 4: K_seams clamping
    print("\n" + "-"*80)
    print("TEST 4: K_seams Clamping Guardrail")
    print("-"*80)

    # Extreme parameters that would produce huge K_seams
    import warnings as w
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        result_extreme = spiral_equivalents(D, 20.0, Re, depth, 0.05, 6,
                                           K_seams_max=5.0)
        # Should have triggered a warning
        warned = any("K_seams" in str(warn.message) for warn in caught)

    K_raw = result_extreme['debug']['K_seams_raw']
    K_clamped = result_extreme['debug']['K_seams']
    print(f"  Extreme params (L=20m, pitch=50mm, starts=6):")
    print(f"    K_seams_raw = {K_raw:.2f}")
    print(f"    K_seams (clamped) = {K_clamped:.2f}")
    print(f"    K_seams_clamped flag = {result_extreme['K_seams_clamped']}")
    print(f"    Warning emitted = {warned}")

    assert result_extreme['K_seams_clamped'] == True, "Should be clamped"
    assert K_clamped <= 5.0, "K_seams should be clamped to K_seams_max"
    assert K_raw > K_clamped, "Raw should exceed clamped"
    print("  PASS: K_seams clamping works")

    # Test 5: Pitch validation
    print("\n" + "-"*80)
    print("TEST 5: Pitch Validation (pitch <= 0 raises ValueError)")
    print("-"*80)

    try:
        spiral_equivalents(D, L, Re, depth, pitch_m=0.0, num_starts=4)
        assert False, "Should have raised ValueError for pitch=0"
    except ValueError as e:
        print(f"  pitch=0: Correctly raised ValueError: {e}")

    try:
        spiral_equivalents(D, L, Re, depth, pitch_m=-0.1, num_starts=4)
        assert False, "Should have raised ValueError for pitch<0"
    except ValueError as e:
        print(f"  pitch<0: Correctly raised ValueError: {e}")

    print("  PASS: Invalid pitch raises ValueError")

    # Test 6: Swirl/HT multipliers within caps
    print("\n" + "-"*80)
    print("TEST 6: Swirl/HT Multiplier Caps (HEURISTIC)")
    print("-"*80)

    m_f = swirl_friction_multiplier(desc['swirl_index'], desc['rel_depth'],
                                    desc['rel_pitch'], Re)
    M_h = spiral_HT_multiplier(desc['swirl_index'], desc['rel_depth'],
                               desc['rel_pitch'], Re)
    print(f"  Typical case: m_f = {m_f:.3f}, M_h = {M_h:.3f}")
    assert 1.0 <= m_f <= 1.15, "Friction multiplier out of cap bounds"
    assert 1.0 <= M_h <= 1.25, "HT multiplier out of cap bounds"

    # Extreme geometry
    desc_large = spiral_descriptors(D, depth=0.02, pitch=pitch)
    m_f_large = swirl_friction_multiplier(desc_large['swirl_index'],
                                         desc_large['rel_depth'],
                                         desc_large['rel_pitch'], Re)
    M_h_large = spiral_HT_multiplier(desc_large['swirl_index'],
                                    desc_large['rel_depth'],
                                    desc_large['rel_pitch'], Re)
    print(f"  Extreme depth (20mm): m_f = {m_f_large:.3f}, M_h = {M_h_large:.3f}")
    assert m_f_large <= 1.15, "Friction cap not enforced"
    assert M_h_large <= 1.25, "HT cap not enforced"

    print("  PASS: Multipliers stay within caps")

    # Test 7: Model separation - swirl doesn't change roughness
    print("\n" + "-"*80)
    print("TEST 7: Model Separation (swirl doesn't change roughness)")
    print("-"*80)

    result_no_swirl = spiral_equivalents(D, L, Re, depth, pitch, num_starts,
                                        enable_swirl=False, enable_HT=False)
    result_with_swirl = spiral_equivalents(D, L, Re, depth, pitch, num_starts,
                                          enable_swirl=True, enable_HT=True)
    print(f"  Without swirl: roughness = {result_no_swirl['roughness_m']*1e6:.1f} um")
    print(f"  With swirl: roughness = {result_with_swirl['roughness_m']*1e6:.1f} um")
    print(f"  m_f without swirl = {result_no_swirl['debug']['friction_multiplier']:.3f}")
    print(f"  m_f with swirl = {result_with_swirl['debug']['friction_multiplier']:.3f}")

    # v3.0: roughness should be the same (swirl affects multipliers, not roughness)
    assert result_no_swirl['roughness_m'] == result_with_swirl['roughness_m'], \
        "v3.0: Swirl should not change roughness"
    assert result_no_swirl['debug']['friction_multiplier'] == 1.0
    assert result_with_swirl['debug']['friction_multiplier'] > 1.0
    assert result_with_swirl['Mh'] > 1.0

    print("  PASS: Swirl affects multipliers only, not roughness")

    # Test 8: Model version in output
    print("\n" + "-"*80)
    print("TEST 8: Model Version Tracking")
    print("-"*80)

    assert result['debug']['model_version'] == '3.0', "Should report v3.0"
    print(f"  Model version: {result['debug']['model_version']}")
    print("  PASS: Version tracking correct")

    print("\n" + "="*80)
    print("ALL v3.0 VALIDATION TESTS PASSED")
    print("="*80)
    print("\nVersion 3.0 model separation validated:")
    print("  - Spiral-seam duct uses K_total (seam losses), not roughness inflation")
    print("  - c_spiral roughness blending removed (was for ribbed tubes, not seams)")
    print("  - Swirl/HT multipliers are optional HEURISTICS with caps")
    print("  - K_seams clamping prevents solver instability")