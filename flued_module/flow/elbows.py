"""
Physics-based elbow loss calculations using Darcy-Weisbach methodology.

This module provides accurate pressure drop calculations for pipe elbows,
supporting both K-factor and equivalent length methods.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, List, Tuple
import math
import logging

logger = logging.getLogger(__name__)

ElbowType = Literal["long_radius", "short_radius", "mitered", "tee"]
Smoothness = Literal["smooth", "commercial_steel", "galvanized", "rough"]

@dataclass
class ElbowSpec:
    """Specification for a single elbow in the flow system."""
    angle_deg: float                 # 0 < angle ≤ 180
    diameter_m: float                # hydraulic D (round flue)
    R_over_D: float                  # centerline bend radius / D
    elbow_type: ElbowType
    smoothness: Smoothness = "commercial_steel"
    vane_factor: float = 1.0         # <1 reduces losses if guide vanes present
    # Optional overrides
    K_override: Optional[float] = None
    Le_override_m: Optional[float] = None

@dataclass
class ElbowLoss:
    """Calculated losses for an elbow."""
    K: float
    Le_m: float
    method_used: Literal["K_table", "Le_from_K", "override"]


# Base loss data from Crane Technical Paper 410 and Idelchik handbook
# K values for fully turbulent flow (Re > 2e4)
BASE_LOSS_DATA: Dict[tuple, Dict[str, List[float]]] = {
    # key: (smoothness_group, elbow_type, angle)  # angle anchors
    ("smooth", "long_radius", 90): {"R_over_D": [1.0, 1.5, 2.5, 5.0], "K": [0.75, 0.35, 0.25, 0.20]},
    ("smooth", "short_radius", 90): {"R_over_D": [0.8, 1.0, 1.5], "K": [1.80, 1.50, 0.90]},
    ("smooth", "mitered", 90): {"R_over_D": [1.0], "K": [2.20]},  # Mitered has fixed R/D
    ("smooth", "long_radius", 45): {"R_over_D": [1.0, 1.5, 2.5, 5.0], "K": [0.35, 0.18, 0.13, 0.10]},
    ("smooth", "short_radius", 45): {"R_over_D": [0.8, 1.0, 1.5], "K": [0.85, 0.70, 0.45]},
    ("smooth", "mitered", 45): {"R_over_D": [1.0], "K": [1.10]},
    
    # Commercial steel/galvanized - slightly higher losses
    ("commercial_steel", "long_radius", 90): {"R_over_D": [1.0, 1.5, 2.5, 5.0], "K": [0.80, 0.40, 0.30, 0.25]},
    ("commercial_steel", "short_radius", 90): {"R_over_D": [0.8, 1.0, 1.5], "K": [1.90, 1.60, 1.00]},
    ("commercial_steel", "mitered", 90): {"R_over_D": [1.0], "K": [2.40]},
    ("commercial_steel", "long_radius", 45): {"R_over_D": [1.0, 1.5, 2.5, 5.0], "K": [0.40, 0.20, 0.15, 0.12]},
    ("commercial_steel", "short_radius", 45): {"R_over_D": [0.8, 1.0, 1.5], "K": [0.95, 0.80, 0.50]},
    
    # Rough pipes - highest losses
    ("rough", "long_radius", 90): {"R_over_D": [1.0, 1.5, 2.5], "K": [0.90, 0.50, 0.40]},
    ("rough", "short_radius", 90): {"R_over_D": [0.8, 1.0, 1.5], "K": [2.00, 1.80, 1.20]},
    ("rough", "mitered", 90): {"R_over_D": [1.0], "K": [2.60]},

    # Tee junctions - branch flow (90° turn into branch)
    # Through-flow would be K≈0.2-0.4, but we model branch flow (higher loss)
    ("smooth", "tee", 90): {"R_over_D": [1.0], "K": [0.90]},
    ("commercial_steel", "tee", 90): {"R_over_D": [1.0], "K": [1.00]},
    ("rough", "tee", 90): {"R_over_D": [1.0], "K": [1.10]},
}

# Surface roughness values (m) from engineering handbooks
ROUGHNESS_VALUES = {
    "smooth": 1.5e-6,           # Drawn tubing, glass
    "commercial_steel": 4.5e-5,  # New commercial steel
    "galvanized": 1.5e-4,        # Galvanized steel
    "rough": 2.5e-4,             # Corroded steel, cast iron
}


def relative_roughness(smoothness: Smoothness, D_m: float) -> float:
    """
    Calculate relative roughness ε/D for common pipe finishes.
    
    Args:
        smoothness: Surface finish type
        D_m: Pipe diameter in meters
        
    Returns:
        Relative roughness ε/D (dimensionless)
        
    Notes:
        Values based on Moody chart and Crane TP-410 Table A-23
    """
    abs_roughness = ROUGHNESS_VALUES.get(smoothness, ROUGHNESS_VALUES["commercial_steel"])
    return abs_roughness / D_m


def churchill_f(Re: float, rel_eps: float) -> float:
    """
    Calculate Darcy friction factor using Churchill equation.
    
    Valid for all flow regimes (laminar, transitional, turbulent) and
    all roughness values. More robust than Colebrook for numerical work.
    
    Args:
        Re: Reynolds number
        rel_eps: Relative roughness ε/D
        
    Returns:
        Darcy friction factor
        
    References:
        Churchill, S.W. (1977) "Friction-factor equation spans all fluid-flow regimes"
        Chemical Engineering, Nov 7, pp. 91-92
    """
    if Re < 1:
        return 64.0  # Limit for very low Re
    
    A = (2.457 * math.log(1.0 / ((7.0 / Re)**0.9 + 0.27 * rel_eps)))**16
    B = (37530.0 / Re)**16

    f = 8.0 * ((8.0 / Re)**12 + 1.0 / (A + B)**1.5)**(1.0/12.0)
    
    logger.debug(f"Churchill f: Re={Re:.0f}, ε/D={rel_eps:.6f}, f={f:.6f}")
    return f


def _interpolate(x: float, x_vals: List[float], y_vals: List[float]) -> float:
    """Linear interpolation helper."""
    if x <= x_vals[0]:
        return y_vals[0]
    if x >= x_vals[-1]:
        return y_vals[-1]
    
    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x <= x_vals[i+1]:
            frac = (x - x_vals[i]) / (x_vals[i+1] - x_vals[i])
            return y_vals[i] + frac * (y_vals[i+1] - y_vals[i])
    
    return y_vals[-1]


def base_K(spec: ElbowSpec) -> float:
    """
    Determine base loss coefficient K from tables and scaling rules.
    
    Args:
        spec: Elbow specification
        
    Returns:
        Base K value for the elbow
        
    Raises:
        ValueError: If parameters are outside valid ranges
    """
    if not (0 < spec.angle_deg <= 180):
        raise ValueError(f"Elbow angle must be 0 < angle ≤ 180, got {spec.angle_deg}")
    if spec.diameter_m <= 0:
        raise ValueError(f"Diameter must be positive, got {spec.diameter_m}")
    if spec.R_over_D < 0.8:
        raise ValueError(f"R/D must be ≥ 0.8, got {spec.R_over_D}")
    
    # Map smoothness to table key
    if spec.smoothness in ["smooth"]:
        smooth_key = "smooth"
    elif spec.smoothness in ["commercial_steel", "galvanized"]:
        smooth_key = "commercial_steel"
    else:
        smooth_key = "rough"
    
    # Find nearest angle anchor (prefer 90° as primary reference)
    angles_available = [45, 90]  # Standard reference angles
    if spec.angle_deg <= 45:
        ref_angle = 45
    else:
        ref_angle = 90
    
    # Lookup table key
    table_key = (smooth_key, spec.elbow_type, ref_angle)
    
    # Get base data or use fallback
    if table_key in BASE_LOSS_DATA:
        data = BASE_LOSS_DATA[table_key]
    else:
        # Fallback to commercial steel 90° short radius
        logger.warning(f"No data for {table_key}, using fallback")
        data = BASE_LOSS_DATA[("commercial_steel", "short_radius", 90)]
        ref_angle = 90
    
    # Interpolate K for the given R/D
    K_at_ref = _interpolate(spec.R_over_D, data["R_over_D"], data["K"])
    
    # Scale by angle (K ∝ θ/90 for most elbows)
    if ref_angle == 90:
        K_base = K_at_ref * (spec.angle_deg / 90.0)
    elif ref_angle == 45:
        # If we used 45° reference, scale appropriately
        K_base = K_at_ref * (spec.angle_deg / 45.0)
    else:
        K_base = K_at_ref
    
    # Additional scaling for very sharp bends
    if spec.angle_deg > 90:
        # Losses increase non-linearly for angles > 90°
        K_base *= 1.0 + 0.3 * ((spec.angle_deg - 90) / 90)
    
    logger.debug(f"Base K: type={spec.elbow_type}, angle={spec.angle_deg}°, "
                f"R/D={spec.R_over_D:.1f}, K={K_base:.3f}")
    
    return K_base


def elbow_losses(spec: ElbowSpec, rho: float, mu: float, 
                 volumetric_flow_m3_s: float) -> ElbowLoss:
    """
    Calculate elbow pressure losses using Darcy-Weisbach methodology.
    
    Args:
        spec: Elbow specification
        rho: Fluid density (kg/m³)
        mu: Dynamic viscosity (Pa·s)
        volumetric_flow_m3_s: Volumetric flow rate (m³/s)
        
    Returns:
        ElbowLoss with K, Le_m, and method used
        
    Notes:
        - K is the loss coefficient for ΔP = K·(ρv²/2)
        - Le is equivalent length where ΔP = f·(Le/D)·(ρv²/2)
        - Relationship: Le/D = K/(4f)
    """
    # Use override if provided
    if spec.K_override is not None:
        K = max(0, spec.K_override)
        # Still need to calculate Le from K
        area = math.pi * (spec.diameter_m / 2)**2
        velocity = volumetric_flow_m3_s / area
        Re = rho * velocity * spec.diameter_m / mu
        rel_eps = relative_roughness(spec.smoothness, spec.diameter_m)
        f = churchill_f(Re, rel_eps)
        Le_m = K * spec.diameter_m / (4 * f)
        
        logger.debug(f"Using K override: K={K:.3f}, Le={Le_m:.2f}m")
        return ElbowLoss(K=K, Le_m=Le_m, method_used="override")
    
    if spec.Le_override_m is not None:
        Le_m = max(0, spec.Le_override_m)
        # Calculate K from Le
        area = math.pi * (spec.diameter_m / 2)**2
        velocity = volumetric_flow_m3_s / area
        Re = rho * velocity * spec.diameter_m / mu
        rel_eps = relative_roughness(spec.smoothness, spec.diameter_m)
        f = churchill_f(Re, rel_eps)
        K = 4 * f * Le_m / spec.diameter_m
        
        logger.debug(f"Using Le override: Le={Le_m:.2f}m, K={K:.3f}")
        return ElbowLoss(K=K, Le_m=Le_m, method_used="override")
    
    area = math.pi * (spec.diameter_m / 2)**2
    velocity = volumetric_flow_m3_s / area
    Re = rho * velocity * spec.diameter_m / mu
    rel_eps = relative_roughness(spec.smoothness, spec.diameter_m)
    f = churchill_f(Re, rel_eps)
    
    # Get base K from tables
    K_base = base_K(spec)
    
    # Apply vane factor (guide vanes reduce losses)
    K = K_base * spec.vane_factor
    K = max(0, K)  # Ensure non-negative
    
    # Optional: Mild Reynolds number correction for transitional flow
    # K_eff = K * (1 + c/√Re) for Re < 2e4
    if Re < 2e4 and Re > 2300:
        # Transitional regime correction
        correction = 1.0 + 500.0 / math.sqrt(max(Re, 100))
        correction = min(correction, 2.0)  # Cap correction factor
        K = K * correction
        logger.debug(f"Applied transitional Re correction: factor={correction:.2f}")
    
    # Calculate equivalent length: Le/D = K/(4f)
    Le_over_D = K / (4 * f) if f > 0 else 0
    Le_m = Le_over_D * spec.diameter_m
    
    logger.debug(f"Elbow losses: Re={Re:.0f}, f={f:.6f}, K={K:.3f}, "
                f"Le/D={Le_over_D:.1f}, Le={Le_m:.2f}m")
    
    return ElbowLoss(K=K, Le_m=Le_m, method_used="K_table")


def delta_p_elbow(K: float, rho: float, volumetric_flow_m3_s: float, 
                  D_m: float) -> float:
    """
    Calculate pressure drop across an elbow.
    
    Args:
        K: Loss coefficient
        rho: Fluid density (kg/m³)
        volumetric_flow_m3_s: Volumetric flow rate (m³/s)
        D_m: Pipe diameter (m)
        
    Returns:
        Pressure drop (Pa)
        
    Notes:
        Uses ΔP = K·(ρv²/2) where v = Q/A
    """
    area = math.pi * (D_m / 2)**2
    velocity = volumetric_flow_m3_s / area
    delta_p = K * 0.5 * rho * velocity**2
    
    logger.debug(f"Elbow ΔP: K={K:.3f}, v={velocity:.2f}m/s, ΔP={delta_p:.1f}Pa")
    
    return delta_p