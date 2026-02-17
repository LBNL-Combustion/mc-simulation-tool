"""
Automatic elbow inference from flue geometry.

This module provides functions to detect elbows from polyline geometry
and create corresponding ElbowSpec objects.
"""

from typing import List, Tuple, Optional
import numpy as np
import math
import logging
from .elbows import ElbowSpec, ElbowType, Smoothness

logger = logging.getLogger(__name__)


def infer_elbows_from_polyline(
    points: List[Tuple[float, float, float]],
    diameter_m: float,
    default_elbow_type: ElbowType = "long_radius",
    default_R_over_D: float = 1.5,
    smoothness: Smoothness = "commercial_steel",
    min_angle_deg: float = 5.0
) -> List[ElbowSpec]:
    """
    Given a 3D polyline of flue centerline points, return one ElbowSpec per bend.
    
    Args:
        points: List of (x, y, z) coordinates defining the flue centerline
        diameter_m: Flue hydraulic diameter (m)
        default_elbow_type: Type to assume for detected elbows
        default_R_over_D: Default bend radius ratio
        smoothness: Surface finish of the flue
        min_angle_deg: Minimum angle to consider as an elbow
        
    Returns:
        List of ElbowSpec objects for detected elbows
        
    Notes:
        - Angle is computed from consecutive segment direction vectors
        - Uses arccos of dot product to find bend angle
        - Skips very small angles (< min_angle_deg)
    """
    if len(points) < 3:
        logger.debug("Insufficient points for elbow detection")
        return []
    
    elbows = []
    
    pts = np.array(points)

    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i-1], pts[i], pts[i+1]

        v1 = p1 - p0
        v2 = p2 - p1

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 1e-6 or len_v2 < 1e-6:
            logger.debug(f"Skipping point {i}: zero-length segment")
            continue
        
        v1_norm = v1 / len_v1
        v2_norm = v2 / len_v2
        
        # cos(θ) = v1·v2 / (|v1||v2|)
        cos_angle = np.dot(v1_norm, v2_norm)

        # Clamp to avoid numerical issues with arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Note: cos(0) = 1 (straight), cos(π/2) = 0 (90°), cos(π) = -1 (180°)
        # The bend angle is π - arccos(dot product)
        angle_rad = math.acos(cos_angle)
        
        # For flow, we want the deviation from straight (supplement angle)
        # If vectors are aligned (straight), angle_rad ≈ 0, bend = 0
        # If vectors are perpendicular, angle_rad = π/2, bend = 90°
        # If vectors are opposite, angle_rad = π, bend = 180°
        bend_angle_deg = math.degrees(angle_rad)
        
        if bend_angle_deg < min_angle_deg:
            logger.debug(f"Skipping point {i}: angle {bend_angle_deg:.1f}° < {min_angle_deg}°")
            continue
        
        elbow = ElbowSpec(
            angle_deg=bend_angle_deg,
            diameter_m=diameter_m,
            R_over_D=default_R_over_D,
            elbow_type=default_elbow_type,
            smoothness=smoothness,
            vane_factor=1.0
        )
        
        elbows.append(elbow)
        
        logger.info(f"Detected elbow at point {i}: angle={bend_angle_deg:.1f}°, "
                   f"type={default_elbow_type}, R/D={default_R_over_D}")
    
    logger.info(f"Detected {len(elbows)} elbows from {len(points)} points")
    return elbows


def polyline_from_chimney_segments(chimney) -> Optional[List[Tuple[float, float, float]]]:
    """
    Extract a 3D polyline from chimney segment data.
    
    Args:
        chimney: Chimney object with parts list
        
    Returns:
        List of (x, y, z) points representing the centerline, or None if unable to extract
    """
    try:
        points = []
        current_x, current_y, current_z = 0.0, 0.0, 0.0
        
        points.append((current_x, current_y, current_z))
        
        for part in chimney.parts:
            if hasattr(part, 'orientation'):
                if part.orientation == 'vertical':
                    current_z += part.length_m
                elif part.orientation == 'horizontal':
                    current_x += part.length_m
                else:
                    logger.warning(f"Unknown orientation: {part.orientation}")
                    continue
            else:
                current_z += part.length_m
            
            points.append((current_x, current_y, current_z))
        
        logger.debug(f"Extracted {len(points)} points from {len(chimney.parts)} chimney parts")
        return points
        
    except Exception as e:
        logger.error(f"Failed to extract polyline from chimney: {e}")
        return None


def infer_elbows_from_chimney(
    chimney,
    default_elbow_type: ElbowType = "long_radius",
    default_R_over_D: float = 1.5,
    smoothness: Optional[Smoothness] = None,
    min_angle_deg: float = 5.0
) -> List[ElbowSpec]:
    """
    Convenience function to infer elbows directly from a chimney object.
    
    Args:
        chimney: Chimney object with parts and geometry
        default_elbow_type: Type to assume for detected elbows
        default_R_over_D: Default bend radius ratio
        smoothness: Surface finish (if None, infer from chimney)
        min_angle_deg: Minimum angle to consider as an elbow
        
    Returns:
        List of ElbowSpec objects for detected elbows
    """
    points = polyline_from_chimney_segments(chimney)
    if not points:
        logger.warning("Could not extract polyline from chimney")
        return []
    
    # Get diameter (assuming uniform for now)
    try:
        diameter_m = chimney.parts[0].inner_diameter_m
    except (IndexError, AttributeError):
        logger.error("Could not determine chimney diameter")
        return []
    
    if smoothness is None:
        smoothness = "commercial_steel"
    
    return infer_elbows_from_polyline(
        points=points,
        diameter_m=diameter_m,
        default_elbow_type=default_elbow_type,
        default_R_over_D=default_R_over_D,
        smoothness=smoothness,
        min_angle_deg=min_angle_deg
    )


def estimate_elbow_type_from_geometry(
    angle_deg: float,
    available_space_m: Optional[float] = None,
    diameter_m: Optional[float] = None
) -> Tuple[ElbowType, float]:
    """
    Estimate appropriate elbow type based on geometric constraints.
    
    Args:
        angle_deg: Bend angle in degrees
        available_space_m: Available space for the bend (optional)
        diameter_m: Pipe diameter (optional)
        
    Returns:
        Tuple of (elbow_type, R_over_D)
        
    Notes:
        - Sharp angles (>120°) typically use mitered elbows
        - Limited space requires short radius
        - Default to long radius for efficiency
    """
    # Very sharp bends typically use mitered construction
    if angle_deg > 120:
        return "mitered", 1.0
    
    # If space is constrained, use short radius
    if available_space_m is not None and diameter_m is not None:
        required_radius = available_space_m / (2 * math.sin(math.radians(angle_deg / 2)))
        R_over_D = required_radius / diameter_m
        
        if R_over_D < 1.0:
            return "short_radius", 0.8
        elif R_over_D < 1.5:
            return "short_radius", 1.0
        else:
            return "long_radius", min(R_over_D, 5.0)
    
    # Default based on angle alone
    if angle_deg <= 45:
        return "long_radius", 2.5  # Gentle bend, use smooth elbow
    elif angle_deg <= 90:
        return "long_radius", 1.5  # Standard 90° elbow
    else:
        return "short_radius", 1.0  # Sharp bend