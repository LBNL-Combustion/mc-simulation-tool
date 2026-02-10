"""
Unit tests for elbow loss calculations.
"""
import pytest
import numpy as np
import math
from flued_module.flow.elbows import (
    ElbowSpec, ElbowLoss, relative_roughness, churchill_f,
    base_K, elbow_losses, delta_p_elbow
)
from flued_module.flow.elbows_infer import (
    infer_elbows_from_polyline, estimate_elbow_type_from_geometry
)


class TestElbowPhysics:
    """Test physics-based elbow calculations."""
    
    def test_relative_roughness(self):
        """Test relative roughness calculations."""
        # Commercial steel, 0.1m diameter
        rel_eps = relative_roughness("commercial_steel", 0.1)
        assert 0.0001 < rel_eps < 0.001  # Reasonable range
        
        # Smooth pipe should have lower roughness
        rel_eps_smooth = relative_roughness("smooth", 0.1)
        assert rel_eps_smooth < rel_eps
        
    def test_churchill_friction_factor(self):
        """Test Churchill friction factor equation."""
        # Laminar flow
        f_laminar = churchill_f(Re=1000, rel_eps=0.0001)
        f_expected = 64.0 / 1000  # Hagen-Poiseuille
        assert abs(f_laminar - f_expected) / f_expected < 0.1
        
        # Turbulent flow, smooth pipe
        f_turbulent = churchill_f(Re=100000, rel_eps=1e-6)
        assert 0.01 < f_turbulent < 0.03  # Reasonable range
        
        # Turbulent flow, rough pipe
        f_rough = churchill_f(Re=100000, rel_eps=0.001)
        assert f_rough > f_turbulent  # Rougher pipes have higher friction
        
    def test_base_K_values(self):
        """Test base K coefficient retrieval."""
        # 90° long radius elbow
        spec = ElbowSpec(
            angle_deg=90,
            diameter_m=0.1,
            R_over_D=1.5,
            elbow_type="long_radius",
            smoothness="smooth"
        )
        K = base_K(spec)
        assert 0.2 < K < 0.5  # Expected range from tables
        
        # 45° should have lower K
        spec_45 = ElbowSpec(
            angle_deg=45,
            diameter_m=0.1,
            R_over_D=1.5,
            elbow_type="long_radius",
            smoothness="smooth"
        )
        K_45 = base_K(spec_45)
        assert K_45 < K  # 45° has less loss than 90°
        
        # Short radius should have higher K
        spec_short = ElbowSpec(
            angle_deg=90,
            diameter_m=0.1,
            R_over_D=1.0,
            elbow_type="short_radius",
            smoothness="smooth"
        )
        K_short = base_K(spec_short)
        assert K_short > K  # Tighter bend has more loss
        
    def test_elbow_losses_calculation(self):
        """Test complete elbow loss calculation."""
        spec = ElbowSpec(
            angle_deg=90,
            diameter_m=0.15,
            R_over_D=1.5,
            elbow_type="long_radius",
            smoothness="commercial_steel"
        )
        
        # Typical conditions
        rho = 1.2  # kg/m³
        mu = 1.8e-5  # Pa·s
        Q = 0.1  # m³/s
        
        losses = elbow_losses(spec, rho, mu, Q)
        
        assert isinstance(losses, ElbowLoss)
        assert losses.K > 0
        assert losses.Le_m > 0
        assert losses.method_used == "K_table"
        
        # Verify Le/D = K/(4f) relationship
        area = math.pi * (spec.diameter_m / 2)**2
        velocity = Q / area
        Re = rho * velocity * spec.diameter_m / mu
        rel_eps = relative_roughness(spec.smoothness, spec.diameter_m)
        f = churchill_f(Re, rel_eps)
        
        Le_over_D_expected = losses.K / (4 * f)
        Le_over_D_actual = losses.Le_m / spec.diameter_m
        
        assert abs(Le_over_D_actual - Le_over_D_expected) < 0.1
        
    def test_pressure_drop_calculation(self):
        """Test pressure drop calculation."""
        K = 0.5
        rho = 1.2  # kg/m³
        Q = 0.05  # m³/s
        D = 0.1  # m
        
        dp = delta_p_elbow(K, rho, Q, D)
        
        # Manual calculation
        area = math.pi * (D / 2)**2
        velocity = Q / area
        dp_expected = K * 0.5 * rho * velocity**2
        
        assert abs(dp - dp_expected) < 0.01
        
    def test_elbow_with_vane_factor(self):
        """Test that guide vanes reduce losses."""
        spec_no_vanes = ElbowSpec(
            angle_deg=90,
            diameter_m=0.1,
            R_over_D=1.5,
            elbow_type="long_radius",
            smoothness="smooth",
            vane_factor=1.0
        )
        
        spec_with_vanes = ElbowSpec(
            angle_deg=90,
            diameter_m=0.1,
            R_over_D=1.5,
            elbow_type="long_radius",
            smoothness="smooth",
            vane_factor=0.6  # 40% reduction with guide vanes
        )
        
        rho = 1.2
        mu = 1.8e-5
        Q = 0.05
        
        losses_no_vanes = elbow_losses(spec_no_vanes, rho, mu, Q)
        losses_with_vanes = elbow_losses(spec_with_vanes, rho, mu, Q)
        
        assert losses_with_vanes.K < losses_no_vanes.K
        assert abs(losses_with_vanes.K / losses_no_vanes.K - 0.6) < 0.01
        
    def test_override_values(self):
        """Test K and Le override functionality."""
        spec_K_override = ElbowSpec(
            angle_deg=90,
            diameter_m=0.1,
            R_over_D=1.5,
            elbow_type="long_radius",
            smoothness="smooth",
            K_override=2.0
        )
        
        losses = elbow_losses(spec_K_override, rho=1.2, mu=1.8e-5, 
                            volumetric_flow_m3_s=0.05)
        
        assert losses.K == 2.0
        assert losses.method_used == "override"
        
    def test_invalid_inputs(self):
        """Test validation of invalid inputs."""
        # Invalid angle
        with pytest.raises(ValueError):
            spec = ElbowSpec(
                angle_deg=200,  # > 180
                diameter_m=0.1,
                R_over_D=1.5,
                elbow_type="long_radius"
            )
            base_K(spec)
        
        # Invalid diameter
        with pytest.raises(ValueError):
            spec = ElbowSpec(
                angle_deg=90,
                diameter_m=-0.1,  # Negative
                R_over_D=1.5,
                elbow_type="long_radius"
            )
            base_K(spec)
        
        # Invalid R/D
        with pytest.raises(ValueError):
            spec = ElbowSpec(
                angle_deg=90,
                diameter_m=0.1,
                R_over_D=0.5,  # < 0.8
                elbow_type="long_radius"
            )
            base_K(spec)


class TestElbowInference:
    """Test elbow detection from geometry."""
    
    def test_straight_line_no_elbows(self):
        """Test that straight line produces no elbows."""
        points = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3)
        ]
        
        elbows = infer_elbows_from_polyline(points, diameter_m=0.1)
        assert len(elbows) == 0
        
    def test_90_degree_elbow_detection(self):
        """Test detection of 90° elbow."""
        points = [
            (0, 0, 0),
            (0, 0, 1),  # Vertical
            (1, 0, 1)   # Horizontal (90° turn)
        ]
        
        elbows = infer_elbows_from_polyline(points, diameter_m=0.1)
        assert len(elbows) == 1
        assert 85 < elbows[0].angle_deg < 95  # Approximately 90°
        
    def test_multiple_elbows(self):
        """Test detection of multiple elbows."""
        points = [
            (0, 0, 0),
            (0, 0, 1),  # Up
            (1, 0, 1),  # Right (90°)
            (1, 0, 2),  # Up again (90°)
            (2, 0, 2)   # Right again (90°)
        ]
        
        elbows = infer_elbows_from_polyline(points, diameter_m=0.1)
        assert len(elbows) == 3
        for elbow in elbows:
            assert 85 < elbow.angle_deg < 95
            
    def test_45_degree_elbow(self):
        """Test detection of 45° elbow."""
        points = [
            (0, 0, 0),
            (0, 0, 1),
            (0.707, 0, 1.707)  # 45° angle
        ]
        
        elbows = infer_elbows_from_polyline(points, diameter_m=0.1, min_angle_deg=5)
        assert len(elbows) == 1
        assert 40 < elbows[0].angle_deg < 50
        
    def test_min_angle_threshold(self):
        """Test minimum angle threshold."""
        points = [
            (0, 0, 0),
            (0, 0, 1),
            (0.05, 0, 2)  # Very small angle
        ]
        
        # Should detect with low threshold
        elbows_low = infer_elbows_from_polyline(points, diameter_m=0.1, 
                                                 min_angle_deg=1.0)
        assert len(elbows_low) == 1
        
        # Should not detect with high threshold  
        elbows_high = infer_elbows_from_polyline(points, diameter_m=0.1,
                                                  min_angle_deg=10.0)
        assert len(elbows_high) == 0
        
    def test_elbow_type_estimation(self):
        """Test estimation of elbow type from geometry."""
        # Sharp angle should suggest mitered
        elbow_type, R_over_D = estimate_elbow_type_from_geometry(angle_deg=135)
        assert elbow_type == "mitered"
        
        # Gentle angle should suggest long radius
        elbow_type, R_over_D = estimate_elbow_type_from_geometry(angle_deg=30)
        assert elbow_type == "long_radius"
        assert R_over_D > 2.0
        
        # With space constraints
        elbow_type, R_over_D = estimate_elbow_type_from_geometry(
            angle_deg=90, available_space_m=0.08, diameter_m=0.1
        )
        assert elbow_type == "short_radius"


class TestIntegration:
    """Integration tests for the complete elbow system."""
    
    def test_polyline_to_pressure_drop(self):
        """Test complete workflow from geometry to pressure drop."""
        # Define a simple L-shaped flue
        points = [
            (0, 0, 0),
            (0, 0, 2),  # 2m vertical
            (3, 0, 2)   # 3m horizontal
        ]
        
        diameter = 0.15  # m
        
        # Detect elbows
        elbows = infer_elbows_from_polyline(
            points, 
            diameter_m=diameter,
            default_elbow_type="long_radius",
            default_R_over_D=1.5
        )
        
        assert len(elbows) == 1
        
        # Calculate losses
        rho = 1.2  # kg/m³
        mu = 1.8e-5  # Pa·s  
        Q = 0.1  # m³/s
        
        total_K = 0
        total_dp = 0
        
        for elbow in elbows:
            losses = elbow_losses(elbow, rho, mu, Q)
            total_K += losses.K
            total_dp += delta_p_elbow(losses.K, rho, Q, diameter)
            
        assert total_K > 0
        assert total_dp > 0
        
        # Verify pressure drop is reasonable (not too high or low)
        assert 10 < total_dp < 1000  # Pa
        
    def test_backward_compatibility(self):
        """Test that old has_fittings flag still works with deprecation warning."""
        from flued_module.comprehensive_heat_transfer import ComprehensiveHeatTransfer
        
        ht = ComprehensiveHeatTransfer()
        
        # Test shim creation
        elbows = ht._create_default_elbow_from_fittings(has_fittings=True, diameter=0.1)
        assert elbows is not None
        assert len(elbows) == 1
        assert elbows[0].angle_deg == 90
        
        elbows_none = ht._create_default_elbow_from_fittings(has_fittings=False, diameter=0.1)
        assert elbows_none is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])