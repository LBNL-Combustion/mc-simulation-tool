"""
Pytest tests for momentum solver physics upgrades.

Tests cover:
1. Distributed leakage direction (VENT-II style)
2. Acceleration/deceleration loss sign
3. Colebrook-White friction factor

All tests are deterministic and do not require running the full coupled energy solve.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flued_module.mass_flow import (
    friction_factor,
    calculate_leakage_mass_flow,
    calculate_accel_pressure_loss
)


class TestDistributedLeakage:
    """Test distributed leakage (VENT-II style) calculations."""

    def test_leakage_out_positive_pressure(self):
        """
        Test 1a: Positive P_gauge => leakage OUT => m_leak_signed negative.

        When gas pressure is above ambient (positive gauge pressure),
        gas leaks OUT of the vent, reducing downstream mass flow.
        This should give a negative m_leak_signed.
        """
        # Setup: positive gauge pressure (gas above ambient)
        P_gauge = 10.0  # Pa (above ambient)
        A_leak_m2 = 1e-4  # 1 cm² leak area
        Cd_leak = 0.61
        rho_ambient = 1.2  # kg/m³
        rho_gas = 0.8  # kg/m³ (hot gas, less dense)

        m_leak = calculate_leakage_mass_flow(
            P_gauge, A_leak_m2, Cd_leak, rho_ambient, rho_gas
        )

        # Leakage OUT should be negative (reduces downstream flow)
        assert m_leak < 0, f"Expected negative m_leak for P_gauge > 0, got {m_leak}"

        # Verify magnitude is physically reasonable
        # m = Cd * A * sqrt(2 * rho * dP)
        expected_magnitude = Cd_leak * A_leak_m2 * np.sqrt(2 * rho_ambient * abs(P_gauge))
        assert abs(m_leak) == pytest.approx(expected_magnitude, rel=1e-10)

    def test_leakage_in_negative_pressure(self):
        """
        Test 1b: Negative P_gauge => leakage IN => m_leak_signed positive.

        When gas pressure is below ambient (negative gauge pressure),
        ambient air leaks INTO the vent, adding to downstream mass flow.
        This should give a positive m_leak_signed.
        """
        # Setup: negative gauge pressure (gas below ambient)
        P_gauge = -10.0  # Pa (below ambient)
        A_leak_m2 = 1e-4  # 1 cm² leak area
        Cd_leak = 0.61
        rho_ambient = 1.2  # kg/m³
        rho_gas = 0.8  # kg/m³ (hot gas, less dense)

        m_leak = calculate_leakage_mass_flow(
            P_gauge, A_leak_m2, Cd_leak, rho_ambient, rho_gas
        )

        # Leakage IN should be positive (adds to downstream flow)
        assert m_leak > 0, f"Expected positive m_leak for P_gauge < 0, got {m_leak}"

        # Verify magnitude (uses rho_gas for inflow per VENT-II)
        expected_magnitude = Cd_leak * A_leak_m2 * np.sqrt(2 * rho_gas * abs(P_gauge))
        assert abs(m_leak) == pytest.approx(expected_magnitude, rel=1e-10)

    def test_zero_leak_area_gives_zero_flow(self):
        """Test that zero leak area gives zero leakage."""
        m_leak = calculate_leakage_mass_flow(
            P_gauge=10.0,
            A_leak_m2=0.0,
            Cd_leak=0.61,
            rho_ambient=1.2,
            rho_gas=0.8
        )
        assert m_leak == 0.0

    def test_zero_pressure_diff_gives_zero_flow(self):
        """Test that zero pressure difference gives zero leakage."""
        m_leak = calculate_leakage_mass_flow(
            P_gauge=0.0,
            A_leak_m2=1e-4,
            Cd_leak=0.61,
            rho_ambient=1.2,
            rho_gas=0.8
        )
        assert m_leak == 0.0


class TestAccelerationLoss:
    """Test acceleration/deceleration loss calculations."""

    def test_accel_loss_density_decrease(self):
        """
        Test 2: Two segments with same m_dot and area, but rho decreases downstream.

        When density decreases downstream (gas heats up), velocity increases,
        which increases velocity head, requiring additional pressure (loss).
        """
        # Setup: same mass flow, same area, different densities
        m_dot = 0.01  # kg/s
        area = 0.01767  # m² (6" diameter)

        rho1 = 1.0  # kg/m³ (upstream, cooler)
        rho2 = 0.8  # kg/m³ (downstream, hotter)

        v1 = m_dot / (rho1 * area)
        v2 = m_dot / (rho2 * area)

        # Calculate acceleration loss
        dp_accel = calculate_accel_pressure_loss(rho1, v1, rho2, v2, C_accel=1.0)

        # Expected: dp_accel = 0.5 * (rho2*v2² - rho1*v1²)
        q1 = 0.5 * rho1 * v1**2
        q2 = 0.5 * rho2 * v2**2
        expected_dp = q2 - q1

        assert dp_accel == pytest.approx(expected_dp, rel=1e-10)

        # Since v2 > v1 (due to lower density), velocity head increases
        # so dp_accel should be positive (a loss)
        assert dp_accel > 0, "Expected positive dp_accel when velocity head increases"

    def test_accel_loss_area_decrease(self):
        """Test acceleration loss when area decreases (velocity increases)."""
        # Setup: same density, different areas
        m_dot = 0.01  # kg/s
        rho = 1.0  # kg/m³

        area1 = 0.02  # m² (larger upstream)
        area2 = 0.01  # m² (smaller downstream)

        v1 = m_dot / (rho * area1)
        v2 = m_dot / (rho * area2)

        dp_accel = calculate_accel_pressure_loss(rho, v1, rho, v2, C_accel=1.0)

        # v2 > v1, so velocity head increases, dp_accel positive
        assert dp_accel > 0

        # Verify exact value
        expected = 0.5 * rho * (v2**2 - v1**2)
        assert dp_accel == pytest.approx(expected, rel=1e-10)

    def test_accel_recovery(self):
        """Test that velocity decrease gives negative dp_accel (recovery)."""
        m_dot = 0.01  # kg/s
        rho = 1.0  # kg/m³

        # Area increases downstream (diffuser)
        area1 = 0.01  # m² (smaller upstream)
        area2 = 0.02  # m² (larger downstream)

        v1 = m_dot / (rho * area1)
        v2 = m_dot / (rho * area2)

        dp_accel = calculate_accel_pressure_loss(rho, v1, rho, v2, C_accel=1.0)

        # v2 < v1, so velocity head decreases, dp_accel negative (recovery)
        assert dp_accel < 0, "Expected negative dp_accel (recovery) when velocity head decreases"

    def test_c_accel_coefficient(self):
        """Test that C_accel scales the loss properly."""
        m_dot = 0.01
        rho = 1.0
        v1 = 1.0
        v2 = 2.0

        dp_base = calculate_accel_pressure_loss(rho, v1, rho, v2, C_accel=1.0)
        dp_half = calculate_accel_pressure_loss(rho, v1, rho, v2, C_accel=0.5)

        assert dp_half == pytest.approx(0.5 * dp_base, rel=1e-10)


class TestColebrookFriction:
    """Test Colebrook-White friction factor calculations."""

    def test_laminar_returns_64_over_Re(self):
        """Test that laminar flow (Re < 2300) returns f = 64/Re regardless of method."""
        Re_laminar = 1000

        f_current = friction_factor(Re_laminar, rel_rough=0.001, method="current")
        f_colebrook = friction_factor(Re_laminar, rel_rough=0.001, method="colebrook")

        expected = 64.0 / Re_laminar

        assert f_current == pytest.approx(expected, rel=1e-10)
        assert f_colebrook == pytest.approx(expected, rel=1e-10)

    def test_colebrook_turbulent_smooth(self):
        """Test Colebrook for turbulent flow in smooth pipe."""
        Re = 100000
        rel_rough = 0.0  # Smooth pipe

        f = friction_factor(Re, rel_rough, method="colebrook")

        # For smooth pipe, Colebrook should give similar to Blasius
        # f_blasius = 0.316 / Re^0.25 ≈ 0.0178 for Re=100000
        f_blasius = 0.316 / Re**0.25

        # Colebrook should be within 10% of Blasius for smooth pipes
        assert f == pytest.approx(f_blasius, rel=0.1)

        # Friction factor should be in reasonable range
        assert 0.01 < f < 0.05

    def test_colebrook_turbulent_rough(self):
        """Test Colebrook for turbulent flow in rough pipe."""
        Re = 100000
        rel_rough = 0.01  # Rough pipe

        f = friction_factor(Re, rel_rough, method="colebrook")

        # Rough pipe should have higher friction factor than smooth
        f_smooth = friction_factor(Re, 0.0, method="colebrook")

        assert f > f_smooth, "Rough pipe should have higher friction than smooth"

        # Friction factor should be in reasonable range for rough pipe
        assert 0.02 < f < 0.1

    def test_colebrook_convergence(self):
        """Test that Colebrook converges to stable value."""
        Re = 50000
        rel_rough = 0.005

        # Run twice to ensure deterministic
        f1 = friction_factor(Re, rel_rough, method="colebrook")
        f2 = friction_factor(Re, rel_rough, method="colebrook")

        assert f1 == f2, "Colebrook should give deterministic results"

    def test_colebrook_vs_swamee_jain(self):
        """Test that Colebrook and Swamee-Jain give similar results."""
        Re = 50000
        rel_rough = 0.001

        f_colebrook = friction_factor(Re, rel_rough, method="colebrook")
        f_current = friction_factor(Re, rel_rough, method="current")

        # Swamee-Jain is an explicit approximation to Colebrook
        # Should be within 1% for typical conditions
        assert f_colebrook == pytest.approx(f_current, rel=0.01)

    def test_friction_factor_bounds(self):
        """Test that friction factor stays within physical bounds."""
        test_cases = [
            (1000, 0.0),     # Laminar, smooth
            (5000, 0.0),     # Turbulent, smooth
            (100000, 0.0),   # High Re, smooth
            (5000, 0.001),   # Turbulent, slightly rough
            (5000, 0.05),    # Turbulent, very rough
        ]

        for Re, rel_rough in test_cases:
            f = friction_factor(Re, rel_rough, method="colebrook")
            assert 1e-4 <= f <= 1.0, f"f={f} out of bounds for Re={Re}, rel_rough={rel_rough}"


class TestBackwardCompatibility:
    """Test that original behavior is preserved when features are OFF."""

    def test_current_method_is_default(self):
        """Test that default friction method matches original Blasius/Swamee-Jain."""
        Re_turbulent = 50000
        rel_rough_smooth = 1e-7  # Effectively smooth
        rel_rough_rough = 0.001

        # Smooth pipe: should match Blasius
        f_smooth = friction_factor(Re_turbulent, rel_rough_smooth, method="current")
        f_blasius = 0.316 / Re_turbulent**0.25
        assert f_smooth == pytest.approx(f_blasius, rel=1e-10)

        # Rough pipe: should match Swamee-Jain
        f_rough = friction_factor(Re_turbulent, rel_rough_rough, method="current")
        f_swamee = 0.25 / (np.log10(rel_rough_rough/3.7 + 5.74/Re_turbulent**0.9))**2
        assert f_rough == pytest.approx(f_swamee, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
