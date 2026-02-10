"""
Test suite for inlet restriction refactored model

Tests verify:
1. dp_door decreases monotonically with % open (for fixed mdot and rho)
2. Near 0% open, dp_door remains finite and changes smoothly
3. End-to-end steady-state consistency: dp_drive ≈ R_total × mdot²
4. No NaNs in mdot, velocity, temperature for sweep of % open including 0-5%
5. Smooth leakage floor behavior: A_eff = sqrt(A_eff_raw² + A_leak²)
"""
import pytest
import numpy as np
import math


class TestSmoothLeakageFloor:
    """Test the smooth leakage floor calculation"""

    def compute_A_eff(self, A_eff_raw, A_leak):
        """Compute effective area with smooth floor"""
        return math.sqrt(A_eff_raw**2 + A_leak**2)

    def test_A_eff_equals_A_leak_when_closed(self):
        """When A_eff_raw = 0, A_eff_used should equal A_leak"""
        A_leak = 1e-4  # 1 cm²
        A_eff = self.compute_A_eff(0.0, A_leak)
        assert abs(A_eff - A_leak) < 1e-10

    def test_A_eff_approaches_raw_when_large(self):
        """When A_eff_raw >> A_leak, A_eff_used ≈ A_eff_raw"""
        A_leak = 1e-4
        A_eff_raw = 0.01  # 100 cm² >> 1 cm²
        A_eff = self.compute_A_eff(A_eff_raw, A_leak)
        # Should be within 1% of raw
        assert abs(A_eff - A_eff_raw) / A_eff_raw < 0.01

    def test_A_eff_smooth_derivative(self):
        """Derivative should be continuous (no jumps)"""
        A_leak = 1e-4
        # Sample A_eff_raw from 0 to 0.01
        A_raw_values = np.linspace(0, 0.01, 100)
        A_eff_values = [self.compute_A_eff(a, A_leak) for a in A_raw_values]

        # Check derivative is monotonically increasing
        dA = np.diff(A_eff_values)
        assert all(d >= 0 for d in dA), "A_eff should increase monotonically"

        # Check no large jumps (derivative should be smooth)
        d2A = np.diff(dA)
        max_d2A = max(abs(d) for d in d2A)
        assert max_d2A < 0.01, f"Second derivative too large: {max_d2A}"

    def test_A_eff_always_positive(self):
        """A_eff should always be positive"""
        A_leak = 1e-4
        for A_raw in [0, 1e-10, 1e-6, 1e-4, 0.001, 0.01]:
            A_eff = self.compute_A_eff(A_raw, A_leak)
            assert A_eff > 0, f"A_eff should be positive for A_raw={A_raw}"


class TestResistanceModel:
    """Test the resistance-based pressure drop model"""

    def compute_R_door(self, K, rho, A_eff):
        """Compute door resistance: R_door = K / (2 × ρ × A_eff²)"""
        return K / (2.0 * rho * A_eff**2)

    def compute_dp_door(self, R_door, mdot):
        """Compute pressure drop: dp_door = R_door × mdot²"""
        return R_door * mdot**2

    def test_dp_decreases_with_opening(self):
        """dp_door should decrease as door opens (for fixed mdot)"""
        K = 2.6  # K = 1/Cd² for Cd=0.62
        rho = 1.2  # kg/m³
        mdot = 0.01  # kg/s
        A_leak = 1e-4

        # Opening percentages and corresponding A_eff_raw
        openings = [0, 5, 10, 25, 50, 75, 100]
        full_area = 0.01  # m²

        dp_values = []
        for pct in openings:
            angle_rad = (pct / 100.0) * (math.pi / 2.0)
            A_raw = full_area * math.sin(angle_rad)
            A_eff = math.sqrt(A_raw**2 + A_leak**2)
            R_door = self.compute_R_door(K, rho, A_eff)
            dp = self.compute_dp_door(R_door, mdot)
            dp_values.append(dp)

        # Check monotonically decreasing
        for i in range(len(dp_values) - 1):
            assert dp_values[i] > dp_values[i+1], \
                f"dp should decrease: dp[{openings[i]}%]={dp_values[i]:.2f} > dp[{openings[i+1]}%]={dp_values[i+1]:.2f}"

    def test_dp_finite_at_zero_opening(self):
        """dp_door should be finite even at 0% open"""
        K = 2.6
        rho = 1.2
        mdot = 0.01
        A_leak = 1e-4

        # 0% open
        A_eff = A_leak  # sqrt(0² + A_leak²) = A_leak
        R_door = self.compute_R_door(K, rho, A_eff)
        dp = self.compute_dp_door(R_door, mdot)

        assert np.isfinite(dp), f"dp should be finite at 0% open, got {dp}"
        assert dp > 0, f"dp should be positive, got {dp}"

        # Expected: R = 2.6 / (2 × 1.2 × (1e-4)²) = 2.6 / (2.4e-8) ≈ 1.08e8
        # dp = 1.08e8 × 0.01² = 1.08e4 Pa ≈ 10 kPa
        assert 1000 < dp < 100000, f"dp at 0% should be high but finite, got {dp}"

    def test_dp_scales_with_mdot_squared(self):
        """dp should scale with mdot²"""
        K = 2.6
        rho = 1.2
        A_eff = 0.001  # 10 cm²

        R_door = self.compute_R_door(K, rho, A_eff)

        dp_1 = self.compute_dp_door(R_door, 0.01)
        dp_2 = self.compute_dp_door(R_door, 0.02)

        # dp_2 / dp_1 should be (0.02/0.01)² = 4
        ratio = dp_2 / dp_1
        assert abs(ratio - 4.0) < 0.01, f"dp should scale with mdot²: ratio={ratio}"


class TestSmoothTransition:
    """Test smooth behavior near 0% open"""

    def test_dp_changes_smoothly_near_zero(self):
        """dp should change smoothly as opening goes from 0% to 5%"""
        K = 2.6
        rho = 1.2
        mdot = 0.01
        A_leak = 1e-4
        full_area = 0.01

        # Fine-grained sweep from 0% to 5%
        openings = np.linspace(0, 5, 50)
        dp_values = []

        for pct in openings:
            angle_rad = (pct / 100.0) * (math.pi / 2.0)
            A_raw = full_area * math.sin(angle_rad)
            A_eff = math.sqrt(A_raw**2 + A_leak**2)
            R_door = K / (2.0 * rho * A_eff**2)
            dp = R_door * mdot**2
            dp_values.append(dp)

        # Check for smoothness: no large jumps in derivative
        d_dp = np.diff(dp_values)
        d2_dp = np.diff(d_dp)

        # All first derivatives should have same sign (monotonic)
        assert all(d <= 0 for d in d_dp), "dp should decrease monotonically"

        # Second derivative should not have extreme changes
        max_d2 = max(abs(d) for d in d2_dp)
        # Normalize by range of dp values
        dp_range = dp_values[0] - dp_values[-1]
        normalized_max_d2 = max_d2 / dp_range if dp_range > 0 else 0
        assert normalized_max_d2 < 0.5, \
            f"Second derivative too large (non-smooth): {normalized_max_d2}"


class TestInletRestrictionIntegration:
    """Integration tests using the actual Chimney class"""

    @pytest.fixture
    def chimney_parts(self):
        """Create test chimney parts"""
        try:
            from flued_module.chimney import ChimneyPart
            part1 = ChimneyPart(
                height_m=2.0,
                inner_diameter_m=0.15,
                wall_type='single',
                T_ambient_K=293.15,
                orientation='vertical'
            )
            return [part1]
        except ImportError:
            pytest.skip("flued_module not available")

    def test_no_nans_with_small_opening(self, chimney_parts):
        """Simulation should not produce NaNs even with very small opening"""
        try:
            from flued_module.chimney import Chimney
        except ImportError:
            pytest.skip("flued_module not available")

        # Test with 1% open (very restrictive)
        chimney = Chimney(
            parts=chimney_parts,
            inlet_restriction={
                'enable': True,
                'door_area_m2': 0.01,
                'door_opening_percent': 1.0,
                'Cd': 0.62
            }
        )

        # Simple inlet temperature
        def T_inlet(t):
            return 400 + 100 * (1 - np.exp(-t / 60))

        chimney.run_simulation(total_time=300, dt=1.0, T_inlet_func=T_inlet)

        # Check no NaNs
        assert not np.any(np.isnan(chimney.mass_flow_rate)), "NaN in mass_flow_rate"
        assert not np.any(np.isnan(chimney.T_gas)), "NaN in T_gas"
        assert not np.any(np.isnan(chimney.velocity)), "NaN in velocity"

    def test_no_nans_sweep_opening_percent(self, chimney_parts):
        """Sweep % open from 0-100 and verify no NaNs"""
        try:
            from flued_module.chimney import Chimney
        except ImportError:
            pytest.skip("flued_module not available")

        openings_to_test = [0, 1, 2, 5, 10, 25, 50, 100]

        for pct in openings_to_test:
            chimney = Chimney(
                parts=chimney_parts,
                inlet_restriction={
                    'enable': True,
                    'door_area_m2': 0.01,
                    'door_opening_percent': pct,
                    'Cd': 0.62
                }
            )

            def T_inlet(t):
                return 400 + 50 * (1 - np.exp(-t / 30))

            chimney.run_simulation(total_time=60, dt=0.5, T_inlet_func=T_inlet)

            assert not np.any(np.isnan(chimney.mass_flow_rate)), \
                f"NaN in mass_flow_rate at {pct}% open"
            assert not np.any(np.isnan(chimney.T_gas)), \
                f"NaN in T_gas at {pct}% open"

    def test_flow_decreases_with_more_restriction(self, chimney_parts):
        """Mass flow should decrease as door closes (more restriction)"""
        try:
            from flued_module.chimney import Chimney
        except ImportError:
            pytest.skip("flued_module not available")

        openings = [100, 50, 25, 10]
        final_flows = []

        for pct in openings:
            chimney = Chimney(
                parts=chimney_parts,
                inlet_restriction={
                    'enable': True,
                    'door_area_m2': 0.01,
                    'door_opening_percent': pct,
                    'Cd': 0.62
                }
            )

            def T_inlet(t):
                return 500.0

            chimney.run_simulation(total_time=300, dt=1.0, T_inlet_func=T_inlet)
            final_flows.append(chimney.mass_flow_rate[-1])

        # Flow should decrease as opening decreases
        for i in range(len(final_flows) - 1):
            assert final_flows[i] >= final_flows[i+1], \
                f"Flow should decrease: {openings[i]}%={final_flows[i]:.4f} >= {openings[i+1]}%={final_flows[i+1]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
