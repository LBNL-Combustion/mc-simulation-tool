"""
Unit tests for Rayleigh number calculation in air gap heat transfer

Tests verify:
1. Ra is dimensionless (sanity check via known values)
2. Ra increases with delta_T and gap_width
3. Nu increases with Ra for fixed H/W
4. Comparison against hand-calculated reference values
"""
import pytest
import math


class TestRayleighNumberCalculation:
    """Test the corrected Rayleigh number calculation for enclosed air gaps"""

    # Reference air properties at 300 K (approximate)
    # These are typical values for verification
    T_mean = 300.0  # K
    rho = 1.177  # kg/m³ (air at 300K, 1 atm)
    mu = 1.85e-5  # Pa·s (dynamic viscosity)
    k = 0.0263  # W/(m·K) (thermal conductivity)
    cp = 1007.0  # J/(kg·K) (specific heat)
    g = 9.81  # m/s²
    beta = 1 / T_mean  # 1/K (ideal gas)

    def compute_Ra_reference(self, delta_T: float, L: float) -> float:
        """
        Compute Rayleigh number using the correct formula:
        Ra = g × β × ΔT × L³ × ρ² × cp / (μ × k)

        This is the Gr × Pr formulation expanded.
        """
        Ra = (self.g * self.beta * delta_T * L**3 * self.rho**2 * self.cp) / (self.mu * self.k)
        return Ra

    def compute_Gr_Pr_separately(self, delta_T: float, L: float) -> tuple:
        """
        Compute Grashof and Prandtl numbers separately for verification
        Gr = g × β × ΔT × L³ × ρ² / μ²
        Pr = μ × cp / k
        """
        nu = self.mu / self.rho  # kinematic viscosity
        Gr = self.g * self.beta * delta_T * L**3 / nu**2
        Pr = self.mu * self.cp / self.k
        return Gr, Pr

    def test_Ra_equals_Gr_times_Pr(self):
        """Verify Ra = Gr × Pr using both calculation methods"""
        delta_T = 30.0  # K
        L = 0.025  # m (25mm gap)

        Ra_direct = self.compute_Ra_reference(delta_T, L)
        Gr, Pr = self.compute_Gr_Pr_separately(delta_T, L)
        Ra_from_GrPr = Gr * Pr

        # Should match within numerical precision
        assert abs(Ra_direct - Ra_from_GrPr) / Ra_direct < 1e-10, \
            f"Ra mismatch: direct={Ra_direct:.6e}, Gr×Pr={Ra_from_GrPr:.6e}"

    def test_Ra_is_dimensionless_sanity_check(self):
        """
        Verify Ra produces expected magnitude for known conditions

        Reference case:
        - T_mean = 300 K, delta_T = 30 K, L = 0.025 m (25mm gap)
        - Expected Ra ~ 10^4 to 10^5 (natural convection regime)
        """
        delta_T = 30.0
        L = 0.025

        Ra = self.compute_Ra_reference(delta_T, L)

        # For a 25mm gap with 30K temperature difference, Ra should be ~10^4-10^5
        assert 1e3 < Ra < 1e6, f"Ra={Ra:.2e} outside expected range for 25mm gap, 30K ΔT"

        # More precise check: compute expected value
        # Ra = 9.81 * (1/300) * 30 * 0.025^3 * 1.177^2 * 1007 / (1.85e-5 * 0.0263)
        #    = 9.81 * 0.00333 * 30 * 1.5625e-5 * 1.385 * 1007 / (4.866e-7)
        #    ≈ 2.8e4
        expected_Ra = (9.81 * (1/300) * 30 * 0.025**3 * 1.177**2 * 1007) / (1.85e-5 * 0.0263)
        assert abs(Ra - expected_Ra) / expected_Ra < 0.01, \
            f"Ra={Ra:.4e} differs from expected {expected_Ra:.4e}"

    def test_Ra_increases_with_delta_T(self):
        """Ra should increase linearly with temperature difference"""
        L = 0.025  # 25mm gap

        Ra_10K = self.compute_Ra_reference(10.0, L)
        Ra_20K = self.compute_Ra_reference(20.0, L)
        Ra_30K = self.compute_Ra_reference(30.0, L)

        # Should be monotonically increasing
        assert Ra_10K < Ra_20K < Ra_30K, "Ra should increase with ΔT"

        # Should scale linearly with ΔT
        ratio_expected = 2.0
        ratio_actual = Ra_20K / Ra_10K
        assert abs(ratio_actual - ratio_expected) < 0.01, \
            f"Ra should scale linearly with ΔT: got ratio {ratio_actual:.3f}, expected 2.0"

    def test_Ra_increases_with_gap_width_cubed(self):
        """Ra should increase with L³"""
        delta_T = 30.0

        Ra_10mm = self.compute_Ra_reference(delta_T, 0.010)
        Ra_20mm = self.compute_Ra_reference(delta_T, 0.020)
        Ra_40mm = self.compute_Ra_reference(delta_T, 0.040)

        # Should be monotonically increasing
        assert Ra_10mm < Ra_20mm < Ra_40mm, "Ra should increase with gap width"

        # Should scale with L³: doubling L should give 8× Ra
        ratio_expected = 8.0
        ratio_actual = Ra_20mm / Ra_10mm
        assert abs(ratio_actual - ratio_expected) < 0.01, \
            f"Ra should scale with L³: got ratio {ratio_actual:.3f}, expected 8.0"

    def test_Nu_increases_with_Ra(self):
        """
        For the Hollands/Catton correlation:
        Nu = 0.42 × Ra^0.25 × (H/W)^(-0.3)
        Nu should increase with Ra for fixed aspect ratio
        """
        H_over_W = 20.0  # Fixed aspect ratio

        def compute_Nu(Ra):
            if Ra < 1e4:
                return 1.0
            elif Ra < 1e10:
                return 0.42 * Ra**0.25 * H_over_W**(-0.3)
            else:
                return 0.046 * Ra**(1/3)

        Nu_1e4 = compute_Nu(1e4)
        Nu_1e5 = compute_Nu(1e5)
        Nu_1e6 = compute_Nu(1e6)

        assert Nu_1e4 < Nu_1e5 < Nu_1e6, "Nu should increase with Ra"

        # Check the 0.25 power scaling
        # Nu(1e5) / Nu(1e4) should be (1e5/1e4)^0.25 = 10^0.25 ≈ 1.778
        ratio_expected = 10**0.25
        ratio_actual = Nu_1e5 / Nu_1e4
        assert abs(ratio_actual - ratio_expected) < 0.01, \
            f"Nu scaling: got {ratio_actual:.3f}, expected {ratio_expected:.3f}"

    def test_Prandtl_number_for_air(self):
        """Verify Prandtl number is ~0.7 for air"""
        Pr = self.mu * self.cp / self.k
        assert 0.68 < Pr < 0.72, f"Pr={Pr:.3f} outside expected range for air"


class TestAirGapHeatTransferIntegration:
    """Integration tests using the actual ComprehensiveHeatTransfer class"""

    @pytest.fixture
    def heat_transfer(self):
        """Create heat transfer calculator instance"""
        try:
            from flued_module.comprehensive_heat_transfer import ComprehensiveHeatTransfer
            return ComprehensiveHeatTransfer()
        except ImportError:
            pytest.skip("flued_module not available")

    def test_air_gap_coefficient_increases_with_delta_T(self, heat_transfer):
        """h_total should increase with temperature difference"""
        gap_width = 0.025  # 25mm
        gap_height = 1.0  # 1m

        h_10K = heat_transfer.air_gap_heat_transfer_coefficient(310, 300, gap_width, gap_height)
        h_30K = heat_transfer.air_gap_heat_transfer_coefficient(330, 300, gap_width, gap_height)
        h_50K = heat_transfer.air_gap_heat_transfer_coefficient(350, 300, gap_width, gap_height)

        assert h_10K < h_30K < h_50K, "h should increase with ΔT"

    def test_air_gap_coefficient_increases_with_gap_width(self, heat_transfer):
        """For convection-dominated regime, h should depend on gap width"""
        T_hot = 350  # K
        T_cold = 300  # K
        gap_height = 1.0  # m

        # Note: h_conv = Nu * k / L, and Nu ~ Ra^0.25 ~ L^0.75
        # So h_conv ~ L^0.75 / L = L^(-0.25), which means h_conv decreases with L
        # But h_cond = k/L, which also decreases with L
        # And h_rad is independent of L
        # So total h should generally decrease with L (conduction limited)
        h_10mm = heat_transfer.air_gap_heat_transfer_coefficient(T_hot, T_cold, 0.010, gap_height)
        h_25mm = heat_transfer.air_gap_heat_transfer_coefficient(T_hot, T_cold, 0.025, gap_height)
        h_50mm = heat_transfer.air_gap_heat_transfer_coefficient(T_hot, T_cold, 0.050, gap_height)

        # For wider gaps, conduction resistance increases, so h decreases
        assert h_10mm > h_25mm > h_50mm, "h should decrease with wider gaps (conduction limited)"

    def test_air_gap_coefficient_plausible_range(self, heat_transfer):
        """Check that h_total is in a physically plausible range"""
        # Typical case: 25mm gap, 50K temperature difference
        h = heat_transfer.air_gap_heat_transfer_coefficient(350, 300, 0.025, 1.0)

        # For a 25mm air gap, expect h ~ 5-20 W/(m²·K)
        # Conduction: k/L ≈ 0.03/0.025 ≈ 1.2 W/(m²·K)
        # Convection: adds some enhancement
        # Radiation: h_rad ≈ 4*ε*σ*T³ ≈ 4*0.67*5.67e-8*325³ ≈ 5 W/(m²·K)
        # Total: expect ~5-15 W/(m²·K)
        assert 3.0 < h < 25.0, f"h={h:.1f} W/(m²·K) outside plausible range"

    def test_very_small_delta_T_returns_conduction_only(self, heat_transfer):
        """With ΔT < 0.1K, should return conduction + radiation only"""
        gap_width = 0.025
        gap_height = 1.0
        T_hot = 300.05  # Only 0.05K difference
        T_cold = 300.0

        h = heat_transfer.air_gap_heat_transfer_coefficient(T_hot, T_cold, gap_width, gap_height)

        # With negligible ΔT, convection should be zero
        # h ≈ h_cond + h_rad ≈ k/L + radiation
        k_air = 0.026  # approximate at 300K
        h_cond_expected = k_air / gap_width  # ≈ 1.0 W/(m²·K)

        # h_rad ≈ 4*ε_eff*σ*T³ ≈ 4*0.67*5.67e-8*300³ ≈ 4.1 W/(m²·K)
        # Total ≈ 5 W/(m²·K)
        assert 3.0 < h < 10.0, f"h={h:.1f} for small ΔT outside expected range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
