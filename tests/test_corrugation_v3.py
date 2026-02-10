"""
Test suite for corrugation_effects.py v3.0 refactor

Tests verify:
1. K_total grows with length/num_starts, shrinks with pitch
2. K_seams clamping at K_seams_max
3. Spiral-seam duct does NOT modify roughness via c_spiral
4. Friction/HT multipliers remain within caps (<=1.15, <=1.25)
5. Pitch validation (pitch <= 0 raises ValueError)
6. Model separation between spiral-seam and annular corrugation
"""
import warnings
import pytest
import numpy as np

from flued_module.corrugation_effects import (
    SpiralCorrugationProperties,
    CorrugationProperties,
    spiral_equivalents,
    compute_spiral_corrugation_effects,
    swirl_friction_multiplier,
    spiral_HT_multiplier,
    spiral_descriptors,
    compute_spiral_factor,
    CorrugatedPipeFriction,
)


class TestKTotalMonotonicity:
    """Test that K_total behaves correctly with geometry changes"""

    # Standard test parameters
    D = 0.1016  # 4 inches
    Re = 20000
    depth = 0.003  # 3mm
    pitch = 0.180  # 180mm
    num_starts = 4

    def test_K_total_increases_with_length(self):
        """K_total should increase with pipe length"""
        K_L1 = spiral_equivalents(self.D, 1.0, self.Re, self.depth, self.pitch, self.num_starts)['K_total']
        K_L2 = spiral_equivalents(self.D, 2.0, self.Re, self.depth, self.pitch, self.num_starts)['K_total']
        K_L4 = spiral_equivalents(self.D, 4.0, self.Re, self.depth, self.pitch, self.num_starts)['K_total']

        assert K_L1 < K_L2 < K_L4, f"K_total should increase with length: {K_L1}, {K_L2}, {K_L4}"
        # Should be roughly linear
        assert abs(K_L2 - 2*K_L1) < 0.1, "K_total should scale roughly linearly with length"

    def test_K_total_increases_with_num_starts(self):
        """K_total should increase with number of spiral starts"""
        K_n1 = spiral_equivalents(self.D, 2.0, self.Re, self.depth, self.pitch, 1)['K_total']
        K_n4 = spiral_equivalents(self.D, 2.0, self.Re, self.depth, self.pitch, 4)['K_total']
        K_n6 = spiral_equivalents(self.D, 2.0, self.Re, self.depth, self.pitch, 6)['K_total']

        assert K_n1 < K_n4 < K_n6, f"K_total should increase with num_starts: {K_n1}, {K_n4}, {K_n6}"

    def test_K_total_decreases_with_pitch(self):
        """K_total should decrease as pitch increases (fewer seam crossings)"""
        K_p1 = spiral_equivalents(self.D, 2.0, self.Re, self.depth, 0.10, self.num_starts)['K_total']
        K_p2 = spiral_equivalents(self.D, 2.0, self.Re, self.depth, 0.18, self.num_starts)['K_total']
        K_p3 = spiral_equivalents(self.D, 2.0, self.Re, self.depth, 0.30, self.num_starts)['K_total']

        assert K_p1 > K_p2 > K_p3, f"K_total should decrease with pitch: {K_p1}, {K_p2}, {K_p3}"


class TestKSeamsClamping:
    """Test K_seams clamping guardrails"""

    D = 0.1016
    Re = 20000
    depth = 0.003

    def test_K_seams_clamped_for_extreme_params(self):
        """K_seams should be clamped when it exceeds K_seams_max"""
        # Extreme: 20m length, 50mm pitch, 6 starts -> huge K
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = spiral_equivalents(
                self.D, 20.0, self.Re, self.depth,
                pitch_m=0.05, num_starts=6, K_seams_max=5.0
            )

            # Should have emitted a warning
            assert any("K_seams" in str(warn.message) for warn in w), \
                "Should warn when K_seams exceeds limit"

        assert result['K_seams_clamped'] == True
        assert result['debug']['K_seams'] <= 5.0
        assert result['debug']['K_seams_raw'] > result['debug']['K_seams']

    def test_K_seams_not_clamped_for_normal_params(self):
        """K_seams should not be clamped for reasonable parameters"""
        result = spiral_equivalents(
            self.D, 2.0, self.Re, self.depth,
            pitch_m=0.18, num_starts=4, K_seams_max=5.0
        )

        assert result['K_seams_clamped'] == False
        assert result['debug']['K_seams'] == result['debug']['K_seams_raw']

    def test_custom_K_seams_max(self):
        """Should respect custom K_seams_max value"""
        result_low_cap = spiral_equivalents(
            self.D, 5.0, self.Re, self.depth,
            pitch_m=0.10, num_starts=4, K_seams_max=1.0
        )
        result_high_cap = spiral_equivalents(
            self.D, 5.0, self.Re, self.depth,
            pitch_m=0.10, num_starts=4, K_seams_max=10.0
        )

        # Same raw K, but different clamped values
        assert result_low_cap['debug']['K_seams_raw'] == result_high_cap['debug']['K_seams_raw']
        assert result_low_cap['debug']['K_seams'] <= 1.0
        assert result_high_cap['debug']['K_seams'] <= 10.0


class TestRoughnessNoCSpiral:
    """Test that v3.0 does NOT apply c_spiral to roughness for spiral-seam duct"""

    D = 0.1016
    L = 2.0
    Re = 20000
    depth = 0.003
    pitch = 0.18
    num_starts = 4
    base_roughness = 1e-6  # 1 micron

    def test_roughness_equals_base_roughness(self):
        """Roughness should equal base_roughness (c_spiral NOT applied)"""
        result = spiral_equivalents(
            self.D, self.L, self.Re, self.depth, self.pitch, self.num_starts,
            base_roughness_m=self.base_roughness,
            include_seam_roughness=False
        )

        assert result['roughness_m'] == self.base_roughness, \
            f"v3.0: roughness should equal base_roughness, got {result['roughness_m']}"

    def test_c_spiral_not_used_in_roughness(self):
        """c_spiral should be in debug output but NOT applied to roughness"""
        result = spiral_equivalents(
            self.D, self.L, self.Re, self.depth, self.pitch, self.num_starts,
            base_roughness_m=self.base_roughness
        )

        c_spiral = result['debug']['spiral_factor_diagnostic']
        assert c_spiral > 0, "c_spiral diagnostic should be computed"

        # If c_spiral were applied: roughness = base + c_spiral * depth
        old_roughness = self.base_roughness + c_spiral * self.depth

        # v3.0: c_spiral is NOT applied
        assert result['roughness_m'] != old_roughness, \
            "c_spiral should NOT be applied to roughness in v3.0"
        assert result['roughness_m'] == self.base_roughness

    def test_optional_seam_roughness_heuristic(self):
        """Optional seam_protrusion can be added to roughness"""
        seam_protrusion = 0.002  # 2mm

        result = spiral_equivalents(
            self.D, self.L, self.Re, self.depth, self.pitch, self.num_starts,
            base_roughness_m=self.base_roughness,
            include_seam_roughness=True,
            seam_protrusion_m=seam_protrusion
        )

        expected = self.base_roughness + seam_protrusion
        assert result['roughness_m'] == expected, \
            f"With seam roughness: expected {expected}, got {result['roughness_m']}"

    def test_swirl_does_not_change_roughness(self):
        """Enabling swirl should NOT change roughness (only multipliers)"""
        result_no_swirl = spiral_equivalents(
            self.D, self.L, self.Re, self.depth, self.pitch, self.num_starts,
            enable_swirl=False, enable_HT=False
        )
        result_with_swirl = spiral_equivalents(
            self.D, self.L, self.Re, self.depth, self.pitch, self.num_starts,
            enable_swirl=True, enable_HT=True
        )

        assert result_no_swirl['roughness_m'] == result_with_swirl['roughness_m'], \
            "v3.0: Swirl should not change roughness"


class TestMultiplierCaps:
    """Test that friction/HT multipliers stay within caps"""

    D = 0.1016
    depth = 0.003
    pitch = 0.18
    Re = 20000

    def test_friction_multiplier_within_cap(self):
        """Friction multiplier should be <= 1.15"""
        desc = spiral_descriptors(self.D, self.depth, self.pitch)

        m_f = swirl_friction_multiplier(
            desc['swirl_index'], desc['rel_depth'], desc['rel_pitch'], self.Re
        )

        assert 1.0 <= m_f <= 1.15, f"m_f = {m_f} out of bounds [1.0, 1.15]"

    def test_HT_multiplier_within_cap(self):
        """HT multiplier should be <= 1.25"""
        desc = spiral_descriptors(self.D, self.depth, self.pitch)

        M_h = spiral_HT_multiplier(
            desc['swirl_index'], desc['rel_depth'], desc['rel_pitch'], self.Re
        )

        assert 1.0 <= M_h <= 1.25, f"M_h = {M_h} out of bounds [1.0, 1.25]"

    def test_caps_enforced_for_extreme_geometry(self):
        """Caps should be enforced even for extreme geometry"""
        # Large depth that would push multipliers high
        desc_large = spiral_descriptors(self.D, depth=0.02, pitch=self.pitch)

        m_f = swirl_friction_multiplier(
            desc_large['swirl_index'], desc_large['rel_depth'],
            desc_large['rel_pitch'], self.Re,
            swirl_friction_cap=0.15
        )
        M_h = spiral_HT_multiplier(
            desc_large['swirl_index'], desc_large['rel_depth'],
            desc_large['rel_pitch'], self.Re,
            HT_enhancement_cap=0.25
        )

        assert m_f <= 1.15, f"Friction cap not enforced: m_f = {m_f}"
        assert M_h <= 1.25, f"HT cap not enforced: M_h = {M_h}"

    def test_custom_caps_respected(self):
        """Custom cap values should be respected"""
        desc = spiral_descriptors(self.D, depth=0.02, pitch=self.pitch)

        m_f_tight = swirl_friction_multiplier(
            desc['swirl_index'], desc['rel_depth'], desc['rel_pitch'], self.Re,
            swirl_friction_cap=0.05  # Tighter cap
        )

        assert m_f_tight <= 1.05, f"Custom cap not enforced: m_f = {m_f_tight}"


class TestPitchValidation:
    """Test pitch input validation"""

    D = 0.1016
    L = 2.0
    Re = 20000
    depth = 0.003

    def test_zero_pitch_raises_error(self):
        """pitch = 0 should raise ValueError"""
        with pytest.raises(ValueError, match="pitch_m must be > 0"):
            spiral_equivalents(self.D, self.L, self.Re, self.depth, pitch_m=0.0, num_starts=4)

    def test_negative_pitch_raises_error(self):
        """pitch < 0 should raise ValueError"""
        with pytest.raises(ValueError, match="pitch_m must be > 0"):
            spiral_equivalents(self.D, self.L, self.Re, self.depth, pitch_m=-0.1, num_starts=4)

    def test_positive_pitch_works(self):
        """Positive pitch should work fine"""
        result = spiral_equivalents(self.D, self.L, self.Re, self.depth, pitch_m=0.18, num_starts=4)
        assert 'K_total' in result


class TestModelVersion:
    """Test model version tracking"""

    def test_version_3_reported(self):
        """Model version should be 3.0"""
        result = spiral_equivalents(0.1016, 2.0, 20000, 0.003, 0.18, 4)
        assert result['debug']['model_version'] == '3.0'


class TestDataclassUpdates:
    """Test that SpiralCorrugationProperties has v3.0 fields"""

    def test_K_seams_max_field_exists(self):
        """SpiralCorrugationProperties should have K_seams_max"""
        props = SpiralCorrugationProperties(
            depth_m=0.003, pitch_m=0.18, num_starts=4, base_roughness_m=1e-6
        )
        assert hasattr(props, 'K_seams_max')
        assert props.K_seams_max == 5.0  # default

    def test_include_seam_roughness_field_exists(self):
        """SpiralCorrugationProperties should have include_seam_roughness"""
        props = SpiralCorrugationProperties(
            depth_m=0.003, pitch_m=0.18, num_starts=4, base_roughness_m=1e-6
        )
        assert hasattr(props, 'include_seam_roughness')
        assert props.include_seam_roughness == False  # default

    def test_compute_spiral_corrugation_effects_uses_K_seams_max(self):
        """compute_spiral_corrugation_effects should respect K_seams_max from props"""
        props = SpiralCorrugationProperties(
            depth_m=0.003, pitch_m=0.05, num_starts=6, base_roughness_m=1e-6,
            K_seams_max=1.0  # Very low cap
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = compute_spiral_corrugation_effects(
                diameter_m=0.1016, length_m=10.0, spiral_props=props
            )

        assert result['K_total_seams'] <= 1.03  # K_seams_max + K_lock


class TestModelSeparation:
    """Test that spiral-seam and annular corrugation models are separate"""

    def test_CorrugationProperties_no_seam_fields(self):
        """CorrugationProperties (annular) should NOT have seam fields"""
        props = CorrugationProperties(
            depth_m=0.003, pitch_m=0.012, base_roughness_m=1e-6
        )

        assert not hasattr(props, 'K_seam_per_crossing')
        assert not hasattr(props, 'num_starts')
        assert not hasattr(props, 'K_seams_max')

    def test_CorrugatedPipeFriction_no_spiral_factor(self):
        """CorrugatedPipeFriction should use C_form, not c_spiral"""
        friction_calc = CorrugatedPipeFriction()
        corr = CorrugationProperties(
            depth_m=0.003, pitch_m=0.012, base_roughness_m=1e-6
        )

        f = friction_calc.calculate_corrugated_friction_factor(20000, 0.1016, corr)

        # Should be enhanced friction, not base
        f_base = friction_calc._colebrook_white(20000, 1e-6 / 0.1016)
        assert f > f_base, "Corrugated friction should be enhanced"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
