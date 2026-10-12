"""Checks that chimney buoyancy is referenced to outdoor air."""
import numpy as np
import pytest

from flued_module.chimney import Chimney
from flued_module.material_properties import MaterialProperties

MAT = MaterialProperties('flued_module/data/pipe_props.json')
G = 9.81
T_IN, T_OUT = 288.7, 249.8          # 15.6 C indoors, -23.3 C outdoors
H_IN_FT, H_OUT_FT = 7.0, 8.0


def build(T_in=T_IN, T_out=T_OUT, total_time=4.0):
    parts = [MAT.create_single_wall_chimney_part(
                 inner_diameter_ft=0.5, height_ft=H_IN_FT, z_ft=0, material='steel_24_gauge',
                 orientation='vertical', T_ambient_K=T_in, num_segments=7),
             MAT.create_triple_wall_chimney_part(
                 inner_diameter_ft=0.5, height_ft=H_OUT_FT, z_ft=0, inner_material='stainless_430',
                 middle_material='ceramic_fiber', outer_material='stainless_316',
                 orientation='vertical', T_ambient_K=T_out, num_segments=8)]
    return Chimney(parts=parts, dt=1.0, total_time=total_time, use_coupled_solver=True,
                   inlet_temp_func=lambda t: T_in, verbose=False)


def rho(ch, T, Pa):
    return ch.physics.air_density(T, Pa)


def ambient_column_densities(ch):
    return np.array([rho(ch, ch.get_segment_info(s)[2].T_ambient_K, ch.get_segment_info(s)[2].Pa)
                     for s in range(ch.total_segments)])


def test_column_at_room_temperature_has_outdoor_stack_effect():
    """Indoor section full of room-temperature gas is lighter than outdoor air, so it pulls."""
    ch = build()
    p0 = ch.parts[0]
    b = ch.coupled_solver._calculate_buoyancy(ch, ambient_column_densities(ch))
    h0 = sum(ch.segment_vertical_heights[:p0.num_segments])
    expected = (rho(ch, T_OUT, p0.Pa) - rho(ch, T_IN, p0.Pa)) * G * h0
    assert expected > 3.0
    assert np.sum(b) == pytest.approx(expected, rel=1e-3)


def test_fire_potential_is_zero_when_gas_is_at_outdoor_temperature():
    ch = build()
    assert ch.coupled_solver._compute_fire_driving_potential_first_part(ch, T_OUT) == \
        pytest.approx(0.0, abs=1e-9)


def test_profile_top_is_anchored_to_outdoor_column():
    ch = build()
    pr = ch.get_static_pressure_profile(timestep=0)
    top_out = ch.parts[0].Pa - sum(
        rho(ch, T_OUT, ch.get_segment_info(s)[2].Pa) * G * ch.segment_vertical_heights[s]
        for s in range(ch.total_segments))
    assert pr['P_absolute_Pa'][-1] == pytest.approx(top_out, abs=1e-6)


def test_gauge_reference_is_room_column_indoors_and_outdoor_column_outdoors():
    ch = build()
    n0 = ch.parts[0].num_segments
    room, outdoor = [ch.parts[0].Pa], [ch.parts[0].Pa]
    for s in range(ch.total_segments):
        part = ch.get_segment_info(s)[2]
        dz = ch.segment_vertical_heights[s]
        room.append(room[-1] - rho(ch, part.T_ambient_K, part.Pa) * G * dz)
        outdoor.append(outdoor[-1] - rho(ch, T_OUT, part.Pa) * G * dz)
    pr = ch.get_static_pressure_profile(timestep=0)
    assert pr['P_ambient_Pa'][:n0] == pytest.approx(room[:n0], abs=1e-6)
    assert pr['P_ambient_Pa'][n0:] == pytest.approx(outdoor[n0:], abs=1e-6)
    assert pr['P_gauge_Pa'] == pytest.approx(pr['P_absolute_Pa'] - pr['P_ambient_Pa'], abs=1e-6)


def test_no_change_when_indoor_equals_outdoor():
    """With one ambient temperature everywhere the room and outdoor columns coincide."""
    ch = build(T_in=270.0, T_out=270.0)
    b = ch.coupled_solver._calculate_buoyancy(ch, ambient_column_densities(ch))
    assert np.sum(b) == pytest.approx(0.0, abs=1e-9)

    ch.T_gas[0, :] = 270.0
    ch.density[0, :] = ambient_column_densities(ch)
    pr = ch.get_static_pressure_profile(timestep=0)
    assert pr['P_ambient_Pa'] == pytest.approx(pr['P_absolute_Pa'] - pr['P_gauge_Pa'], abs=1e-9)
    room = ch.parts[0].Pa - np.concatenate(([0.0], np.cumsum(
        ambient_column_densities(ch) * G * ch.segment_vertical_heights)))
    assert np.max(np.abs(pr['P_ambient_Pa'] - room)) < 1e-9


def test_stack_differential_matches_static_gauge_reference():
    """Both gauge views use the room column indoors and the outdoor column outdoors."""
    ch = build()
    ch.T_gas[0, :] = 400.0
    ch.density[0, :] = [rho(ch, 400.0, ch.get_segment_info(s)[2].Pa) for s in range(ch.total_segments)]
    ch.velocity[0, :] = 0.0
    n0 = ch.parts[0].num_segments
    gauge = ch.get_static_pressure_profile(timestep=0)['P_gauge_Pa']
    dp = ch.get_stack_differential_pressure(timestep=0)['dP_Pa']
    assert gauge[3] != pytest.approx(0.0, abs=1e-3)
    assert dp[3] == pytest.approx(gauge[3], abs=1e-6)
    assert dp[n0 + 2] == pytest.approx(gauge[n0 + 2], abs=1e-6)
