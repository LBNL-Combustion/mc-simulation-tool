"""Objective functions for staged calibration."""

import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_stage_A_objective(
    ss_mdot_sim: Dict[str, float],
    ss_mdot_exp: Dict[str, float],
    ss_mdot_exp_std: Dict[str, float],
    ss_pressure_sim: Dict[str, Dict[str, float]],
    ss_pressure_exp: Dict[str, Dict[str, float]],
    test_ids: List[str],
    press_sensors_per_test: Dict[str, List[str]],
    w_mdot: float = 1.0,
    w_press: float = 1.0,
    sigma_P: float = 1.0,
    exclude_press_sensors: Optional[List[str]] = None,
) -> Tuple[float, dict]:
    """Stage A objective: closed-door mdot + pressure profile.

    J_A = w_m * sum_tests ((mdot_sim - mdot_exp) / sigma_mdot)^2
        + w_P * sum_tests sum_sensors ((P_sim - P_exp) / sigma_P)^2

    Parameters
    ----------
    ss_mdot_sim : dict test_id -> float (g/s)
    ss_mdot_exp : dict test_id -> float (g/s)
    ss_mdot_exp_std : dict test_id -> float (g/s), 1-sigma
    ss_pressure_sim : dict test_id -> {sensor: float} (Pa)
    ss_pressure_exp : dict test_id -> {sensor: float} (Pa)
    test_ids : list of test IDs to include
    press_sensors_per_test : dict test_id -> list of sensor names for pressure
    w_mdot, w_press : relative weights
    sigma_P : pressure noise scale (Pa)
    exclude_press_sensors : sensors to exclude from pressure objective (e.g. ['TC1'])

    Returns
    -------
    J_A : float (total objective)
    details : dict with per-test breakdowns
    """
    if exclude_press_sensors is None:
        exclude_press_sensors = []

    J_mdot = 0.0
    J_press = 0.0
    details = {'per_test': {}, 'J_mdot': 0.0, 'J_press': 0.0}

    for tid in test_ids:
        m_sim = ss_mdot_sim.get(tid, np.nan)
        m_exp = ss_mdot_exp.get(tid, np.nan)
        m_std = ss_mdot_exp_std.get(tid, np.nan)
        if np.isnan(m_exp) or m_exp <= 0:
            continue
        if np.isnan(m_std) or m_std <= 0:
            m_std = 0.2 * m_exp  # fallback: 20% uncertainty
        r_m = ((m_sim - m_exp) / m_std) ** 2
        J_mdot += r_m

        r_p = 0.0
        n_p = 0
        sensors = press_sensors_per_test.get(tid, [])
        p_sim = ss_pressure_sim.get(tid, {})
        p_exp = ss_pressure_exp.get(tid, {})
        for s in sensors:
            if s in exclude_press_sensors:
                continue
            ps = p_sim.get(s, np.nan)
            pe = p_exp.get(s, np.nan)
            if np.isnan(ps) or np.isnan(pe):
                continue
            r_p += ((ps - pe) / sigma_P) ** 2
            n_p += 1

        J_press += r_p

        details['per_test'][tid] = {
            'mdot_sim': m_sim, 'mdot_exp': m_exp, 'mdot_std': m_std,
            'mdot_resid_sq': r_m,
            'press_resid_sq': r_p, 'n_press_sensors': n_p,
        }

    details['J_mdot'] = J_mdot
    details['J_press'] = J_press
    J_A = w_mdot * J_mdot + w_press * J_press
    details['J_A'] = J_A
    return J_A, details


def compute_stage_B_objective(
    temp_rmse_per_test: Dict[str, Dict[str, float]],
    test_ids: List[str],
    temp_sensors_per_test: Dict[str, List[str]],
    w_temp: float = 1.0,
) -> Tuple[float, dict]:
    """Stage B objective: closed-door temperature RMSE.

    J_B = w_T * sum_tests sum_sensors RMSE(T_sim - T_exp)^2

    Parameters
    ----------
    temp_rmse_per_test : dict test_id -> {sensor: RMSE_in_C}
    test_ids : list of test IDs
    temp_sensors_per_test : dict test_id -> list of sensor names
    w_temp : weight

    Returns
    -------
    J_B : float
    details : dict with per-test breakdowns
    """
    J_B = 0.0
    details = {'per_test': {}}
    for tid in test_ids:
        sensors = temp_sensors_per_test.get(tid, [])
        rmses = temp_rmse_per_test.get(tid, {})
        tid_sum = 0.0
        for s in sensors:
            r = rmses.get(s, np.nan)
            if not np.isnan(r):
                tid_sum += r ** 2
        J_B += tid_sum
        details['per_test'][tid] = {s: rmses.get(s, np.nan) for s in sensors}
    J_B *= w_temp
    details['J_B'] = J_B
    return J_B, details


def compute_stage_C_objective(
    ss_mdot_sim: Dict[str, float],
    ss_mdot_exp: Dict[str, float],
    ss_mdot_exp_std: Dict[str, float],
    ss_press_top_sim: Dict[str, float],
    ss_press_top_exp: Dict[str, float],
    test_ids: List[str],
    w_mdot: float = 1.0,
    w_press: float = 0.5,
    sigma_P: float = 1.0,
) -> Tuple[float, dict]:
    """Stage C objective: cross-test mdot + TOP pressure bias vs wind.

    J_C = w_m * sum_tests((mdot_sim - mdot_exp)/sigma)^2
        + w_P * sum_tests((P_top_sim - P_top_exp)/sigma_P)^2
    """
    J_m = 0.0
    J_p = 0.0
    details = {'per_test': {}}
    for tid in test_ids:
        m_sim = ss_mdot_sim.get(tid, np.nan)
        m_exp = ss_mdot_exp.get(tid, np.nan)
        m_std = ss_mdot_exp_std.get(tid, np.nan)
        if np.isnan(m_exp) or m_exp <= 0:
            continue
        if np.isnan(m_std) or m_std <= 0:
            m_std = 0.2 * m_exp
        J_m += ((m_sim - m_exp) / m_std) ** 2

        pt_s = ss_press_top_sim.get(tid, np.nan)
        pt_e = ss_press_top_exp.get(tid, np.nan)
        r_p = ((pt_s - pt_e) / sigma_P) ** 2 if not (np.isnan(pt_s) or np.isnan(pt_e)) else 0.0
        J_p += r_p

        details['per_test'][tid] = {
            'mdot_sim': m_sim, 'mdot_exp': m_exp,
            'press_top_sim': pt_s, 'press_top_exp': pt_e,
        }

    details['J_mdot'] = J_m
    details['J_press'] = J_p
    J_C = w_mdot * J_m + w_press * J_p
    details['J_C'] = J_C
    return J_C, details
