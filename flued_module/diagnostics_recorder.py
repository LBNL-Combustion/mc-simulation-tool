"""
Lightweight diagnostics recorder for momentum and energy budgets.

Records per-timestep quantities for post-simulation analysis.
Default: OFF. Enable via DiagnosticsRecorder(enable=True) or
by passing enable_diagnostics=True to Chimney constructor.

No behavior changes to the solver -- purely observational.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TimestepBudget:
    """Per-timestep momentum and energy budget snapshot."""
    time_s: float = 0.0
    step: int = 0

    # --- momentum budget (Pa) ---
    mdot_kg_s: float = 0.0
    velocity_mean_m_s: float = 0.0
    rho_mean_kg_m3: float = 0.0

    dp_buoyancy_Pa: float = 0.0
    dp_wind_cap_Pa: float = 0.0
    dp_room_Pa: float = 0.0
    dp_available_Pa: float = 0.0       # buoyancy + wind + room

    dp_friction_Pa: float = 0.0
    dp_fittings_Pa: float = 0.0
    dp_inlet_Pa: float = 0.0
    dp_stove_Pa: float = 0.0
    dp_bypass_Pa: float = 0.0
    dp_exit_Pa: float = 0.0
    dp_accel_Pa: float = 0.0
    dp_leakage_Pa: float = 0.0
    dp_losses_total_Pa: float = 0.0

    dp_residual_Pa: float = 0.0        # dp_available - dp_losses - dp_inertia

    # --- energy budget (W) ---
    Q_advection_W: float = 0.0         # mdot * cp * (T_in - T_out) summed
    Q_wall_total_W: float = 0.0        # total heat to walls
    Q_latent_W: float = 0.0            # condensation latent heat
    Q_energy_residual_W: float = 0.0   # advection - wall - latent (should be ~0)

    # --- key temperatures (K) ---
    T_inlet_K: float = 0.0
    T_outlet_K: float = 0.0
    T_gas_mean_K: float = 0.0
    T_wall_mean_K: float = 0.0


class DiagnosticsRecorder:
    """Records per-timestep momentum and energy budgets.

    Usage:
        recorder = DiagnosticsRecorder(enable=True)
        # ... inside solver, call recorder.record_momentum_budget(...)
        # After simulation:
        df = recorder.to_dataframe()
    """

    def __init__(self, enable: bool = False):
        self.enable = enable
        self._records: List[TimestepBudget] = []

    def reset(self):
        """Clear all recorded data."""
        self._records.clear()

    # ------------------------------------------------------------------
    # Recording methods (called from coupled solver hooks)
    # ------------------------------------------------------------------

    def record_momentum_budget(
        self,
        step: int,
        time_s: float,
        mdot: float,
        velocity_mean: float,
        rho_mean: float,
        dp_buoyancy: float,
        dp_wind_cap: float,
        dp_room: float,
        dp_friction: float,
        dp_fittings: float,
        dp_inlet: float,
        dp_stove: float,
        dp_bypass: float,
        dp_exit: float,
        dp_accel: float,
        dp_leakage: float,
    ):
        """Record momentum-side quantities for one timestep."""
        if not self.enable:
            return

        rec = self._get_or_create(step, time_s)
        rec.mdot_kg_s = mdot
        rec.velocity_mean_m_s = velocity_mean
        rec.rho_mean_kg_m3 = rho_mean

        rec.dp_buoyancy_Pa = dp_buoyancy
        rec.dp_wind_cap_Pa = dp_wind_cap
        rec.dp_room_Pa = dp_room
        rec.dp_available_Pa = dp_buoyancy - dp_wind_cap - dp_room

        rec.dp_friction_Pa = dp_friction
        rec.dp_fittings_Pa = dp_fittings
        rec.dp_inlet_Pa = dp_inlet
        rec.dp_stove_Pa = dp_stove
        rec.dp_bypass_Pa = dp_bypass
        rec.dp_exit_Pa = dp_exit
        rec.dp_accel_Pa = dp_accel
        rec.dp_leakage_Pa = dp_leakage
        rec.dp_losses_total_Pa = (
            dp_friction + dp_fittings + dp_inlet + dp_stove
            + dp_bypass + dp_exit + dp_accel + dp_leakage
        )
        rec.dp_residual_Pa = rec.dp_available_Pa - rec.dp_losses_total_Pa

    def record_energy_budget(
        self,
        step: int,
        time_s: float,
        Q_advection: float,
        Q_wall_total: float,
        Q_latent: float,
        T_inlet: float,
        T_outlet: float,
        T_gas_mean: float,
        T_wall_mean: float,
    ):
        """Record energy-side quantities for one timestep."""
        if not self.enable:
            return

        rec = self._get_or_create(step, time_s)
        rec.Q_advection_W = Q_advection
        rec.Q_wall_total_W = Q_wall_total
        rec.Q_latent_W = Q_latent
        rec.Q_energy_residual_W = Q_advection - Q_wall_total - Q_latent

        rec.T_inlet_K = T_inlet
        rec.T_outlet_K = T_outlet
        rec.T_gas_mean_K = T_gas_mean
        rec.T_wall_mean_K = T_wall_mean

    # ------------------------------------------------------------------
    # Access / export
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[TimestepBudget]:
        return self._records

    def to_dataframe(self):
        """Export records as a pandas DataFrame."""
        import pandas as pd
        if not self._records:
            return pd.DataFrame()
        rows = []
        for r in self._records:
            rows.append(vars(r))
        return pd.DataFrame(rows)

    def to_dict_arrays(self) -> Dict[str, np.ndarray]:
        """Export as dict of numpy arrays (avoids pandas dependency)."""
        if not self._records:
            return {}
        keys = list(vars(self._records[0]).keys())
        out = {k: np.array([getattr(r, k) for r in self._records]) for k in keys}
        return out

    # ------------------------------------------------------------------
    # Self-check functions
    # ------------------------------------------------------------------

    def check_momentum_residual(self, tol_Pa: float = 5.0) -> bool:
        """Assert momentum residual is within tolerance for all steps.

        Args:
            tol_Pa: Acceptable residual in Pascals (default 5 Pa).

        Returns:
            True if all residuals within tolerance.

        Raises:
            AssertionError with details if any step exceeds tolerance.
        """
        if not self._records:
            return True
        residuals = np.array([r.dp_residual_Pa for r in self._records])
        max_res = np.max(np.abs(residuals))
        worst_step = int(np.argmax(np.abs(residuals)))
        assert max_res <= tol_Pa, (
            f"Momentum residual exceeded tolerance: "
            f"max |dp_residual| = {max_res:.2f} Pa at step {worst_step} "
            f"(tolerance = {tol_Pa} Pa)"
        )
        return True

    def check_energy_residual(self, tol_W: float = 50.0) -> bool:
        """Assert energy residual is within tolerance for all steps.

        Args:
            tol_W: Acceptable residual in Watts (default 50 W).

        Returns:
            True if all residuals within tolerance.

        Raises:
            AssertionError with details if any step exceeds tolerance.
        """
        if not self._records:
            return True
        residuals = np.array([r.Q_energy_residual_W for r in self._records])
        max_res = np.max(np.abs(residuals))
        worst_step = int(np.argmax(np.abs(residuals)))
        assert max_res <= tol_W, (
            f"Energy residual exceeded tolerance: "
            f"max |Q_residual| = {max_res:.2f} W at step {worst_step} "
            f"(tolerance = {tol_W} W)"
        )
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, step: int, time_s: float) -> TimestepBudget:
        """Get existing record for step, or create a new one."""
        # Fast path: records are appended in order
        if self._records and self._records[-1].step == step:
            return self._records[-1]
        rec = TimestepBudget(time_s=time_s, step=step)
        self._records.append(rec)
        return rec
