"""
Room plenum pressure dynamics for chimney simulation.

Models the room as a pneumatic capacitance: when the chimney draws air
from the room and inflow from outside is restricted (e.g., tight house),
room pressure drops below ambient, reducing chimney draft.

    dP_room/dt = (mdot_in_leaks - mdot_chimney) * R_air * T_room / V_room

where mdot_in_leaks is the air leaking into the room from outdoors
through the building envelope, computed via an orifice model:

    mdot_in = Cd * A_room_leak * sqrt(2 * rho * |P_amb - P_room|)

Default: OFF.  Enable via Chimney(room_plenum={...}).

Numerical method: adaptive subcycling with forward Euler.
The orifice leak-in model creates stiff dynamics (large gain at small
depressurization), so the global chimney dt (e.g. 0.5s) is often too
large for stability.  We estimate the local stability limit and
subcycle automatically.
"""
import numpy as np
from math import ceil, sqrt
from dataclasses import dataclass


R_AIR = 287.0  # J/(kg K)


@dataclass
class RoomPlenumConfig:
    """Configuration for room plenum pressure dynamics.

    Attributes:
        enable: Activate room plenum feature (default False).
        V_room_m3: Room volume in m^3 (default 50 m^3 ~ 1800 ft^3).
        A_room_leak_m2: Equivalent leakage area of building envelope in m^2
            (default 0.02 m^2 = 200 cm^2, typical tight house).
        Cd_room: Discharge coefficient for room leakage (default 0.65).
        T_room_K: Room air temperature in K (default 294.15 = 21 C).
        P_room_init_Pa: Initial room gauge pressure relative to ambient
            (default 0.0 = neutral).
    """
    enable: bool = False
    V_room_m3: float = 50.0
    A_room_leak_m2: float = 0.02
    Cd_room: float = 0.65
    T_room_K: float = 294.15
    P_room_init_Pa: float = 0.0


class RoomPlenum:
    """Tracks room pressure relative to ambient.

    Usage:
        plenum = RoomPlenum(config)
        # each timestep:
        dp_room = plenum.update(mdot_chimney, P_ambient, dt)
        # dp_room is the pressure reduction applied to chimney draft
    """

    def __init__(self, config: RoomPlenumConfig):
        self.cfg = config
        # P_room is gauge pressure: P_room_abs = P_ambient + P_room_gauge
        # Negative P_room_gauge means room is depressurized
        self.P_room_gauge = config.P_room_init_Pa
        self._history = []  # optional: track P_room over time

    def reset(self):
        self.P_room_gauge = self.cfg.P_room_init_Pa
        self._history.clear()

    def _leak_in_mdot(self, P_gauge, P_ambient):
        """Compute leak-in mass flow for a given gauge pressure.

        Returns positive mdot when room is depressurized (P_gauge < 0),
        meaning air flows INTO the room from outdoors.
        """
        dP_envelope = -P_gauge  # positive when room is below ambient
        rho_outdoor = P_ambient / (R_AIR * self.cfg.T_room_K)
        if abs(dP_envelope) < 1e-12:
            return 0.0
        sign = 1.0 if dP_envelope > 0 else -1.0
        return (self.cfg.Cd_room * self.cfg.A_room_leak_m2
                * sign * sqrt(2.0 * rho_outdoor * abs(dP_envelope)))

    def _stable_dt(self, P_ambient):
        """Estimate maximum stable forward-Euler timestep.

        The stiffest regime is near P_gauge ~ 0 (small depressurization,
        large d(mdot_in)/dP).  We evaluate the gain at |P_gauge| = P_ref
        (a small reference pressure) and require dt * gain < 1.
        """
        P_ref = max(abs(self.P_room_gauge), 0.5)  # Pa, floor at 0.5 Pa
        rho = P_ambient / (R_AIR * self.cfg.T_room_K)
        # d(mdot_in)/d(P_gauge) at P_ref (magnitude):
        #   |d/dP [Cd*A*sqrt(2*rho*|P|)]| = Cd*A*rho / sqrt(2*rho*P_ref)
        dmdot_dP = (self.cfg.Cd_room * self.cfg.A_room_leak_m2
                    * rho / sqrt(2.0 * rho * P_ref))
        G = R_AIR * self.cfg.T_room_K / self.cfg.V_room_m3  # Pa per (kg/s) per s
        gain = G * dmdot_dP  # 1/s
        if gain < 1e-6:
            return 10.0  # effectively no limit
        return 0.8 / gain  # safety factor 0.8

    def update(self, mdot_chimney: float, P_ambient: float, dt: float) -> float:
        """Advance room pressure by one timestep with adaptive subcycling.

        Args:
            mdot_chimney: Mass flow rate drawn by chimney [kg/s].
                Positive = chimney draws air from room.
            P_ambient: Outdoor ambient pressure [Pa].
            dt: Timestep [s].

        Returns:
            Room depressurization (Pa, positive means room is below ambient).
            This value should be subtracted from chimney driving pressure.
        """
        if not self.cfg.enable:
            return 0.0

        # Determine subcycling
        dt_stable = self._stable_dt(P_ambient)
        n_sub = max(1, int(ceil(dt / dt_stable)))
        dt_sub = dt / n_sub

        for _ in range(n_sub):
            mdot_in = self._leak_in_mdot(self.P_room_gauge, P_ambient)

            # Net mass flow into room = leakage in - chimney draw
            mdot_net = mdot_in - mdot_chimney

            # dP_room/dt = mdot_net * R_air * T_room / V_room
            dP_dt = mdot_net * R_AIR * self.cfg.T_room_K / self.cfg.V_room_m3

            # Forward Euler substep
            self.P_room_gauge += dP_dt * dt_sub

            # Clamp to physically reasonable range
            self.P_room_gauge = max(self.P_room_gauge, -50.0)
            self.P_room_gauge = min(self.P_room_gauge, 50.0)

        self._history.append(self.P_room_gauge)

        # Return depressurization as positive value (subtracted from draft)
        return -self.P_room_gauge

    @property
    def history(self):
        """Room gauge pressure history as numpy array."""
        return np.array(self._history)
