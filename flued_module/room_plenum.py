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
"""
import numpy as np
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

    def update(self, mdot_chimney: float, P_ambient: float, dt: float) -> float:
        """Advance room pressure by one timestep.

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

        # Room absolute pressure
        P_room_abs = P_ambient + self.P_room_gauge

        # Air leaking INTO room from outdoors through building envelope
        # Orifice model: mdot_in = Cd * A * sqrt(2 * rho * |dP|)
        dP_envelope = P_ambient - P_room_abs  # positive when room is depressurized
        rho_outdoor = P_ambient / (R_AIR * self.cfg.T_room_K)
        mdot_in = (self.cfg.Cd_room * self.cfg.A_room_leak_m2
                   * np.sign(dP_envelope)
                   * np.sqrt(2.0 * rho_outdoor * abs(dP_envelope)))

        # Net mass flow into room = leakage in - chimney draw
        mdot_net = mdot_in - mdot_chimney

        # dP_room/dt = mdot_net * R_air * T_room / V_room
        dP_dt = mdot_net * R_AIR * self.cfg.T_room_K / self.cfg.V_room_m3

        # Forward Euler
        self.P_room_gauge += dP_dt * dt

        # Clamp: room can't go more than ~100 Pa below ambient in practice
        self.P_room_gauge = max(self.P_room_gauge, -100.0)
        # Room can't be significantly above ambient either (doors/windows open)
        self.P_room_gauge = min(self.P_room_gauge, 50.0)

        self._history.append(self.P_room_gauge)

        # Return depressurization as positive value (subtracted from draft)
        return -self.P_room_gauge

    @property
    def history(self):
        """Room gauge pressure history as numpy array."""
        return np.array(self._history)
