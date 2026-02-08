"""
PyFluesim - Python package for chimney heat transfer and draft calculations

Converted from FLUESIM5.BAS - Dynamic analysis program for furnace and flue transient performance
Original version 5.0 developed by Scanada Consultants Limited (December 1989)

This package provides tools for simulating heat transfer, mass flow, and pressure dynamics
in chimney and flue systems.
"""

from .chimney import Chimney
from .mass_flow import MassFlow
from .physics import PhysicalProperties
from .diagnostics_recorder import DiagnosticsRecorder
from .coupled_solver import MomentumConfig
from .room_plenum import RoomPlenumConfig

__version__ = "1.0.0"
__author__ = "Converted from FLUESIM5.BAS"

__all__ = [
    "Chimney",
    "MassFlow",
    "PhysicalProperties",
    "DiagnosticsRecorder",
    "MomentumConfig",
    "RoomPlenumConfig",
]