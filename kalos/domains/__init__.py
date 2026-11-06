"""Domain layer: declared column roles, domain profiles, and mixed design
spaces that map a tabular problem onto the domain-neutral engine.

Torch-free: importing this package never loads torch/botorch/gpytorch.
"""
from __future__ import annotations

from .bioprocess import BIOPROCESS_PROFILE
from .generic import GENERIC_PROFILE
from .profile import (
    DIRECTIONS,
    ColumnRoles,
    DesignSpace,
    Dimension,
    DimensionKind,
    DomainProfile,
    InvalidDirection,
    build_design_space,
    parse_direction,
)

__all__ = [
    "BIOPROCESS_PROFILE",
    "DIRECTIONS",
    "GENERIC_PROFILE",
    "ColumnRoles",
    "DesignSpace",
    "Dimension",
    "DimensionKind",
    "DomainProfile",
    "InvalidDirection",
    "build_design_space",
    "parse_direction",
]
