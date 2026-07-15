"""
physics.advanced — v2.1 professional physics extensions.

Adds the capabilities identified as gaps in the integrity assessment:
  optics_tmm     transfer-matrix optics (replaces Beer-Lambert generation)
  temperature    Varshni Eg(T), ni(T), mobility(T), Nc/Nv(T)
  defects        multi-level + Gaussian + exponential-tail SRH recombination
  ion_migration  mobile-ion model -> scan-rate-dependent J-V hysteresis
  impedance      Mott-Schottky C-V and impedance spectroscopy
  dd1d           compact validated 1-D drift-diffusion solver (Gummel + SG)

Every module ships a self-test on a known physical limit (run as __main__).
"""
from . import optics_tmm, temperature, defects, ion_migration, impedance, dd1d
__all__ = ["optics_tmm", "temperature", "defects", "ion_migration", "impedance", "dd1d"]
