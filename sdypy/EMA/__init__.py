from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sdypy-EMA")
except PackageNotFoundError:  # source checkout without installed metadata
    __version__ = "0+unknown"

from .EMA import Model
from .tools import MAC, MSF, MCF, complex_freq_to_freq_and_damp

from . import stabilization
from . import normal_modes
from . import pole_picking

__all__ = [
    "Model",
    "MAC",
    "MSF",
    "MCF",
    "complex_freq_to_freq_and_damp",
    "stabilization",
    "normal_modes",
    "pole_picking",
]
