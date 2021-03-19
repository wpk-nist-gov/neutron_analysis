from .photonatten import PhotonAtten
from .shielding import NeutronShieldingpPT, NeutronShieldingUser, read_mass_excel
from .utils import loc_levels, reorder_index_levels

try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("cmomy").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"

__all__ = [
    "NeutronShieldingpPT",
    "NeutronShieldingUser",
    "read_mass_excel",
    "PhotonAtten",
    "reorder_index_levels",
    "loc_levels",
    "__version__",
]
