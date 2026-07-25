"""Version specification for CarvusTrain."""

__version__ = "1.1.0"
__author__ = "Aadil Fazal"
__license__ = "MIT"
__copyright__ = "Copyright 2026 Aadil Fazal"


def get_version_info() -> str:
    """Return formatted version and system build information.

    Returns:
        Formatted string containing version number, build target, and license.
    """
    return f"CarvusTrain v{__version__} ({__license__} License) — AI Development Ecosystem"
