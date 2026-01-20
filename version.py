"""
Merlina Version Information
Semantic Versioning (SemVer) - MAJOR.MINOR.PATCH

MAJOR version: Incompatible API changes
MINOR version: Backwards-compatible functionality additions
PATCH version: Backwards-compatible bug fixes
"""

__version__ = "1.3.0"
__version_info__ = tuple(int(i) for i in __version__.split("."))

# Version metadata
VERSION_MAJOR = __version_info__[0]
VERSION_MINOR = __version_info__[1]
VERSION_PATCH = __version_info__[2]

# Release information
RELEASE_NAME = "Message Magic"  # Codename for this release
RELEASE_DATE = "2026-01-20"  # Release date of current version

def get_version() -> str:
    """Returns the current version string."""
    return __version__

def get_version_info() -> dict:
    """Returns detailed version information."""
    return {
        "version": __version__,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "release_name": RELEASE_NAME,
        "release_date": RELEASE_DATE,
    }

def get_version_string() -> str:
    """Returns a formatted version string with release name."""
    return f"Merlina v{__version__} - {RELEASE_NAME}"
