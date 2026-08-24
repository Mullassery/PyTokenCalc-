"""Single source of truth for the package version.

Keep this in sync with the version in pyproject.toml (there is no automated
sync between the two -- setuptools reads pyproject.toml for packaging
metadata, this module is what the library reads at runtime/import time).
"""

__version__ = "1.2.0"
