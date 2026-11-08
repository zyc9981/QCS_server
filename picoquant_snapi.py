"""Load the installed Linux snAPI instead of the bundled Windows-only copy."""

import importlib.util
import sys
import sysconfig
from pathlib import Path


if sys.platform.startswith("linux"):
    # The script directory precedes site-packages on sys.path, so its old
    # snAPI directory otherwise hides the Linux-capable installed package.
    package_dir = Path(sysconfig.get_path("purelib")) / "snAPI"
    package_init = package_dir / "__init__.py"
    if not package_init.is_file():
        raise ImportError("Install snapi into this Python environment with: uv pip install snapi")
    spec = importlib.util.spec_from_file_location(
        "snAPI", package_init, submodule_search_locations=[str(package_dir)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["snAPI"] = package
    spec.loader.exec_module(package)
    from snAPI.Main import LogLevel, MeasMode, RefSource, snAPI

    def create_hydraharp_api():
        return snAPI()

else:
    from snAPI.Main import LibType, LogLevel, MeasMode, RefSource, snAPI

    def create_hydraharp_api():
        return snAPI(libType=LibType.HH)
