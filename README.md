# QCS server controller runtime files

This directory keeps the Python GUI controller and its local runtime dependencies copied from `QCS_integrated_controller/Modular_Python_Controller/`. The source project remains unchanged.

Included are the controller entry point, imported helper modules, required JSON/Tcl settings, the internet HydraHarp protocol helper, and optional HydraHarp SDK/udev support. Standalone diagnostic scripts, historical diagnostic reports/logs, test data, the separate experimental clock-control package, IDE files, bytecode caches, and Windows-only legacy DLL bundles were removed from this copy.

This is a source-file migration only; dependencies, instrument SDK availability, serial/device paths, and permissions still need to be configured on the target host.
