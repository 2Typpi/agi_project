import os
import glob
import importlib

# Get the directory of the current module
module_dir = os.path.dirname(__file__)

# Find all .py files in the directory (excluding __init__.py)
for module_file in glob.glob(os.path.join(module_dir, "*.py")):
    module_name = os.path.basename(module_file)[:-3]
    if module_name != "__init__":
        # Dynamically import the module
        importlib.import_module(f".{module_name}", package=__package__)
