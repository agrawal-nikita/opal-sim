# SPDX-License-Identifier: Apache-2.0
import os

# Repository root: the directory containing the `opal` package. Anchored here so
# modules can move between subpackages without recounting ".." levels.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "configs", "defaults.json")
