# SPDX-License-Identifier: Apache-2.0
"""The tool-hanging family's declarations, beside the geometry they own.

A package rather than a module because the family owns files -- four tool
classes' meshes, their CoACD hull sets with machine-written manifests, and the
holders they hang on -- and the rule of this tree is that mesh, declaration and
``PROVENANCE.md`` travel together. See :mod:`tasks/tool_hanging/assets/records.py`
for the declarations themselves.
"""

from .records import *  # noqa: F403
