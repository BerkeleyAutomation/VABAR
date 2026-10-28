# SPDX-License-Identifier: Apache-2.0
"""The suite's asset records, registered where RoboSimStudio's loader looks for them.

The generic directory loader executes a suite's ``staging.py``, ``scenes.py``
and ``records.py`` before lowering its documents. This family keeps its records
in the ``assets/`` package beside the meshes they describe, so this module is
only the door the loader walks through.
"""

from .assets import *  # noqa: F403
