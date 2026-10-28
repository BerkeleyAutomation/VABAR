# SPDX-License-Identifier: Apache-2.0
"""Every module under ``assets/``, imported so its records, entities and atoms register.

The RoboSimStudio this suite was written against walked a suite's ``assets/``
package itself before lowering its documents. Upstream's generic loader runs a
suite's ``staging.py``, ``scenes.py`` and ``records.py`` instead, and leaves
``assets/`` to be imported from one of them -- so this file is that walk, in the
same order (``pkgutil.walk_packages``, subpackages included).
"""

from __future__ import annotations

import importlib
import pkgutil

from . import assets as _assets

for _info in pkgutil.walk_packages(_assets.__path__, prefix=f"{_assets.__name__}."):
    importlib.import_module(_info.name)
