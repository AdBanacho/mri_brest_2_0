"""Session-wide test settings.

Bytecode writing is disabled for the whole session: some parity helpers put the read-only reference
repository on ``sys.path`` and any later import from it must never create ``__pycache__`` there
(enforced by the guard in ``tests/parity/conftest.py``).
"""

import sys

sys.dont_write_bytecode = True
