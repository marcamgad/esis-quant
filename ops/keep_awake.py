"""
Keep Windows from sleeping while a long job runs.

WHY. On 2026-10-03 the archive fetch stopped silently after an hour: the laptop
went to sleep mid-run (no error in the log, the process simply did not come
back). Scheduled tasks can WAKE the machine, but nothing stopped it sleeping
again halfway through. SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)
holds the system awake until the context exits; the display may still turn off.
Does nothing on other platforms.
"""

import contextlib
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake():
    k32 = None
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception:  # noqa: BLE001
            k32 = None
    try:
        yield
    finally:
        if k32 is not None:
            k32.SetThreadExecutionState(ES_CONTINUOUS)
