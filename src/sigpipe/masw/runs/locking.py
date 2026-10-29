"""Locks on files across processes, alike on every system: Linux and macOS take them with flock,
Windows with LockFileEx. A lock is shared with the other shared holders, or exclusive; each open
file holds its own, so that another, in this process or another, waits for it or is refused.

Windows bars the bytes a lock covers to every other handle, where flock binds only those that
lock: there the lock covers one byte far past any content, so that the file stays readable and
writable by all (a run's lock file names its holder to the writers it refuses), the lock binding
only those that take it."""

import sys
from typing import IO, Any

if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _EXCLUSIVE = 0x2  # LOCKFILE_EXCLUSIVE_LOCK
    _FAIL_IMMEDIATELY = 0x1  # LOCKFILE_FAIL_IMMEDIATELY
    _LOCK_VIOLATION = 33  # ERROR_LOCK_VIOLATION
    # The byte locked, far past any file's content.
    _OFFSET = 1 << 62

    class _Overlapped(ctypes.Structure):
        _fields_ = (
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    _kernel32.UnlockFileEx.restype = wintypes.BOOL

    def _region() -> _Overlapped:
        return _Overlapped(Offset=_OFFSET & 0xFFFFFFFF, OffsetHigh=_OFFSET >> 32)

    def _lock(file: IO[Any], shared: bool, wait: bool) -> None:
        handle = msvcrt.get_osfhandle(file.fileno())
        flags = (0 if shared else _EXCLUSIVE) | (0 if wait else _FAIL_IMMEDIATELY)
        if not _kernel32.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(_region())):
            error = ctypes.get_last_error()
            if error == _LOCK_VIOLATION:
                raise BlockingIOError(error, "The file is locked by another holder.")
            raise ctypes.WinError(error)

    def _unlock(file: IO[Any]) -> None:
        handle = msvcrt.get_osfhandle(file.fileno())
        if not _kernel32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(_region())):
            raise ctypes.WinError(ctypes.get_last_error())

else:
    import fcntl

    def _lock(file: IO[Any], shared: bool, wait: bool) -> None:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(file, mode if wait else mode | fcntl.LOCK_NB)

    def _unlock(file: IO[Any]) -> None:
        fcntl.flock(file, fcntl.LOCK_UN)


def lock_file(file: IO[Any], *, shared: bool = False, wait: bool = True) -> None:
    """`file` locked, `shared` with the other shared holders or exclusively. Held otherwise, it
    is waited for; not to `wait`, BlockingIOError is raised."""
    _lock(file, shared, wait)


def unlock_file(file: IO[Any]) -> None:
    """`file`'s lock released."""
    _unlock(file)
