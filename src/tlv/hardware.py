"""Record the machine a run was made on (no usernames or paths)."""
from __future__ import annotations

import ctypes
import os
import platform


def _cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined,unused-ignore]
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()  # type: ignore[attr-defined,unused-ignore]
    except Exception:
        return platform.processor() or "unknown CPU"


def _ram_gb() -> float:
    try:
        class MEM(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                        ("ext", ctypes.c_ulonglong)]

        m = MEM()
        m.l = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined,unused-ignore]
        return round(m.total / 2**30, 1)
    except Exception:
        return float("nan")


def describe(tf_version: str = "") -> dict:
    return {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "tensorflow": tf_version,
        "accelerator": "none (CPU only, no CUDA)",
    }


def short(info: dict) -> str:
    return f"{info['cpu']}, {info['logical_cores']} logical cores, {info['ram_gb']} GB RAM, CPU only"
