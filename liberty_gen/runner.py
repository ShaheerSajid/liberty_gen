"""
ngspice subprocess runner and .meas result parser.

ngspice is invoked with ``ngspice -b <deck>`` (batch mode).  In batch mode
ngspice prints ``.meas`` results to stdout in the form::

    tpd_rise            =  1.234560e-10  targ=...  trig=...
    q_sample            =  1.750000e+00
    tpd_fall            failed

The parser:
* extracts all ``name = value`` pairs
* detects failed measurements (value = 0 with "failed" annotation or explicit
  "failed" text) and stores them as ``None``
* converts timing measures from seconds → nanoseconds
  (any measure whose name starts with ``t`` and is not ``t_win`` / ``t_cycle``
  gets the ×1e9 conversion; voltage-like measures ``q_sample`` are left as-is)
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import tempfile
from typing import Optional

# Match:  name = ±number[eE±int]  (optionally followed by more text on the line)
_MEAS_VALUE_RE = re.compile(
    r"^(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)
# Match lines that ngspice marks as failed
_MEAS_FAILED_RE = re.compile(r"^(\w+)\s+failed", re.MULTILINE | re.IGNORECASE)

# Names that return volts (not time) — do NOT convert to ns
_VOLTAGE_MEASURES = {"q_sample"}

# Names that are already in seconds but should be converted to ns
_PARAM_MEASURES = {"t_win", "t_cycle"}   # these are .param strings, not real meas


def run_ngspice(deck: str, timeout: int = 180) -> dict[str, Optional[float]]:
    """Write ``deck`` to a temp file, run ``ngspice -b``, parse stdout.

    Returns a dict mapping lowercase measure name → float value or ``None``
    (if the measurement failed or was not found).

    * Timing measures (name starts with ``t``, excluding voltage-like ones):
      converted from **seconds → nanoseconds** (×1e9).
    * Voltage measures (``q_sample``): returned in volts, unchanged.
    * Integral measures (``i_avg``, ``i_op``): returned in SI units (A·s), unchanged.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sp", delete=False, prefix="fabram_char_"
    ) as f:
        f.write(deck)
        deck_path = f.name

    try:
        proc = subprocess.run(
            ["ngspice", "-b", deck_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {}
    except FileNotFoundError:
        raise RuntimeError(
            "ngspice not found. Install ngspice and ensure it is on PATH."
        ) from None
    finally:
        pathlib.Path(deck_path).unlink(missing_ok=True)

    return _parse_output(output)


def _parse_output(text: str) -> dict[str, Optional[float]]:
    """Parse ngspice stdout/stderr for .meas results."""
    result: dict[str, Optional[float]] = {}

    # First collect all value matches
    for m in _MEAS_VALUE_RE.finditer(text):
        name = m.group(1).lower()
        val  = float(m.group(2))
        result[name] = val

    # Mark explicitly failed measurements as None
    for m in _MEAS_FAILED_RE.finditer(text):
        result[m.group(1).lower()] = None

    # Convert time measures (seconds → nanoseconds)
    # Heuristic: starts with 't' and is not a voltage or integral measure
    converted: dict[str, Optional[float]] = {}
    for name, val in result.items():
        if val is None:
            converted[name] = None
        elif name in _VOLTAGE_MEASURES:
            converted[name] = val          # volts, as-is
        elif name.startswith("i_"):
            converted[name] = val          # current integral (A·s), as-is
        elif name.startswith("t"):
            converted[name] = val * 1e9    # seconds → nanoseconds
        else:
            converted[name] = val

    return converted


def is_correct(q_sample: Optional[float], cfg) -> bool:
    """Return True if Q0 sampled high (q_val=1 convention throughout)."""
    if q_sample is None:
        return False
    return q_sample > cfg.vdd * cfg.th_mid
