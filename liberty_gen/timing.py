"""
Timing measurement engine.

Provides:
* True bisection for setup / hold time (O(log N) convergence).
* Grid sweep for CLK-to-Q with parallel execution.
* Leakage and dynamic power measurement.

Bisection invariant
-------------------
    lo  → timing MET   (output correct)
    hi  → timing VIOLATED (output wrong or unmeasurable)

The bracket [lo, hi] is narrowed by half each iteration.  After
``max_iterations`` or when ``hi - lo < timing_resolution``, ``hi`` is
returned as the conservative (larger) bound.

Parallelism
-----------
CLK-to-Q and the outer (pin_slew × clk_slew) grid for setup/hold run in a
``ThreadPoolExecutor``.  Individual bisections run serially within each worker
(correct, since each iteration depends on the previous result).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

from liberty_gen.config import CharConfig
from liberty_gen.runner import run_ngspice, is_correct
from liberty_gen.testbench import (
    build_clkq_testbench,
    build_constraint_testbench,
    build_leakage_testbench,
    build_power_testbench,
    _ramp,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLK-to-Q
# ─────────────────────────────────────────────────────────────────────────────

def _run_clkq_point(
    netlist_path: str, cfg: CharConfig, macro: str,
    addr_bits: int, bits: int,
    clk_slew: float, load_pf: float, q_val: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Run one (clk_slew, load_pf, q_val) point.

    Returns (tpd_rise, tpd_fall, ttr_rise, ttr_fall) — all in ns.
    Entries irrelevant to q_val will be None (ngspice can't measure them).
    """
    deck = build_clkq_testbench(
        netlist_path, cfg, macro, addr_bits, bits, clk_slew, load_pf, q_val
    )
    meas = run_ngspice(deck, cfg.sim_timeout)
    return (
        meas.get("tpd_rise"),
        meas.get("tpd_fall"),
        meas.get("ttr_rise"),
        meas.get("ttr_fall"),
    )


def measure_clkq(
    netlist_path: str,
    cfg: CharConfig,
    macro: str,
    addr_bits: int,
    bits: int,
) -> dict:
    """Measure CLK-to-Q propagation and transition times over the full grid.

    Returns a dict with keys:
        ``cell_rise``        — 2-D list [slew_idx][load_idx] (ns)
        ``cell_fall``        — 2-D list [slew_idx][load_idx] (ns)
        ``rise_transition``  — 2-D list [slew_idx][load_idx] (ns)
        ``fall_transition``  — 2-D list [slew_idx][load_idx] (ns)

    Two runs per grid point: q_val=1 gives rise arc; q_val=0 gives fall arc.
    """
    slews = cfg.input_slews
    loads = cfg.output_loads
    ns = len(slews)
    nl = len(loads)

    # Pre-allocate tables
    cell_rise  = [[None] * nl for _ in range(ns)]
    cell_fall  = [[None] * nl for _ in range(ns)]
    rise_tr    = [[None] * nl for _ in range(ns)]
    fall_tr    = [[None] * nl for _ in range(ns)]

    tasks: list[tuple] = []
    for (si, sl), (li, ld), q in product(enumerate(slews), enumerate(loads), [0, 1]):
        tasks.append((si, li, sl, ld, q))

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = {
            ex.submit(_run_clkq_point, netlist_path, cfg, macro, addr_bits, bits,
                      sl, ld, q): (si, li, q)
            for si, li, sl, ld, q in tasks
        }
        for fut in as_completed(futs):
            si, li, q = futs[fut]
            try:
                r_rise, r_fall, tr_rise, tr_fall = fut.result()
            except Exception:
                continue
            if q == 1:
                cell_rise[si][li] = r_rise
                rise_tr[si][li]   = tr_rise
            else:
                cell_fall[si][li] = r_fall
                fall_tr[si][li]   = tr_fall

    return {
        "cell_rise":       cell_rise,
        "cell_fall":       cell_fall,
        "rise_transition": rise_tr,
        "fall_transition": fall_tr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Setup / Hold bisection
# ─────────────────────────────────────────────────────────────────────────────

def _clk_edge(n: int, cfg: CharConfig, clk_slew: float) -> float:
    """Absolute time (ns) of the n-th CLK rising edge (1-indexed).

    For PULSE(0 VDD 0 TR TR PW PER), the 50 % crossing of the n-th rising
    edge is at  (n-1)*T + ramp*th_mid  where ramp = TR = clk_slew/(th_hi-th_lo).
    """
    ramp = _ramp(clk_slew, cfg.th_lo, cfg.th_hi)
    return (n - 1) * cfg.clk_period + ramp * cfg.th_mid


def _bisect(
    *,
    netlist_path: str,
    cfg: CharConfig,
    macro: str,
    addr_bits: int,
    bits: int,
    pin: str,
    clk_slew: float,
    pin_slew: float,
    mode: str,           # "setup" | "hold"
    ref_edge: float,     # absolute CLK edge time (ns) around which we bisect
) -> float:
    """True bisection returning the conservative timing bound (ns).

    Setup: returns minimum setup time (pin must be stable this many ns *before*
           the reference CLK edge).
    Hold:  returns minimum hold time (pin must remain stable this many ns *after*
           the reference CLK edge).
    """
    lo, hi = 0.0, cfg.clk_period / 2

    for _ in range(cfg.max_iterations):
        if (hi - lo) < cfg.timing_resolution:
            break
        mid = (lo + hi) / 2.0
        if mode == "setup":
            t_arrive = ref_edge - mid   # earlier = more setup margin
        else:
            t_arrive = ref_edge + mid   # later = more hold margin

        deck = build_constraint_testbench(
            netlist_path, cfg, macro, addr_bits, bits,
            pin=pin, clk_slew=clk_slew, pin_slew=pin_slew,
            t_arrive=t_arrive, mode=mode,
        )
        meas = run_ngspice(deck, cfg.sim_timeout)
        passed = is_correct(meas.get("q_sample"), cfg)

        if passed:
            hi = mid   # timing met → try smaller margin
        else:
            lo = mid   # timing violated → need more margin

    return hi  # conservative (larger) bound


def _measure_sh_point(
    netlist_path: str, cfg: CharConfig, macro: str,
    addr_bits: int, bits: int,
    pin: str, pin_slew: float, clk_slew: float,
) -> tuple[float, float]:
    """Measure setup and hold at one (pin_slew, clk_slew) grid point."""
    # Reference clock edge depends on which pin is constrained
    # addr0 is constrained relative to CLK3 (READ edge, edge index 3)
    # din0, CS, WRITE are constrained relative to CLK2 (WRITE edge, edge index 2)
    if pin == "addr0":
        ref_edge = _clk_edge(3, cfg, clk_slew)
    else:
        ref_edge = _clk_edge(2, cfg, clk_slew)

    setup = _bisect(
        netlist_path=netlist_path, cfg=cfg, macro=macro,
        addr_bits=addr_bits, bits=bits,
        pin=pin, clk_slew=clk_slew, pin_slew=pin_slew,
        mode="setup", ref_edge=ref_edge,
    )
    hold = _bisect(
        netlist_path=netlist_path, cfg=cfg, macro=macro,
        addr_bits=addr_bits, bits=bits,
        pin=pin, clk_slew=clk_slew, pin_slew=pin_slew,
        mode="hold", ref_edge=ref_edge,
    )
    return setup, hold


def measure_setup_hold(
    netlist_path: str,
    cfg: CharConfig,
    macro: str,
    addr_bits: int,
    bits: int,
    pin: str,
) -> dict:
    """Measure setup and hold times over the full (pin_slew × clk_slew) grid.

    Returns:
        ``setup`` — 2-D list [pin_slew_idx][clk_slew_idx] (ns)
        ``hold``  — 2-D list [pin_slew_idx][clk_slew_idx] (ns)

    Liberty convention: index_1 = constrained_pin_transition,
                        index_2 = related_pin_transition (CLK).
    """
    slews = cfg.input_slews
    n = len(slews)
    setup_tbl = [[None] * n for _ in range(n)]
    hold_tbl  = [[None] * n for _ in range(n)]

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = {
            ex.submit(
                _measure_sh_point,
                netlist_path, cfg, macro, addr_bits, bits,
                pin, pin_slew, clk_slew,
            ): (pi, ci)
            for (pi, pin_slew), (ci, clk_slew) in product(
                enumerate(slews), enumerate(slews)
            )
        }
        for fut in as_completed(futs):
            pi, ci = futs[fut]
            try:
                s, h = fut.result()
            except Exception:
                s, h = None, None
            setup_tbl[pi][ci] = s
            hold_tbl[pi][ci]  = h

    return {"setup": setup_tbl, "hold": hold_tbl}


# ─────────────────────────────────────────────────────────────────────────────
# Minimum pulse width
# ─────────────────────────────────────────────────────────────────────────────

def measure_min_pulse_width(
    netlist_path: str,
    cfg: CharConfig,
    macro: str,
    addr_bits: int,
    bits: int,
) -> float:
    """Bisect to find minimum CLK high pulse width (ns)."""
    from fabram.characterize.testbench import build_pulse_width_testbench

    lo, hi = 0.0, cfg.clk_period / 2

    for _ in range(cfg.max_iterations):
        if (hi - lo) < cfg.timing_resolution:
            break
        mid = (lo + hi) / 2.0
        deck = build_pulse_width_testbench(
            netlist_path, cfg, macro, addr_bits, bits, pw=mid
        )
        meas = run_ngspice(deck, cfg.sim_timeout)
        if is_correct(meas.get("q_sample"), cfg):
            hi = mid
        else:
            lo = mid

    return hi


# ─────────────────────────────────────────────────────────────────────────────
# Leakage and dynamic power
# ─────────────────────────────────────────────────────────────────────────────

def measure_leakage(
    netlist_path: str,
    cfg: CharConfig,
    macro: str,
    addr_bits: int,
    bits: int,
) -> float:
    """Return standby leakage power in nW.

    Window = cycles 2-3 of the 3-cycle testbench = 2×T seconds.
    Uses vvss#branch; abs() handles sign convention (current sinks into GND).
    """
    deck = build_leakage_testbench(netlist_path, cfg, macro, addr_bits, bits)
    meas = run_ngspice(deck, cfg.sim_timeout)

    i_avg = meas.get("i_avg")    # A·s  (INTEG of vvss#branch over window)
    if i_avg is None:
        return 0.0

    t_win_s = 2.0 * cfg.clk_period * 1e-9   # ns → s
    i_avg_A = abs(i_avg) / t_win_s
    return i_avg_A * cfg.vdd * 1e9          # W → nW


def measure_dynamic_power(
    netlist_path: str,
    cfg: CharConfig,
    macro: str,
    addr_bits: int,
    bits: int,
) -> dict:
    """Return read and write dynamic power in nW (averaged over one CLK cycle).

    Returns:
        ``write_power`` (nW), ``read_power`` (nW)
    """
    t_cycle_s = cfg.clk_period * 1e-9   # window = 1 cycle (T to 2T in testbench)
    results = {}
    for op in ("write", "read"):
        deck = build_power_testbench(netlist_path, cfg, macro, addr_bits, bits, op=op)
        meas = run_ngspice(deck, cfg.sim_timeout)
        i_op = meas.get("i_op")    # A·s (integral of vdd#branch over cycle 2)
        if i_op is None:
            results[f"{op}_power"] = 0.0
            continue
        i_avg_A = abs(i_op) / t_cycle_s
        results[f"{op}_power"] = i_avg_A * cfg.vdd * 1e9   # → nW

    return results


# ─────────────────────────────────────────────────────────────────────────────
# All-pins setup/hold sweep (convenience wrapper used by CharCompiler)
# ─────────────────────────────────────────────────────────────────────────────

_PINS = ["CS", "WRITE", "addr0", "din0"]


def measure_all_setup_hold(
    netlist_path: str,
    cfg: CharConfig,
    macro: str,
    addr_bits: int,
    bits: int,
) -> dict[str, dict]:
    """Run setup/hold characterization for all constrained pins.

    Returns ``{pin_name: {"setup": 2D-list, "hold": 2D-list}}``.
    """
    return {
        pin: measure_setup_hold(netlist_path, cfg, macro, addr_bits, bits, pin)
        for pin in _PINS
    }
