"""
SPICE testbench generators for SRAM characterization.

Each function returns a complete ngspice batch deck (string) that can be
written to a temp file and run with ``ngspice -b``.

Design decisions
----------------
* DUT netlist is referenced via ``.include`` (written once externally).
* All signal timing is computed in nanoseconds in Python, converted to
  seconds only when emitting SPICE text.
* Non-constrained signals use a 1 ps (0.001 ns) transition ramp.
* Constrained pins use ``ramp = pin_slew / (th_hi - th_lo)`` so the
  actual 10 %–90 % transition equals ``pin_slew`` exactly.
* Characterization always writes ``q_val=1`` to addr=0 so that timing
  violations are detected as Q0 < VDD/2 (unambiguous with .nodeset all=0).
* ``.save`` limits stored nodes to only those needed for ``.meas``.

4-cycle simulation structure
-----------------------------
    Cycle:  1 (INIT)    2 (WRITE)   3 (READ)    4 (sample)
    CLK:     ___         ___         ___
             T           2T          3T          4T
    CS:      0           1           1           —
    WRITE:   0           1           0           —
    addr:    0           0           0 (varies)  —
    din:     0           data        —           —
    Q0:      X           X           settling    → sampled at 3.5T

CLK edges (50 % crossing of PULSE):
    edge_n = (n-1)*T + ramp_clk * th_mid      (n = 1, 2, 3, …)
"""
from __future__ import annotations
from liberty_gen.config import CharConfig

# Fixed ramp for non-constrained digital transitions (1 ps)
_FAST_RAMP_NS = 0.001


def _s(ns: float) -> str:
    """Convert nanoseconds to seconds string for SPICE."""
    return f"{ns * 1e-9:.6e}"


def _pwl(pairs: list[tuple[float, float]]) -> str:
    """Convert [(t_ns, v), ...] to 'PWL(t1_s v1 t2_s v2 ...)'."""
    return "PWL(" + " ".join(f"{t * 1e-9:.6e} {v:.6e}" for t, v in pairs) + ")"


def _ramp(slew_ns: float, th_lo: float, th_hi: float) -> float:
    """Actual rise/fall time (ns) of a linear ramp with slew = 10%–90% time."""
    return slew_ns / (th_hi - th_lo)


def _dut_ports(addr_bits: int, bits: int) -> str:
    """Build the full DUT port list string matching SRAMCompiler._build_top order.

    VDD VSS CLK CS WRITE addr{N-1}..addr0 din{B-1}..din0 Q{B-1}..Q0
    """
    ports = ["VDD", "VSS", "CLK", "CS", "WRITE"]
    ports += [f"addr{i}" for i in range(addr_bits - 1, -1, -1)]
    ports += [f"din{k}" for k in range(bits - 1, -1, -1)]
    ports += [f"Q{k}" for k in range(bits - 1, -1, -1)]
    return " ".join(ports)


def _save_nodes(bits: int) -> str:
    """Only save the nodes needed for measurements to reduce memory."""
    nodes = ["v(CLK)", "v(Q0)", "v(CS)", "v(WRITE)", "v(addr0)", "v(din0)"]
    return " ".join(nodes)


def _common_header(netlist_path: str, cfg: CharConfig, macro_name: str,
                   addr_bits: int, bits: int, uic: bool = False) -> str:
    """Common deck header: include, supplies, DUT instance.

    uic=True omits the .nodeset line — for UIC sims ngspice initialises all
    nodes to 0 automatically (no DCOP solve, no .nodeset needed).
    """
    ports = _dut_ports(addr_bits, bits)
    vdd = cfg.vdd
    nodeset = "" if uic else "\n* Start all internal nodes at 0 to help DCOP convergence\n.nodeset all=0"
    return f"""\
* fabram characterization testbench — {macro_name}
.include "{netlist_path}"
.temp {cfg.temp}

* Supply
VVDD VDD 0 DC {vdd}
VVSS VSS 0 DC 0

* DUT instance
XDUT {ports} {macro_name}
{nodeset}
"""


def _control_block(timestep_ns: float, end_ns: float, save_nodes: str,
                   meas_lines: list[str], uic: bool = False,
                   svg_path: str | None = None,
                   svg_nodes: list[str] | None = None) -> str:
    """Wrap the .tran and .meas directives in a .control block.

    uic=True adds 'UIC' to .tran.
    svg_path / svg_nodes: if given, emit a ``wrdata`` command after measurements
    to write ASCII waveform data that can be plotted externally (matplotlib).
    ``hardcopy`` is intentionally avoided because it requires X11.
    """
    meas_str = "\n".join(f"  {m}" for m in meas_lines)
    uic_str  = " UIC" if uic else ""
    # wrdata writes: time col1 col2 ... (space-separated, one row per timestep)
    wrdata_str = (f"  wrdata {svg_path} {' '.join(svg_nodes or [])}\n"
                  if svg_path else "")
    return f"""\
.save {save_nodes}
.control
  run
{meas_str}
{wrdata_str}  quit
.endc
.tran {_s(timestep_ns)} {_s(end_ns)}{uic_str}
.end
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLK-to-Q propagation testbench
# ─────────────────────────────────────────────────────────────────────────────

def build_clkq_testbench(
    netlist_path: str,
    cfg: CharConfig,
    macro_name: str,
    addr_bits: int,
    bits: int,
    clk_slew: float,
    load_pf: float,
    q_val: int,            # 0 or 1 — which arc to measure
    svg_path: str | None = None,
    svg_nodes: list[str] | None = None,
) -> str:
    """Return an ngspice deck measuring CLK-to-Q propagation and transition times.

    Two structures depending on q_val, both ensuring Q transitions AFTER the
    TRIG clock edge so that tpd is positive:

    q_val=1 (rise arc) — 5-cycle, TRIG = CLK RISE=3:
        Cycle 1 (INIT):      CS=0, Q=VDD from DCOP
        Cycle 2 (WRITE 0):   CS=1, WRITE=1, din0=0  → Q falls to 0
        Cycle 3 (WRITE 1):   CS=1, WRITE=1, din0=VDD → Q rises ← TRIG CLK RISE=3
        Cycle 4 (idle):      end

    q_val=0 (fall arc) — 3-cycle, TRIG = CLK RISE=2:
        Cycle 1 (INIT):      CS=0, Q=VDD from DCOP
        Cycle 2 (WRITE 0):   CS=1, WRITE=1, din0=0  → Q falls ← TRIG CLK RISE=2
        Cycle 3 (idle):      end

    The self-timed SRAM activates SAEN (and drives Q) during WRITE cycles, so
    Q updates right after the triggering CLK edge in both cases.
    """
    T = cfg.clk_period
    vdd = cfg.vdd
    ramp_c = _ramp(clk_slew, cfg.th_lo, cfg.th_hi)
    r = _FAST_RAMP_NS

    # Absolute voltage thresholds
    vmid = vdd * cfg.th_mid
    vlo  = vdd * cfg.th_lo
    vhi  = vdd * cfg.th_hi

    # CLK: 50% duty cycle PULSE — same for both arcs
    pw_ns = T / 2.0 - ramp_c
    clk_src = (f"VCLK CLK 0 PULSE(0 {vdd} 0 {_s(ramp_c)} {_s(ramp_c)}"
               f" {_s(pw_ns)} {_s(T)})")

    if q_val == 1:
        # ── Rise arc ───────────────────────────────────────────────────────
        end_ns = 4.0 * T
        # CS: high from cycle 2 onward
        cs_src = f"VCS CS 0 {_pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (end_ns, vdd)])}"
        # WRITE: high through cycles 2 AND 3 (0.75T → 2.75T)
        wr_src = (f"VWRITE WRITE 0 "
                  f"{_pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (2.75*T, vdd), (2.75*T+r, 0), (end_ns, 0)])}")
        # din0: 0 in cycle 2, transitions to VDD at 1.5T (before CLK RISE=3 at 2T)
        din0_src = (f"VDIN0 din0 0 "
                    f"{_pwl([(0, 0), (1.5*T, 0), (1.5*T+r, vdd), (end_ns, vdd)])}")
        trig_rise = 3
    else:
        # ── Fall arc ───────────────────────────────────────────────────────
        end_ns = 3.0 * T
        # CS: high from cycle 2 onward
        cs_src = f"VCS CS 0 {_pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (end_ns, vdd)])}"
        # WRITE: high only cycle 2 (0.75T → 1.75T)
        wr_src = (f"VWRITE WRITE 0 "
                  f"{_pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (1.75*T, vdd), (1.75*T+r, 0), (end_ns, 0)])}")
        # din0: always 0 (write 0 to cell, Q falls from VDD)
        din0_src = "VDIN0 din0 0 DC 0"
        trig_rise = 2

    # addr0: always 0
    a0_src = "VADDR0 addr0 0 DC 0"
    other_addr = "\n".join(f"VADDR{i} addr{i} 0 DC 0" for i in range(1, addr_bits))
    other_din  = "\n".join(f"VDIN{k} din{k} 0 DC 0"  for k in range(1, bits))

    # Load cap on Q0
    cload = f"CLOAD Q0 0 {load_pf * 1e-12:.4e}"

    # .meas directives
    meas = [
        f"meas tran tpd_rise TRIG v(CLK) VAL={vmid:.4f} RISE={trig_rise} TARG v(Q0) VAL={vmid:.4f} RISE=1",
        f"meas tran tpd_fall TRIG v(CLK) VAL={vmid:.4f} RISE={trig_rise} TARG v(Q0) VAL={vmid:.4f} FALL=1",
        f"meas tran ttr_rise TRIG v(Q0)  VAL={vlo:.4f}  RISE=1 TARG v(Q0)  VAL={vhi:.4f}  RISE=1",
        f"meas tran ttr_fall TRIG v(Q0)  VAL={vhi:.4f}  FALL=1 TARG v(Q0)  VAL={vlo:.4f}  FALL=1",
    ]

    header = _common_header(netlist_path, cfg, macro_name, addr_bits, bits)
    body = "\n".join(filter(None, [
        clk_src, cs_src, wr_src, a0_src, other_addr, din0_src, other_din, cload,
    ]))
    ctrl = _control_block(cfg.sim_timestep, end_ns, _save_nodes(bits), meas,
                          svg_path=svg_path, svg_nodes=svg_nodes)
    return f"{header}\n{body}\n\n{ctrl}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Setup / Hold constraint testbench
# ─────────────────────────────────────────────────────────────────────────────

def build_constraint_testbench(
    netlist_path: str,
    cfg: CharConfig,
    macro_name: str,
    addr_bits: int,
    bits: int,
    pin: str,              # "addr0" | "din0" | "CS" | "WRITE"
    clk_slew: float,       # ns
    pin_slew: float,       # ns — 10%–90% transition time for constrained pin
    t_arrive: float,       # ns (absolute) — when pin transitions to "correct" value
    mode: str,             # "setup" | "hold"
) -> str:
    """Return a deck measuring whether timing is met for a given pin arrival time.

    The testbench writes 1 to addr=0 in cycle 2 (always q_val=1), then reads
    addr=0 in cycle 3 with the constrained pin transitioning at ``t_arrive``.

    The caller checks Q0 > VDD*th_mid in the parsed results to determine pass/fail.

    Setup: pin transitions wrong→correct at t_arrive (before CLK edge)
    Hold:  pin was correct before CLK edge, transitions correct→wrong at t_arrive
           (after CLK edge)
    """
    T = cfg.clk_period
    vdd = cfg.vdd
    ramp_c  = _ramp(clk_slew, cfg.th_lo, cfg.th_hi)   # CLK ramp
    ramp_p  = _ramp(pin_slew, cfg.th_lo, cfg.th_hi)   # constrained pin ramp
    r       = _FAST_RAMP_NS                            # fast ramp for others
    end_ns  = 4.0 * T

    # CLK source
    pw_ns  = T / 2.0 - ramp_c
    clk_src = (f"VCLK CLK 0 PULSE(0 {vdd} 0 {_s(ramp_c)} {_s(ramp_c)}"
               f" {_s(pw_ns)} {_s(T)})")

    # Default stable values for non-constrained signals
    # (overridden below for the pin under test)
    cs_default    = True   # CS high for cycles 2+3
    write_default = True   # WRITE high only cycle 2
    addr0_default = True   # addr0 = 0 throughout
    din0_default  = True   # din0 = 1 throughout (write 1 to addr=0)

    # ── Generate sources based on pin under test ──────────────────────────────

    # Pre-compute common PWL waveforms (avoids nested f-string issues on Py<3.12)
    _cs_pwl   = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (end_ns, vdd)])
    _wr_pwl   = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd),
                      (1.75*T, vdd), (1.75*T+r, 0), (end_ns, 0)])
    _din1_pwl = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (end_ns, vdd)])

    if pin == "addr0":
        # addr0 is the constrained pin relative to CLK3 (READ edge)
        # "correct" = 0 (addr=0 = where we wrote 1), "wrong" = 1 (addr=1 = unwritten = 0)
        # Setup: addr0 at 0 for write, transitions to 1 (wrong) at ~1.25T,
        #        then back to 0 (correct) at t_arrive
        # Hold:  addr0 at 0 (correct) transitions to 1 (wrong) at t_arrive
        if mode == "setup":
            wrong_rise_t = 1.25 * T     # transition to "wrong" (1) before READ
            _a0_pwl = _pwl([(0, 0), (wrong_rise_t, 0), (wrong_rise_t + r, vdd),
                            (t_arrive, vdd), (t_arrive + ramp_p, 0), (end_ns, 0)])
        else:  # hold
            _a0_pwl = _pwl([(0, 0), (t_arrive, 0), (t_arrive + ramp_p, vdd), (end_ns, vdd)])
        addr0_src = f"VADDR0 addr0 0 {_a0_pwl}"
        addr0_default = False
        cs_src   = f"VCS CS 0 {_cs_pwl}"
        wr_src   = f"VWRITE WRITE 0 {_wr_pwl}"
        din0_src = f"VDIN0 din0 0 {_din1_pwl}"

    elif pin == "din0":
        # din0 constrained relative to CLK2 (WRITE edge)
        # "correct" = 1 (write 1 to addr=0), "wrong" = 0 (write nothing useful)
        # Setup: din0 transitions from 0 to 1 at t_arrive (before CLK2)
        # Hold:  din0 is at 1 before CLK2, transitions to 0 at t_arrive (after CLK2)
        if mode == "setup":
            _d0_pwl = _pwl([(0, 0), (t_arrive, 0), (t_arrive + ramp_p, vdd), (end_ns, vdd)])
        else:  # hold
            _d0_pwl = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd),
                            (t_arrive, vdd), (t_arrive + ramp_p, 0), (end_ns, 0)])
        din0_src = f"VDIN0 din0 0 {_d0_pwl}"
        din0_default = False
        cs_src    = f"VCS CS 0 {_cs_pwl}"
        wr_src    = f"VWRITE WRITE 0 {_wr_pwl}"
        addr0_src = "VADDR0 addr0 0 DC 0"
        addr0_default = False  # override with explicit DC

    elif pin == "CS":
        # CS constrained relative to CLK2 (WRITE edge)
        # Setup: CS transitions from 0 to 1 at t_arrive (before CLK2)
        #        CS must stay high for READ cycle (cycle 3) too
        # Hold:  CS is at 1 before CLK2, transitions to 0 at t_arrive (after CLK2)
        #        CS comes back high before CLK3 (READ must still work)
        if mode == "setup":
            _csp = _pwl([(0, 0), (t_arrive, 0), (t_arrive + ramp_p, vdd), (end_ns, vdd)])
        else:  # hold — CS goes low after CLK2 write, comes back up before CLK3 read
            _csp = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd),
                         (t_arrive, vdd), (t_arrive + ramp_p, 0),
                         (1.75*T, 0), (1.75*T + r, vdd), (end_ns, vdd)])
        cs_src = f"VCS CS 0 {_csp}"
        cs_default = False
        wr_src   = f"VWRITE WRITE 0 {_wr_pwl}"
        din0_src = f"VDIN0 din0 0 {_din1_pwl}"
        din0_default = False
        addr0_src = "VADDR0 addr0 0 DC 0"
        addr0_default = False

    else:  # pin == "WRITE"
        # WRITE constrained relative to CLK2 (WRITE edge)
        # Setup: WRITE transitions from 0 to 1 at t_arrive, back to 0 before CLK3
        # Hold:  WRITE is at 1 before CLK2, transitions to 0 at t_arrive (after CLK2)
        if mode == "setup":
            _wrp = _pwl([(0, 0), (t_arrive, 0), (t_arrive + ramp_p, vdd),
                         (1.75*T, vdd), (1.75*T + r, 0), (end_ns, 0)])
        else:  # hold
            _wrp = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd),
                         (t_arrive, vdd), (t_arrive + ramp_p, 0), (end_ns, 0)])
        wr_src = f"VWRITE WRITE 0 {_wrp}"
        write_default = False
        cs_src   = f"VCS CS 0 {_cs_pwl}"
        din0_src = f"VDIN0 din0 0 {_din1_pwl}"
        din0_default = False
        addr0_src = "VADDR0 addr0 0 DC 0"
        addr0_default = False

    # Fill in defaults for pins not overridden
    if cs_default:
        cs_src    = f"VCS CS 0 {_cs_pwl}"
    if write_default:
        wr_src    = f"VWRITE WRITE 0 {_wr_pwl}"
    if addr0_default:
        addr0_src = "VADDR0 addr0 0 DC 0"
    if din0_default:
        din0_src  = f"VDIN0 din0 0 {_din1_pwl}"

    # Remaining addr and din bits
    other_addr = "\n".join(f"VADDR{i} addr{i} 0 DC 0" for i in range(1, addr_bits))
    other_din  = "\n".join(f"VDIN{k} din{k} 0 DC 0" for k in range(1, bits))

    # Sample Q0 at mid-cycle 4
    t_sample = 3.5 * T
    vmid = vdd * cfg.th_mid
    meas = [
        f"meas tran q_sample FIND v(Q0) AT={_s(t_sample)}",
    ]

    header = _common_header(netlist_path, cfg, macro_name, addr_bits, bits)
    body = "\n".join(filter(None, [
        clk_src, cs_src, wr_src, addr0_src, other_addr, din0_src, other_din,
    ]))
    ctrl = _control_block(cfg.sim_timestep, end_ns, _save_nodes(bits), meas)
    return f"{header}\n{body}\n\n{ctrl}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Minimum pulse width testbench
# ─────────────────────────────────────────────────────────────────────────────

def build_pulse_width_testbench(
    netlist_path: str,
    cfg: CharConfig,
    macro_name: str,
    addr_bits: int,
    bits: int,
    pw: float,             # CLK high pulse width to test (ns)
) -> str:
    """Return a deck measuring whether ``pw`` is a sufficient CLK high pulse width.

    Runs WRITE (cycle 2) then READ (cycle 3) with a truncated CLK high time.
    Checks Q0 at 3.5T; pass = Q0 > VDD/2.
    """
    T = cfg.clk_period
    vdd = cfg.vdd
    r = _FAST_RAMP_NS
    end_ns = 4.0 * T

    # Custom CLK: use pw as the high pulse width (not T/2)
    # PULSE(V1 V2 TD TR TF PW PER)  — TR=TF=r (fast edges)
    clk_src = f"VCLK CLK 0 PULSE(0 {vdd} 0 {_s(r)} {_s(r)} {_s(pw)} {_s(T)})"

    cs_src  = f"VCS CS 0 {_pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (end_ns, vdd)])}"
    wr_src  = (f"VWRITE WRITE 0 "
               f"{_pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (1.75*T, vdd), (1.75*T+r, 0), (end_ns, 0)])}")
    addr0_src = "VADDR0 addr0 0 DC 0"
    din0_src  = (f"VDIN0 din0 0 {_pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (end_ns, vdd)])}")
    other_addr = "\n".join(f"VADDR{i} addr{i} 0 DC 0" for i in range(1, addr_bits))
    other_din  = "\n".join(f"VDIN{k} din{k} 0 DC 0" for k in range(1, bits))

    t_sample = 3.5 * T
    meas = [f"meas tran q_sample FIND v(Q0) AT={_s(t_sample)}"]

    header = _common_header(netlist_path, cfg, macro_name, addr_bits, bits)
    body = "\n".join(filter(None, [
        clk_src, cs_src, wr_src, addr0_src, other_addr, din0_src, other_din,
    ]))
    ctrl = _control_block(cfg.sim_timestep, end_ns, _save_nodes(bits), meas)
    return f"{header}\n{body}\n\n{ctrl}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Leakage / dynamic power testbenches
# ─────────────────────────────────────────────────────────────────────────────

def build_leakage_testbench(
    netlist_path: str,
    cfg: CharConfig,
    macro_name: str,
    addr_bits: int,
    bits: int,
    svg_path: str | None = None,
    svg_nodes: list[str] | None = None,
) -> str:
    """Standby leakage: CS=0, all inputs tied to 0, DCOP via .nodeset all=0.

    Structure (3 cycles, measure over last 2):
      Cycle 1: settle from DCOP
      Cycles 2-3: measurement window (CS=0 standby throughout)

    Uses vvss#branch (current into the VSS source) following the same sign
    convention as the reference implementation.  Leakage = VDD × |avg_i_gnd|.
    """
    T   = cfg.clk_period
    vdd = cfg.vdd
    total_ns = 3.0 * T
    start_ns = T          # skip cycle 1 (DCOP transient)

    # .nodeset all=0 from _common_header handles DCOP convergence; no UIC.
    header = _common_header(netlist_path, cfg, macro_name, addr_bits, bits, uic=False)

    ramp_c  = _ramp(cfg.input_slews[0], cfg.th_lo, cfg.th_hi)
    pw_ns   = T / 2.0 - ramp_c
    clk_src = f"VCLK CLK 0 PULSE(0 {vdd} 0 {_s(ramp_c)} {_s(ramp_c)} {_s(pw_ns)} {_s(T)})"
    cs_src  = "VCS CS 0 DC 0"      # chip disabled throughout
    wr_src  = "VWRITE WRITE 0 DC 0"
    addr_src = "\n".join(f"VADDR{i} addr{i} 0 DC 0" for i in range(addr_bits))
    din_src  = "\n".join(f"VDIN{k} din{k} 0 DC 0"  for k in range(bits))

    # vvss#branch = current into the VSS (ground) source; negate → positive leakage
    meas = [
        f"meas tran i_avg INTEG vvss#branch FROM={_s(start_ns)} TO={_s(total_ns)}",
    ]

    body = "\n".join([clk_src, cs_src, wr_src, addr_src, din_src])
    ts = getattr(cfg, "power_sim_timestep", cfg.sim_timestep)
    ctrl = _control_block(ts, total_ns, "v(CLK) vvss#branch", meas,
                          svg_path=svg_path, svg_nodes=svg_nodes)
    return f"{header}\n{body}\n\n{ctrl}"


def build_power_testbench(
    netlist_path: str,
    cfg: CharConfig,
    macro_name: str,
    addr_bits: int,
    bits: int,
    op: str,               # "write" or "read"
    load_pf: float = 0.01,
    svg_path: str | None = None,
    svg_nodes: list[str] | None = None,
) -> str:
    """Dynamic power for one operation (write or read) over one CLK cycle."""
    T = cfg.clk_period
    vdd = cfg.vdd
    r = _FAST_RAMP_NS
    # 3-cycle sim: INIT + target-op + idle
    end_ns = 3.0 * T
    ramp_c = _ramp(cfg.input_slews[0], cfg.th_lo, cfg.th_hi)
    pw_ns  = T / 2.0 - ramp_c
    clk_src = (f"VCLK CLK 0 PULSE(0 {vdd} 0 {_s(ramp_c)} {_s(ramp_c)}"
               f" {_s(pw_ns)} {_s(T)})")

    # Cycle 2 is the operation under test
    write_en = vdd if op == "write" else 0.0
    _cs_pwl  = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (end_ns, vdd)])
    _wr_pwl  = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, write_en),
                     (1.75*T, write_en), (1.75*T+r, 0), (end_ns, 0)])
    _din_pwl = _pwl([(0, 0), (0.75*T, 0), (0.75*T+r, vdd), (end_ns, vdd)])
    cs_src   = f"VCS CS 0 {_cs_pwl}"
    wr_src   = f"VWRITE WRITE 0 {_wr_pwl}"
    addr_src = "\n".join(f"VADDR{i} addr{i} 0 DC 0" for i in range(addr_bits))
    din_src  = "\n".join(f"VDIN{k} din{k} 0 {_din_pwl}" for k in range(bits))
    cload    = "\n".join(f"CLOAD{k} Q{k} 0 {load_pf * 1e-12:.4e}" for k in range(bits))

    # vvdd#branch = ngspice branch-current vector for voltage source VVDD
    # vvss#branch: current into ground source; consistent with leakage measurement
    meas = [
        f"meas tran i_op INTEG vvss#branch FROM={_s(T)} TO={_s(2.0*T)}",
    ]

    header = _common_header(netlist_path, cfg, macro_name, addr_bits, bits, uic=True)
    body = "\n".join(filter(None, [clk_src, cs_src, wr_src, addr_src, din_src, cload]))
    ts = getattr(cfg, "power_sim_timestep", cfg.sim_timestep)
    ctrl = _control_block(ts, end_ns, "v(CLK) v(Q0) vvss#branch", meas, uic=True,
                          svg_path=svg_path, svg_nodes=svg_nodes)
    return f"{header}\n{body}\n\n{ctrl}"
