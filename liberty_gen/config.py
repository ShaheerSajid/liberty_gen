"""CharConfig — characterization parameters for the SRAM Liberty flow."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CharConfig:
    """All parameters that control the characterization sweep.

    Defaults are reasonable for sky130A 1.8 V at 27 °C with a 10 ns test
    clock.  Reduce ``input_slews`` / ``output_loads`` table sizes or increase
    ``sim_timestep`` for faster (less accurate) runs.

    Example — quick characterization for debugging::

        cfg = CharConfig(
            input_slews=[0.05, 0.2],
            output_loads=[0.005, 0.05],
            sim_timestep=0.005,   # 5 ps
            max_workers=8,
        )
    """

    # ── Operating point ───────────────────────────────────────────────────────
    vdd: float = 1.8         # Supply voltage (V)
    temp: float = 27.0       # Temperature (°C)

    # ── LUT index vectors ─────────────────────────────────────────────────────
    # Deterministic, log-spaced. No random factors (unlike some other tools).
    input_slews: list[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.1, 0.2, 0.5]
    )   # Input transition times (ns) — used for CLK slew AND pin slew axes
    output_loads: list[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1]
    )   # Output load capacitances (pF) — for CLK-to-Q arcs only

    # ── Voltage thresholds (fractions of VDD) ────────────────────────────────
    th_lo:  float = 0.1    # Low  threshold: 10 % of VDD  (transition start)
    th_hi:  float = 0.9    # High threshold: 90 % of VDD  (transition end)
    th_mid: float = 0.5    # Mid  threshold: 50 % of VDD  (propagation ref)

    # ── Bisection control ─────────────────────────────────────────────────────
    timing_resolution: float = 0.001   # ns — stop bisection when bracket < this
    max_iterations:    int   = 60      # Hard cap; 60 iters covers 2^60 range

    # ── Simulation control ────────────────────────────────────────────────────
    clk_period:         float = 10.0   # ns — simulation clock period (>> expected tclkq)
    sim_timestep:       float = 0.01   # ns (10 ps) — default timestep; use 1 ps for high precision
    power_sim_timestep: float = 0.01   # ns (10 ps) — leakage/power sim timestep
    sim_timeout:        int   = 300    # seconds per ngspice subprocess invocation

    # ── Parallelism ───────────────────────────────────────────────────────────
    max_workers: int = 4    # ThreadPoolExecutor workers for parallel simulations
