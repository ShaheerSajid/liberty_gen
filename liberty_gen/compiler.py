"""
CharCompiler — orchestrates all characterization sweeps and renders Liberty.

Usage::

    from liberty_gen import CharConfig, CharCompiler

    char = CharCompiler(
        netlist_path="/tmp/sram.sp",
        macro="SRAM_32x4_CM4",
        addr_bits=5,
        bits=4,
    )
    lib_text = char.characterize()
    open("sram.lib", "w").write(lib_text)

The caller is responsible for writing the SPICE netlist to a file and cleaning
it up afterwards.  CharCompiler does not render SPICE — it takes a pre-rendered
``.sp`` path so that it has no dependency on the upstream netlist generator.
"""
from __future__ import annotations

import logging

from liberty_gen.config import CharConfig
from liberty_gen.timing import (
    measure_clkq,
    measure_all_setup_hold,
    measure_min_pulse_width,
    measure_leakage,
    measure_dynamic_power,
)
from liberty_gen.liberty import render_liberty

log = logging.getLogger(__name__)


class CharCompiler:
    """Run all characterization sweeps for a compiled SRAM netlist.

    Parameters
    ----------
    netlist_path:
        Path to a pre-rendered ngspice-compatible ``.sp`` file.
        The file must already exist; CharCompiler does not create or delete it.
    macro:
        SPICE subcircuit name of the SRAM macro (top-level .subckt identifier).
    addr_bits:
        Number of address bits.
    bits:
        Data width in bits.
    cfg:
        Characterization parameters.  Defaults to ``CharConfig()`` which uses
        sky130A-appropriate settings at TT / 27 °C / 1.8 V.
    """

    def __init__(
        self,
        netlist_path: str,
        macro: str,
        addr_bits: int,
        bits: int,
        cfg: CharConfig | None = None,
    ) -> None:
        self.netlist_path = netlist_path
        self.macro = macro
        self.addr_bits = addr_bits
        self.bits = bits
        self.cfg = cfg or CharConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def characterize(self) -> str:
        """Run all sweeps and return a complete Liberty (.lib) string.

        Measurement order
        -----------------
        1. Leakage power (standby, quick sim).
        2. Dynamic power (write + read, two sims).
        3. CLK-to-Q grid (parallel across slew × load × q_val).
        4. Setup / hold for all constrained pins (parallel outer).
        5. Minimum CLK pulse width (serial bisection).
        6. Render Liberty string.
        """
        return self._run_all(self.netlist_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_all(self, netlist_path: str) -> str:
        macro     = self.macro
        cfg       = self.cfg
        addr_bits = self.addr_bits
        bits      = self.bits

        log.info("[char] Measuring leakage …")
        leakage_nw = measure_leakage(netlist_path, cfg, macro, addr_bits, bits)
        log.info("[char]   leakage = %.3f nW", leakage_nw)

        log.info("[char] Measuring dynamic power …")
        dyn = measure_dynamic_power(netlist_path, cfg, macro, addr_bits, bits)
        write_power_nw = dyn.get("write_power", 0.0)
        read_power_nw  = dyn.get("read_power",  0.0)
        log.info(
            "[char]   write_power = %.3f nW  read_power = %.3f nW",
            write_power_nw, read_power_nw,
        )

        log.info("[char] Measuring CLK-to-Q (%d slews × %d loads × 2) …",
                 len(cfg.input_slews), len(cfg.output_loads))
        clkq_data = measure_clkq(netlist_path, cfg, macro, addr_bits, bits)
        log.info("[char]   CLK-to-Q done.")

        log.info("[char] Measuring setup/hold for all constrained pins …")
        sh_data = measure_all_setup_hold(netlist_path, cfg, macro, addr_bits, bits)
        log.info("[char]   setup/hold done.")

        log.info("[char] Measuring minimum CLK pulse width …")
        min_pw = measure_min_pulse_width(netlist_path, cfg, macro, addr_bits, bits)
        log.info("[char]   min_pulse_width = %.4f ns", min_pw)

        log.info("[char] Rendering Liberty …")
        lib_str = render_liberty(
            macro_name=macro,
            cfg=cfg,
            addr_bits=addr_bits,
            bits=bits,
            clkq_data=clkq_data,
            setup_hold_data=sh_data,
            min_pw=min_pw,
            leakage_nw=leakage_nw,
            write_power_nw=write_power_nw,
            read_power_nw=read_power_nw,
        )
        log.info("[char] Done.")
        return lib_str
