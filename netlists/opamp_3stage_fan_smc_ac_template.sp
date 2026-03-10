* Parametric 3-Stage Fan SMC Operational Amplifier Template - AC Analysis
* Same circuit as opamp_3stage_fan_smc_template.sp but with:
*   - Middlebrook loop-breaking network at inverting input (vinn)
*   - AC analysis directive for open-loop gain measurement
*   - gm/gds .save directives for small-signal parameter extraction
*
* Loop-breaking: vinn is split into vinn_fb (feedback side: Rin, Rfb)
*   and vinn_gate (amplifier side: M8 gate). At DC they are shorted
*   via Lbreak (1T inductor). At AC the loop is broken and Vac_inj
*   injects through Cac_inj (1T capacitor).
*   Loop gain: T(f) = -V(vinn_fb) / V(vinn_gate)
*
* Parameters (group-based): same as DC template

.title 3-Stage Fan SMC Operational Amplifier - AC Analysis

* Include SKY130 models
.param mc_mm_switch=0

* Define missing mismatch parameters
.param sky130_fd_pr__nfet_01v8__toxe_slope=0
.param sky130_fd_pr__nfet_01v8__toxe_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__vth0_slope=0
.param sky130_fd_pr__nfet_01v8__vth0_slope1=0
.param sky130_fd_pr__nfet_01v8__vth0_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__voff_slope=0
.param sky130_fd_pr__nfet_01v8__voff_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__nfactor_slope=0
.param sky130_fd_pr__pfet_01v8__toxe_slope=0
.param sky130_fd_pr__pfet_01v8__toxe_slope1=0
.param sky130_fd_pr__pfet_01v8__toxe_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__vth0_slope=0
.param sky130_fd_pr__pfet_01v8__vth0_slope1=0
.param sky130_fd_pr__pfet_01v8__vth0_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__voff_slope=0
.param sky130_fd_pr__pfet_01v8__voff_slope1=0
.param sky130_fd_pr__pfet_01v8__voff_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__nfactor_slope=0
.param sky130_fd_pr__pfet_01v8__nfactor_slope1=0

* Define diff parameters (process variation)
.param sky130_fd_pr__pfet_01v8__wlod_diff=0
.param sky130_fd_pr__pfet_01v8__kvth0_diff=0
.param sky130_fd_pr__pfet_01v8__lkvth0_diff=0
.param sky130_fd_pr__pfet_01v8__wkvth0_diff=0
.param sky130_fd_pr__pfet_01v8__ku0_diff=0
.param sky130_fd_pr__pfet_01v8__lku0_diff=0
.param sky130_fd_pr__pfet_01v8__wku0_diff=0
.param sky130_fd_pr__pfet_01v8__kvsat_diff=0

.include "/home/kevin/tech/sky130/libraries/sky130_fd_pr/latest/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice"
.include "/home/kevin/tech/sky130/libraries/sky130_fd_pr/latest/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"

* Power supplies
Vdd vdda 0 DC {VDD}
Vss gnda 0 DC 0

* Input configuration (inverting amplifier)
* MODIFIED: vinn split into vinn_fb (feedback) and vinn_gate (M8 gate)
* for Middlebrook loop gain measurement
Vcm_ref  vinp     0    DC {VIN_P}
Vsig     vsig     0    DC {VIN_N}
Rin      vsig     vinn_fb {R_IN}
Rfb      vout     vinn_fb {R_F}

* =========================
* LOOP-BREAKING NETWORK
* =========================
* Lbreak: DC short (preserves bias), AC open (breaks loop)
* Cac_inj + Vac_inj: DC open (no bias effect), AC short (injects signal)
Lbreak  vinn_fb  vinn_gate  1e12
Vac_inj vac_node 0 DC 0 AC 1
Cac_inj vac_node vinn_gate 1e12

* =========================
* PMOS BIAS MIRROR (M0-M7)
* =========================

* M0: diode-connected PMOS bias reference
Xm0 nbias nbias vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M1: bias mirror -> VB4 generation
Xm1 vb4 nbias vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M2: bias mirror -> DM_1 (cascode bias branch)
Xm2 dm_1 nbias vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M3: bias mirror -> VB3 generation
Xm3 vb3 nbias vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M4: tail current for diff pair (4x multiplier)
Xm4 net31 nbias vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* M5: diode-connected at VOUTN (common-mode feedback)
Xm5 voutn voutn vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M6: mirror from VOUTN -> net050
Xm6 net050 voutn vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M7: bias mirror -> net049 (stage 2 output node bias)
Xm7 net049 nbias vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* =========================
* STAGE 1: PMOS DIFF PAIR
* =========================

* M8: inverting input (VINN) — MODIFIED: gate connects to vinn_gate
Xm8 dm_2 vinn_gate net31 net31 sky130_fd_pr__pfet_01v8 W={W_GM1} L={L_GM1} M={M_GM1}

* M9: non-inverting input (VINP)
Xm9 net063 vinp net31 net31 sky130_fd_pr__pfet_01v8 W={W_GM1} L={L_GM1} M={M_GM1}

* =========================
* STAGE 1: NMOS CASCODE LOAD
* =========================

* Bottom bias devices (VB4 gate)
Xm19 dm_2 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'
Xm20 net063 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* Cascode devices (VB3 gate)
Xm15 voutn vb3 dm_2 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm16 net050 vb3 net063 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* =========================
* NMOS BIAS GENERATION
* =========================

Xm14 vb3 vb3 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M={M_BIASCM_N}
Xm17 net54 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm12 vb4 vb3 net54 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm18 net56 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm13 dm_1 vb3 net56 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* =========================
* STAGE 2: COMMON-SOURCE
* =========================

Xm10 net043 net050 vdda vdda sky130_fd_pr__pfet_01v8 W={W_GM2} L={L_GM2} M={M_GM2}
Xm21 net043 net043 gnda gnda sky130_fd_pr__nfet_01v8 W={W_LOAD2} L={L_LOAD2} M={M_LOAD2}
Xm22 net049 net043 gnda gnda sky130_fd_pr__nfet_01v8 W={W_LOAD2} L={L_LOAD2} M={M_LOAD2}

* =========================
* STAGE 3: PUSH-PULL OUTPUT
* =========================

Xm11 vout net050 vdda vdda sky130_fd_pr__pfet_01v8 W={W_GMF2} L={L_GMF2} M={M_GMF2}
Xm23 vout net049 gnda gnda sky130_fd_pr__nfet_01v8 W={W_GM3} L={L_GM3} M={M_GM3}

* =========================
* BIAS CURRENT SOURCE
* =========================

I0 nbias gnda DC {I_BIAS}

* =========================
* COMPENSATION NETWORK
* =========================

C0 net050 vout {C_COMP}

* =========================
* OUTPUT LOAD
* =========================

Cload vout 0 10p

* =========================
* SAVE DEVICE CURRENTS
* =========================

.save all
* MOSFET drain currents (PMOS)
.save @m.xm0.msky130_fd_pr__pfet_01v8[id]
.save @m.xm1.msky130_fd_pr__pfet_01v8[id]
.save @m.xm2.msky130_fd_pr__pfet_01v8[id]
.save @m.xm3.msky130_fd_pr__pfet_01v8[id]
.save @m.xm4.msky130_fd_pr__pfet_01v8[id]
.save @m.xm5.msky130_fd_pr__pfet_01v8[id]
.save @m.xm6.msky130_fd_pr__pfet_01v8[id]
.save @m.xm7.msky130_fd_pr__pfet_01v8[id]
.save @m.xm8.msky130_fd_pr__pfet_01v8[id]
.save @m.xm9.msky130_fd_pr__pfet_01v8[id]
.save @m.xm10.msky130_fd_pr__pfet_01v8[id]
.save @m.xm11.msky130_fd_pr__pfet_01v8[id]
* MOSFET drain currents (NMOS)
.save @m.xm12.msky130_fd_pr__nfet_01v8[id]
.save @m.xm13.msky130_fd_pr__nfet_01v8[id]
.save @m.xm14.msky130_fd_pr__nfet_01v8[id]
.save @m.xm15.msky130_fd_pr__nfet_01v8[id]
.save @m.xm16.msky130_fd_pr__nfet_01v8[id]
.save @m.xm17.msky130_fd_pr__nfet_01v8[id]
.save @m.xm18.msky130_fd_pr__nfet_01v8[id]
.save @m.xm19.msky130_fd_pr__nfet_01v8[id]
.save @m.xm20.msky130_fd_pr__nfet_01v8[id]
.save @m.xm21.msky130_fd_pr__nfet_01v8[id]
.save @m.xm22.msky130_fd_pr__nfet_01v8[id]
.save @m.xm23.msky130_fd_pr__nfet_01v8[id]
* MOSFET operating point parameters (PMOS)
.save @m.xm0.msky130_fd_pr__pfet_01v8[vgs] @m.xm0.msky130_fd_pr__pfet_01v8[vds] @m.xm0.msky130_fd_pr__pfet_01v8[vdsat] @m.xm0.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm1.msky130_fd_pr__pfet_01v8[vgs] @m.xm1.msky130_fd_pr__pfet_01v8[vds] @m.xm1.msky130_fd_pr__pfet_01v8[vdsat] @m.xm1.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm2.msky130_fd_pr__pfet_01v8[vgs] @m.xm2.msky130_fd_pr__pfet_01v8[vds] @m.xm2.msky130_fd_pr__pfet_01v8[vdsat] @m.xm2.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm3.msky130_fd_pr__pfet_01v8[vgs] @m.xm3.msky130_fd_pr__pfet_01v8[vds] @m.xm3.msky130_fd_pr__pfet_01v8[vdsat] @m.xm3.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm4.msky130_fd_pr__pfet_01v8[vgs] @m.xm4.msky130_fd_pr__pfet_01v8[vds] @m.xm4.msky130_fd_pr__pfet_01v8[vdsat] @m.xm4.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm5.msky130_fd_pr__pfet_01v8[vgs] @m.xm5.msky130_fd_pr__pfet_01v8[vds] @m.xm5.msky130_fd_pr__pfet_01v8[vdsat] @m.xm5.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm6.msky130_fd_pr__pfet_01v8[vgs] @m.xm6.msky130_fd_pr__pfet_01v8[vds] @m.xm6.msky130_fd_pr__pfet_01v8[vdsat] @m.xm6.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm7.msky130_fd_pr__pfet_01v8[vgs] @m.xm7.msky130_fd_pr__pfet_01v8[vds] @m.xm7.msky130_fd_pr__pfet_01v8[vdsat] @m.xm7.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm8.msky130_fd_pr__pfet_01v8[vgs] @m.xm8.msky130_fd_pr__pfet_01v8[vds] @m.xm8.msky130_fd_pr__pfet_01v8[vdsat] @m.xm8.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm9.msky130_fd_pr__pfet_01v8[vgs] @m.xm9.msky130_fd_pr__pfet_01v8[vds] @m.xm9.msky130_fd_pr__pfet_01v8[vdsat] @m.xm9.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm10.msky130_fd_pr__pfet_01v8[vgs] @m.xm10.msky130_fd_pr__pfet_01v8[vds] @m.xm10.msky130_fd_pr__pfet_01v8[vdsat] @m.xm10.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm11.msky130_fd_pr__pfet_01v8[vgs] @m.xm11.msky130_fd_pr__pfet_01v8[vds] @m.xm11.msky130_fd_pr__pfet_01v8[vdsat] @m.xm11.msky130_fd_pr__pfet_01v8[vth]
* MOSFET operating point parameters (NMOS)
.save @m.xm12.msky130_fd_pr__nfet_01v8[vgs] @m.xm12.msky130_fd_pr__nfet_01v8[vds] @m.xm12.msky130_fd_pr__nfet_01v8[vdsat] @m.xm12.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm13.msky130_fd_pr__nfet_01v8[vgs] @m.xm13.msky130_fd_pr__nfet_01v8[vds] @m.xm13.msky130_fd_pr__nfet_01v8[vdsat] @m.xm13.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm14.msky130_fd_pr__nfet_01v8[vgs] @m.xm14.msky130_fd_pr__nfet_01v8[vds] @m.xm14.msky130_fd_pr__nfet_01v8[vdsat] @m.xm14.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm15.msky130_fd_pr__nfet_01v8[vgs] @m.xm15.msky130_fd_pr__nfet_01v8[vds] @m.xm15.msky130_fd_pr__nfet_01v8[vdsat] @m.xm15.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm16.msky130_fd_pr__nfet_01v8[vgs] @m.xm16.msky130_fd_pr__nfet_01v8[vds] @m.xm16.msky130_fd_pr__nfet_01v8[vdsat] @m.xm16.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm17.msky130_fd_pr__nfet_01v8[vgs] @m.xm17.msky130_fd_pr__nfet_01v8[vds] @m.xm17.msky130_fd_pr__nfet_01v8[vdsat] @m.xm17.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm18.msky130_fd_pr__nfet_01v8[vgs] @m.xm18.msky130_fd_pr__nfet_01v8[vds] @m.xm18.msky130_fd_pr__nfet_01v8[vdsat] @m.xm18.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm19.msky130_fd_pr__nfet_01v8[vgs] @m.xm19.msky130_fd_pr__nfet_01v8[vds] @m.xm19.msky130_fd_pr__nfet_01v8[vdsat] @m.xm19.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm20.msky130_fd_pr__nfet_01v8[vgs] @m.xm20.msky130_fd_pr__nfet_01v8[vds] @m.xm20.msky130_fd_pr__nfet_01v8[vdsat] @m.xm20.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm21.msky130_fd_pr__nfet_01v8[vgs] @m.xm21.msky130_fd_pr__nfet_01v8[vds] @m.xm21.msky130_fd_pr__nfet_01v8[vdsat] @m.xm21.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm22.msky130_fd_pr__nfet_01v8[vgs] @m.xm22.msky130_fd_pr__nfet_01v8[vds] @m.xm22.msky130_fd_pr__nfet_01v8[vdsat] @m.xm22.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm23.msky130_fd_pr__nfet_01v8[vgs] @m.xm23.msky130_fd_pr__nfet_01v8[vds] @m.xm23.msky130_fd_pr__nfet_01v8[vdsat] @m.xm23.msky130_fd_pr__nfet_01v8[vth]
* MOSFET small-signal parameters (PMOS)
.save @m.xm0.msky130_fd_pr__pfet_01v8[gm] @m.xm0.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm1.msky130_fd_pr__pfet_01v8[gm] @m.xm1.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm2.msky130_fd_pr__pfet_01v8[gm] @m.xm2.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm3.msky130_fd_pr__pfet_01v8[gm] @m.xm3.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm4.msky130_fd_pr__pfet_01v8[gm] @m.xm4.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm5.msky130_fd_pr__pfet_01v8[gm] @m.xm5.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm6.msky130_fd_pr__pfet_01v8[gm] @m.xm6.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm7.msky130_fd_pr__pfet_01v8[gm] @m.xm7.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm8.msky130_fd_pr__pfet_01v8[gm] @m.xm8.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm9.msky130_fd_pr__pfet_01v8[gm] @m.xm9.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm10.msky130_fd_pr__pfet_01v8[gm] @m.xm10.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm11.msky130_fd_pr__pfet_01v8[gm] @m.xm11.msky130_fd_pr__pfet_01v8[gds]
* MOSFET small-signal parameters (NMOS)
.save @m.xm12.msky130_fd_pr__nfet_01v8[gm] @m.xm12.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm13.msky130_fd_pr__nfet_01v8[gm] @m.xm13.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm14.msky130_fd_pr__nfet_01v8[gm] @m.xm14.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm15.msky130_fd_pr__nfet_01v8[gm] @m.xm15.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm16.msky130_fd_pr__nfet_01v8[gm] @m.xm16.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm17.msky130_fd_pr__nfet_01v8[gm] @m.xm17.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm18.msky130_fd_pr__nfet_01v8[gm] @m.xm18.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm19.msky130_fd_pr__nfet_01v8[gm] @m.xm19.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm20.msky130_fd_pr__nfet_01v8[gm] @m.xm20.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm21.msky130_fd_pr__nfet_01v8[gm] @m.xm21.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm22.msky130_fd_pr__nfet_01v8[gm] @m.xm22.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm23.msky130_fd_pr__nfet_01v8[gm] @m.xm23.msky130_fd_pr__nfet_01v8[gds]
* Resistor and source currents
.save @rin[i]
.save @rfb[i]
.save @i0[i]

* =========================
* ANALYSIS
* =========================

.op
.ac dec 100 1 10G

.end
