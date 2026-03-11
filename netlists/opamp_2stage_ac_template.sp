* Parametric 2-Stage Operational Amplifier Template - AC Analysis
* Same circuit as opamp_2stage_template.sp but with:
*   - Middlebrook loop-breaking network at inverting input (vp)
*   - AC analysis directive for open-loop gain measurement
*   - gm/gds .save directives for small-signal parameter extraction
*
* Loop-breaking: vp is split into vp_fb (feedback side: Rin, Rfb)
*   and vp_gate (amplifier side: M1 gate). At DC they are shorted
*   via Lbreak (1T inductor). At AC the loop is broken and Vac_inj
*   injects through Cac_inj (1T capacitor).
*   Loop gain: T(f) = -V(vp_fb) / V(vp_gate)
*
* Parameters: {W_DIFF}, {L_DIFF} (M1,M2), {W_MIRROR}, {L_MIRROR} (M3,M4),
*             {W_M5}, {W_M7}, {W_M8}, {L_BIAS} (M5,M7,M8 share L for mirror matching),
*             {W_M6}, {L_M6},
*             {C_C}, {R_Z}, {R_IN}, {R_F}, {I_REF}, {VIN_P}, {VIN_N}

.title 2-Stage Operational Amplifier - AC Analysis

* Include SKY130 models (direct include of corner files for NgSpiceShared compatibility)
.param mc_mm_switch=0

* Define missing mismatch parameters to avoid "undefined parameter" errors
* These are statistical mismatch parameters for Monte Carlo analysis - set to 0 for nominal sim
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

* Define diff parameters (process variation parameters)
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

* Power supplies (parametric for dataset generation)
Vdd vdd 0 DC {VDD}
Vss vss 0 DC 0

* Input configuration (closed-loop inverting amplifier)
* MODIFIED: vp split into vp_fb (feedback) and vp_gate (M1 gate)
* for Middlebrook loop gain measurement
Vcm_ref  vn    0    DC {VIN_P}
Vsig     vsig  0    DC {VIN_N}
Rin      vsig  vp_fb   {R_IN}
Rfb      vout  vp_fb   {R_F}

* =========================
* LOOP-BREAKING NETWORK
* =========================
* Lbreak: DC short (preserves bias), AC open (breaks loop)
* Cac_inj + Vac_inj: DC open (no bias effect), AC short (injects signal)
Lbreak  vp_fb  vp_gate  1e12
Vac_inj vac_node 0 DC 0 AC 1
Cac_inj vac_node vp_gate 1e12

* =========================
* STAGE 1: Diff Pair + Load
* =========================

* M1, M2: NMOS differential pair
* MODIFIED: M1 gate connects to vp_gate (amplifier side of loop break)
Xm1 vd1         vp_gate  tail vss sky130_fd_pr__nfet_01v8 W={W_DIFF} L={L_DIFF}
Xm2 vout_stage1 vn       tail vss sky130_fd_pr__nfet_01v8 W={W_DIFF} L={L_DIFF}

* M3, M4: PMOS active load (current mirror)
Xm3 vd1         vd1 vdd  vdd sky130_fd_pr__pfet_01v8 W={W_MIRROR} L={L_MIRROR}   ; diode-connected
Xm4 vout_stage1 vd1 vdd  vdd sky130_fd_pr__pfet_01v8 W={W_MIRROR} L={L_MIRROR}   ; mirror leg

* =========================
* BOTTOM NMOS BIAS MIRROR
* =========================

* Reference current source
Iref vdd nbias DC {I_REF}

* M8: diode-connected NMOS (bias reference)
Xm8 nbias nbias vss vss sky130_fd_pr__nfet_01v8 W={W_M8} L={L_BIAS}

* M5: tail current source for the diff pair (mirror leg)
Xm5 tail nbias vss vss sky130_fd_pr__nfet_01v8 W={W_M5} L={L_BIAS}

* M7: bottom device of second stage (mirror leg)
Xm7 vout nbias vss vss sky130_fd_pr__nfet_01v8 W={W_M7} L={L_BIAS}

* =========================
* STAGE 2: Output Stage
* =========================

* M6: PMOS common-source gain transistor (top right device)
Xm6 vout vg2 vdd vdd sky130_fd_pr__pfet_01v8 W={W_M6} L={L_M6}

* =========================
* COMPENSATION NETWORK
* =========================

* Series RC Miller compensation
Cc  vout_stage1 vc  {C_C}
Rz  vc          vg2 {R_Z}

* DC bias resistor for M6 gate
Rbias_g vout_stage1 vg2 10G

* =========================
* OUTPUT LOAD / PROBE
* =========================

Cload vout 0 10p

* =========================
* SAVE DEVICE CURRENTS
* =========================

.save all
* MOSFET drain currents
.save @m.xm1.msky130_fd_pr__nfet_01v8[id]
.save @m.xm2.msky130_fd_pr__nfet_01v8[id]
.save @m.xm3.msky130_fd_pr__pfet_01v8[id]
.save @m.xm4.msky130_fd_pr__pfet_01v8[id]
.save @m.xm5.msky130_fd_pr__nfet_01v8[id]
.save @m.xm6.msky130_fd_pr__pfet_01v8[id]
.save @m.xm7.msky130_fd_pr__nfet_01v8[id]
.save @m.xm8.msky130_fd_pr__nfet_01v8[id]
* MOSFET operating point parameters (for region classification)
.save @m.xm1.msky130_fd_pr__nfet_01v8[vgs] @m.xm1.msky130_fd_pr__nfet_01v8[vds] @m.xm1.msky130_fd_pr__nfet_01v8[vdsat] @m.xm1.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm2.msky130_fd_pr__nfet_01v8[vgs] @m.xm2.msky130_fd_pr__nfet_01v8[vds] @m.xm2.msky130_fd_pr__nfet_01v8[vdsat] @m.xm2.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm3.msky130_fd_pr__pfet_01v8[vgs] @m.xm3.msky130_fd_pr__pfet_01v8[vds] @m.xm3.msky130_fd_pr__pfet_01v8[vdsat] @m.xm3.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm4.msky130_fd_pr__pfet_01v8[vgs] @m.xm4.msky130_fd_pr__pfet_01v8[vds] @m.xm4.msky130_fd_pr__pfet_01v8[vdsat] @m.xm4.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm5.msky130_fd_pr__nfet_01v8[vgs] @m.xm5.msky130_fd_pr__nfet_01v8[vds] @m.xm5.msky130_fd_pr__nfet_01v8[vdsat] @m.xm5.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm6.msky130_fd_pr__pfet_01v8[vgs] @m.xm6.msky130_fd_pr__pfet_01v8[vds] @m.xm6.msky130_fd_pr__pfet_01v8[vdsat] @m.xm6.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm7.msky130_fd_pr__nfet_01v8[vgs] @m.xm7.msky130_fd_pr__nfet_01v8[vds] @m.xm7.msky130_fd_pr__nfet_01v8[vdsat] @m.xm7.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm8.msky130_fd_pr__nfet_01v8[vgs] @m.xm8.msky130_fd_pr__nfet_01v8[vds] @m.xm8.msky130_fd_pr__nfet_01v8[vdsat] @m.xm8.msky130_fd_pr__nfet_01v8[vth]
* MOSFET small-signal parameters (gm, gds for analytical AC computation)
.save @m.xm1.msky130_fd_pr__nfet_01v8[gm] @m.xm1.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm2.msky130_fd_pr__nfet_01v8[gm] @m.xm2.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm3.msky130_fd_pr__pfet_01v8[gm] @m.xm3.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm4.msky130_fd_pr__pfet_01v8[gm] @m.xm4.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm5.msky130_fd_pr__nfet_01v8[gm] @m.xm5.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm6.msky130_fd_pr__pfet_01v8[gm] @m.xm6.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm7.msky130_fd_pr__nfet_01v8[gm] @m.xm7.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm8.msky130_fd_pr__nfet_01v8[gm] @m.xm8.msky130_fd_pr__nfet_01v8[gds]
* Resistor currents
.save @rin[i]
.save @rfb[i]
.save @rz[i]
.save @iref[i]
* Compensation network resistor
.save @rbias_g[i]
* Capacitor currents (0 DC but needed for KCL completeness)
.save @cc[i]
.save @cload[i]
* Voltage source branch currents (for supply net KCL)
.save vdd#branch
.save vss#branch
.save vcm_ref#branch
.save vsig#branch

* =========================
* ANALYSIS
* =========================

.op
.ac dec 100 1 10G

.end
