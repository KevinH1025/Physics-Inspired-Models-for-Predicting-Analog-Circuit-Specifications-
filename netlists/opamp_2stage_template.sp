* Parametric 2-Stage Operational Amplifier Template for Dataset Generation
* Parameters: W and L for all transistors, compensation capacitor
*
* Topology:
* - Stage 1: Differential pair (M1, M2) with active load (M3, M4) and tail current (M5)
* - Stage 2: Common-source amplifier (M6) with active load (M7)
* - Bias: M8 (NMOS), M9 (PMOS)
* - Compensation: Miller capacitor Cc
*
* Parameters: {W_DIFF}, {L_DIFF} (M1,M2), {W_MIRROR}, {L_MIRROR} (M3,M4),
*             {W_M5}, {W_M7}, {W_M8}, {L_BIAS} (M5,M7,M8 share L for mirror matching),
*             {W_M6}, {L_M6},
*             {C_C}, {R_Z}, {R_IN}, {R_F}, {I_REF}, {VIN_P}, {VIN_N}

.title 2-Stage Operational Amplifier - Parametric

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
* - Note: In this op-amp topology, vp (M1 gate) is electrically the inverting input
*   because vout decreases when vp > vn. This is due to the current mirror + PMOS output.
* - For the inverting amplifier topology:
*   - Input signal VIN_N drives through R_IN to the inverting terminal (vp)
*   - Feedback from vout through R_F to the inverting terminal (vp)
*   - Non-inverting reference VIN_P held at vn
* - Gain = -R_F / R_IN
Vcm_ref  vn    0    DC {VIN_P}
Vsig     vsig  0    DC {VIN_N}
Rin      vsig  vp   {R_IN}
Rfb      vout  vp   {R_F}

* =========================
* STAGE 1: Diff Pair + Load
* =========================

* M1, M2: NMOS differential pair
* Left drain -> diode-connected PMOS M3 (node vd1)
* Right drain -> first-stage output (vout_stage1)
Xm1 vd1         vp  tail vss sky130_fd_pr__nfet_01v8 W={W_DIFF} L={L_DIFF}
Xm2 vout_stage1 vn  tail vss sky130_fd_pr__nfet_01v8 W={W_DIFF} L={L_DIFF}

* M3, M4: PMOS active load (current mirror)
Xm3 vd1         vd1 vdd  vdd sky130_fd_pr__pfet_01v8 W={W_MIRROR} L={L_MIRROR}   ; diode-connected
Xm4 vout_stage1 vd1 vdd  vdd sky130_fd_pr__pfet_01v8 W={W_MIRROR} L={L_MIRROR}   ; mirror leg

* =========================
* BOTTOM NMOS BIAS MIRROR
* =========================

* Reference current source (left bottom symbol in Fig. 5)
* Flows from vdd down into the reference NMOS M8
Iref vdd nbias DC {I_REF}

* M8: diode-connected NMOS (bias reference)
* Its Vgs defines the bias for the mirror
Xm8 nbias nbias vss vss sky130_fd_pr__nfet_01v8 W={W_M8} L={L_BIAS}

* M5: tail current source for the diff pair (mirror leg)
Xm5 tail nbias vss vss sky130_fd_pr__nfet_01v8 W={W_M5} L={L_BIAS}

* M7: bottom device of second stage (mirror leg)
Xm7 vout nbias vss vss sky130_fd_pr__nfet_01v8 W={W_M7} L={L_BIAS}

* =========================
* STAGE 2: Output Stage
* =========================

* Gate nodes:
*   - vout_stage1 directly drives bottom device gate (NMOS M7) via separate branch in the figure
*   - PMOS gate is driven through the RC Miller network (see below)

* M6: PMOS common-source gain transistor (top right device)
Xm6 vout vg2 vdd vdd sky130_fd_pr__pfet_01v8 W={W_M6} L={L_M6}

* (M7 already defined above as the NMOS sink of this stage)

* =========================
* COMPENSATION NETWORK
* =========================

* Series RC Miller compensation between first-stage output and M6 gate,
* matching the cap-plus-resistor drawn on the right of Fig. 5.
* vout_stage1 ---- Cc ---- node vc ---- Rz ---- node vg2 (M6 gate)
Cc  vout_stage1 vc  {C_C}
Rz  vc          vg2 {R_Z}

* DC bias resistor for M6 gate (provides DC path through the capacitor)
* Very high value (10G) doesn't affect AC behavior but sets DC operating point
Rbias_g vout_stage1 vg2 10G

* =========================
* OUTPUT LOAD / PROBE
* =========================

* Simple capacitive load at the output
Cload vout 0 10p

* (If you want to emulate the right-hand voltage source symbol as an output probe,
* you can add a small-signal source here, but for pure op-amp behavior it's fine
* to just leave vout as a node with Cload.)

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
* Resistor currents using @device[i] syntax (NgSpice requires lowercase!)
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

.end
