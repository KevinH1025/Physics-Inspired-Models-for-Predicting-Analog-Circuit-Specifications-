* Parametric 3-Stage Fan SMC Operational Amplifier Template for Dataset Generation
* Topology: Folded-cascode input + common-source gain + push-pull output
*
* Stage 1: PMOS diff pair (M8, M9) with folded-cascode NMOS load
*          Tail current: M4 (4x bias mirror)
*          Cascode: M15/M19 (left), M16/M20 (right)
*          Output: net050 (also drives Stage 2 & 3 PMOS gates)
*
* Stage 2: PMOS common-source M10 (gate=net050)
*          NMOS active load: M21 (diode), M22 (mirror)
*          Output: net049 (drives Stage 3 NMOS gate)
*
* Stage 3: Push-pull output
*          PMOS M11 (gate=net050) + NMOS M23 (gate=net049)
*          Output: vout
*
* Bias: I0 -> M0 (diode PMOS) -> mirrors M1-M7
*       M14 (diode NMOS, VB3) + M12 cascode -> VB4
*
* Compensation: C0 Miller cap (net050 -> vout)
*
* Parameters (group-based):
*   BIASCM_P: {W_BIASCM_P}, {L_BIASCM_P}, {M_BIASCM_P}  (M0-M7, PMOS bias)
*   GM1:      {W_GM1}, {L_GM1}, {M_GM1}                    (M8-M9, diff pair)
*   GM2:      {W_GM2}, {L_GM2}, {M_GM2}                    (M10, stage 2)
*   GMF2:     {W_GMF2}, {L_GMF2}, {M_GMF2}                 (M11, stage 3 PMOS)
*   BIASCM_N: {W_BIASCM_N}, {L_BIASCM_N}, {M_BIASCM_N}    (M12-M20, NMOS bias/cascode)
*   LOAD2:    {W_LOAD2}, {L_LOAD2}, {M_LOAD2}              (M21-M22, stage 2 load)
*   GM3:      {W_GM3}, {L_GM3}, {M_GM3}                    (M23, stage 3 NMOS)
*   {I_BIAS}, {C_COMP}, {R_IN}, {R_F}, {VDD}, {VIN_P}, {VIN_N}

.title 3-Stage Fan SMC Operational Amplifier - Parametric

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
* vinp = non-inverting input (M9 gate), held at VCM
* vinn = inverting input (M8 gate), receives signal + feedback
Vcm_ref  vinp   0    DC {VIN_P}
Vsig     vsig   0    DC {VIN_N}
Rin      vsig   vinn {R_IN}
Rfb      vout   vinn {R_F}

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

* M8: inverting input (VINN)
Xm8 dm_2 vinn net31 net31 sky130_fd_pr__pfet_01v8 W={W_GM1} L={L_GM1} M={M_GM1}

* M9: non-inverting input (VINP)
Xm9 net063 vinp net31 net31 sky130_fd_pr__pfet_01v8 W={W_GM1} L={L_GM1} M={M_GM1}

* =========================
* STAGE 1: NMOS CASCODE LOAD
* =========================

* Bottom bias devices (VB4 gate)
* M19: left branch bottom (8x)
Xm19 dm_2 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* M20: right branch bottom (8x)
Xm20 net063 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* Cascode devices (VB3 gate)
* M15: left branch cascode (4x) -> VOUTN
Xm15 voutn vb3 dm_2 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* M16: right branch cascode (4x) -> net050 (stage 1 output)
Xm16 net050 vb3 net063 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* =========================
* NMOS BIAS GENERATION
* =========================

* M14: diode-connected NMOS -> VB3
Xm14 vb3 vb3 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M={M_BIASCM_N}

* M17: bottom for VB4 cascode generation (4x)
Xm17 net54 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* M12: cascode for VB4 generation (4x)
Xm12 vb4 vb3 net54 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* M18: bottom for DM_1 cascode branch (4x)
Xm18 net56 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* M13: cascode for DM_1 branch (4x)
Xm13 dm_1 vb3 net56 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* =========================
* STAGE 2: COMMON-SOURCE
* =========================

* M10: PMOS gain stage (gate=net050)
Xm10 net043 net050 vdda vdda sky130_fd_pr__pfet_01v8 W={W_GM2} L={L_GM2} M={M_GM2}

* M21: NMOS diode-connected load
Xm21 net043 net043 gnda gnda sky130_fd_pr__nfet_01v8 W={W_LOAD2} L={L_LOAD2} M={M_LOAD2}

* M22: NMOS mirror load -> net049
Xm22 net049 net043 gnda gnda sky130_fd_pr__nfet_01v8 W={W_LOAD2} L={L_LOAD2} M={M_LOAD2}

* =========================
* STAGE 3: PUSH-PULL OUTPUT
* =========================

* M11: PMOS output (gate=net050)
Xm11 vout net050 vdda vdda sky130_fd_pr__pfet_01v8 W={W_GMF2} L={L_GMF2} M={M_GMF2}

* M23: NMOS output (gate=net049)
Xm23 vout net049 gnda gnda sky130_fd_pr__nfet_01v8 W={W_GM3} L={L_GM3} M={M_GM3}

* =========================
* BIAS CURRENT SOURCE
* =========================

I0 nbias gnda DC {I_BIAS}

* =========================
* COMPENSATION NETWORK
* =========================

* Miller capacitor between stage 1 output and final output
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
* Resistor and source currents
.save @rin[i]
.save @rfb[i]
.save @i0[i]

* =========================
* ANALYSIS
* =========================

.op

.end
