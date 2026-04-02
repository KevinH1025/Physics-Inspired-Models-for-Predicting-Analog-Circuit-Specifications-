"""Load a saved checkpoint and run full final evaluation (same as train_v3.py end section).
Uses original_config.yaml from the experiment folder to match the model exactly."""
import sys, os, torch, numpy as np, argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.config import load_config, parse_training_config
from src.training.checkpoint import create_model_from_args
from src.training.data_loading import (
    PrebatchedLoader, load_prebatched_variant,
    compute_vdc_normalization, compute_current_normalization, compute_ss_normalization,
    normalize_batches_vdc, normalize_batches_current, normalize_batches_ss,
    add_ss_node_targets, add_vth_node_targets, add_region_node_targets, add_mosfet_gt_vov,
)
from src.training.loops import validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir')
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = parser.parse_args()

    exp = a.experiment_dir
    cfg_path = os.path.join(exp, 'original_config.yaml')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(exp, 'config.yaml')
    cfg = load_config(cfg_path)
    parsed = parse_training_config(cfg)

    # Force disable new features that may not exist in checkpoint
    if 'gm_id_head_config' in parsed:
        parsed['gm_id_head_config'] = {'enabled': False}
    if 'ss_head_config' in parsed and parsed['ss_head_config'].get('predict_gm_id', False):
        pass  # keep if original config had it

    ns = argparse.Namespace(**parsed)
    dp = Path(cfg['data']['path'])

    # Load & preprocess (same as train_v3.py)
    tb = load_prebatched_variant(dp / 'train', 0)
    vb = load_prebatched_variant(dp / 'val', 0)
    add_ss_node_targets(tb); add_ss_node_targets(vb)
    add_vth_node_targets(tb); add_vth_node_targets(vb)
    add_region_node_targets(tb); add_region_node_targets(vb)
    add_mosfet_gt_vov(tb); add_mosfet_gt_vov(vb)

    vm, vs = compute_vdc_normalization(dp, tb, 'zscore', 1.8)
    cm, cs = compute_current_normalization([tb])
    has_ss = any(hasattr(b, 'mosfet_drain_mask') and b.mosfet_drain_mask.any() for b in tb[:3])
    gm_m, gm_s, gds_m, gds_s = 0., 1., 0., 1.
    if has_ss:
        gm_m, gm_s, gds_m, gds_s = compute_ss_normalization([tb])
    normalize_batches_vdc(tb, vm, vs); normalize_batches_vdc(vb, vm, vs)
    normalize_batches_current(tb, cm, cs); normalize_batches_current(vb, cm, cs)
    if has_ss:
        normalize_batches_ss(tb, gm_m, gm_s, gds_m, gds_s)
        normalize_batches_ss(vb, gm_m, gm_s, gds_m, gds_s)

    dcv = [x for b in tb if hasattr(b, 'ac_dc_gain') for x in b.ac_dc_gain.tolist()]
    dc_m = float(np.mean(dcv)) if dcv else 0.
    dc_s = max(float(np.std(dcv)), 1e-6) if dcv else 1.

    vl = PrebatchedLoader(vb, shuffle=False)
    dim = tb[0].x.shape[1] + tb[0].type_tens.shape[1]

    model, mc = create_model_from_args(ns, dim, a.device)
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(vm, vs, gm_m, gm_s, gds_m, gds_s, current_mean=cm, current_std=cs)

    mp = os.path.join(exp, 'best_model.pt')
    st = torch.load(mp, map_location=a.device, weights_only=False)
    sd = st['model_state_dict'] if isinstance(st, dict) and 'model_state_dict' in st else st
    missing, unexpected = model.load_state_dict(sd, strict=False)
    critical_missing = [k for k in missing if 'buf' not in k and '_cached' not in k]
    if critical_missing:
        print(f"WARNING: Missing critical keys: {critical_missing}", file=sys.stderr)

    def log(msg):
        print(msg); print(msg, file=sys.stderr)

    log(f"\n=== Evaluation: {os.path.basename(exp)} ===")

    # Run validate()
    pc = getattr(ns, 'predict_currents', True)
    dcw = getattr(ns, 'dc_gain_loss_weight', 0.)
    r = validate(model, vl, a.device, vm, vs, cm, cs,
                 predict_currents=pc,
                 current_weight=getattr(ns, 'current_weight', 1.0),
                 ss_gm_mean=gm_m, ss_gm_std=gm_s, ss_gds_mean=gds_m, ss_gds_std=gds_s,
                 ss_gm_loss_weight=getattr(ns, 'ss_gm_loss_weight', 0.),
                 ss_gds_loss_weight=getattr(ns, 'ss_gds_loss_weight', 0.),
                 dc_gain_loss_weight=dcw, dc_gain_mean=dc_m, dc_gain_std=dc_s)

    # Unpack (same order as validate() return)
    mae_mv, c_mae = r[1], r[4]
    acc80, acc50, acc20, acc10 = r[5], r[6], r[7], r[8]
    rm = r[28]  # rel_metrics

    # Extract all metrics from rel_metrics (same as train_v3.py lines 1672-1728)
    ia = rm.get('i_abs_acc', {}) if rm else {}
    vr = rm.get('v_rel_acc', {}) if rm else {}
    ir = rm.get('i_rel_acc', {}) if rm else {}
    ss_m = rm.get('ss_metrics') if rm else None

    log(f"\n{'='*50}\n=== EVALUATION COMPLETE ===\n{'='*50}")
    if rm:
        log(f"Best Val MAE: {mae_mv:.2f}mV (median rel: {rm['v_rel_median']:.2f}%)")
    else:
        log(f"Best Val MAE: {mae_mv:.2f}mV")
    if pc:
        if rm:
            log(f"Best Val Current MAE: {c_mae:.1f}µA (median rel: {rm['i_rel_median']:.2f}%)")
        else:
            log(f"Best Val Current MAE: {c_mae:.1f}µA")

    if pc and ia:
        log(f"Voltage Abs Acc @80mV: {acc80:5.2f}% | Current Abs Acc @50uA: {ia[50]:5.2f}%")
        log(f"Voltage Abs Acc @50mV: {acc50:5.2f}% | Current Abs Acc @20uA: {ia[20]:5.2f}%")
        log(f"Voltage Abs Acc @20mV: {acc20:5.2f}% | Current Abs Acc  @5uA: {ia[5]:5.2f}%")
        log(f"Voltage Abs Acc @10mV: {acc10:5.2f}% | Current Abs Acc  @2uA: {ia[2]:5.2f}%")
    else:
        log(f"Voltage Abs Acc @80mV: {acc80:.2f}%")
        log(f"Voltage Abs Acc @50mV: {acc50:.2f}%")
        log(f"Voltage Abs Acc @20mV: {acc20:.2f}%")
        log(f"Voltage Abs Acc @10mV: {acc10:.2f}%")

    if vr:
        if pc:
            log(f"Voltage Rel Acc  @1%: {vr[1]:5.2f}% | Current Rel Acc  @1%: {ir[1]:5.2f}%")
            log(f"Voltage Rel Acc  @5%: {vr[5]:5.2f}% | Current Rel Acc  @5%: {ir[5]:5.2f}%")
            log(f"Voltage Rel Acc @10%: {vr[10]:5.2f}% | Current Rel Acc @10%: {ir[10]:5.2f}%")
            log(f"Voltage Rel Acc @20%: {vr[20]:5.2f}% | Current Rel Acc @20%: {ir[20]:5.2f}%")
        else:
            log(f"Voltage Rel Acc  @1%: {vr[1]:5.2f}%")
            log(f"Voltage Rel Acc  @5%: {vr[5]:5.2f}%")
            log(f"Voltage Rel Acc @10%: {vr[10]:5.2f}%")
            log(f"Voltage Rel Acc @20%: {vr[20]:5.2f}%")

    if ss_m is not None:
        log(f"\n--- SS Evaluation ---")
        log(f"gm  MAE: {ss_m['gm_log_mae']:.3f} log10  (median {ss_m['gm_log_median']:.3f}, median rel: {ss_m['gm_median']:.1f}%)")
        log(f"gds MAE: {ss_m['gds_log_mae']:.3f} log10  (median {ss_m['gds_log_median']:.3f}, median rel: {ss_m['gds_median']:.1f}%)")
        log(f"gm  Acc @10%: {ss_m['gm_acc'][10]:5.1f}% | gds Acc @10%: {ss_m['gds_acc'][10]:5.1f}%")
        log(f"gm  Acc @20%: {ss_m['gm_acc'][20]:5.1f}% | gds Acc @20%: {ss_m['gds_acc'][20]:5.1f}%")
        log(f"gm  Acc @50%: {ss_m['gm_acc'][50]:5.1f}% | gds Acc @50%: {ss_m['gds_acc'][50]:5.1f}%")

    gm_id_m = rm.get('gm_id_metrics') if rm else None
    if gm_id_m is not None:
        log(f"\n--- gm/Id Evaluation ---")
        log(f"gm/Id MAE: {gm_id_m['log_mae']:.3f} log10  (median {gm_id_m['log_median']:.3f}, median rel: {gm_id_m['median_rel']:.1f}%)")
        log(f"gm/Id Acc @10%: {gm_id_m['acc'][10]:5.1f}%")
        log(f"gm/Id Acc @20%: {gm_id_m['acc'][20]:5.1f}%")
        log(f"gm/Id Acc @50%: {gm_id_m['acc'][50]:5.1f}%")

    # DC gain eval (same as train_v3.py lines 1808-1850)
    if dcw > 0:
        errs, pv, gv = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(a.device)
                od = model(batch)
                dp_ = od.get('dc_gain_pred')
                if dp_ is None: log("DC gain pred is None"); break
                dt = batch.ac_dc_gain.to(a.device)
                pdb = dp_ * dc_s + dc_m
                errs.extend((pdb - dt).abs().cpu().tolist())
                pv.extend((pdb > 0).cpu().tolist())
                gv.extend((dt > 0).cpu().tolist())
        if errs:
            e = np.array(errs)
            acc = 100. * (np.array(pv) == np.array(gv)).mean()
            dc_lines = [
                f"\n--- DC Gain Evaluation (best model, val set) ---",
                f"  MAE: {e.mean():.2f} dB  (median {np.median(e):.2f})",
                f"  within  3dB:  {100*np.mean(e < 3):.1f}%",
                f"  within  5dB:  {100*np.mean(e < 5):.1f}%",
                f"  within 10dB:  {100*np.mean(e < 10):.1f}%",
                f"  Validity classifier (dc_gain>0): {acc:.1f}% accuracy",
            ]
            for l in dc_lines:
                log(l)
            if a.append:
                with open(os.path.join(exp, 'training.log'), 'a') as f:
                    for l in dc_lines:
                        f.write(l + '\n')
                log(f"Appended to training.log")

if __name__ == '__main__':
    main()
