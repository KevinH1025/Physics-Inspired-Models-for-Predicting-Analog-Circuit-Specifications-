"""Re-evaluate saved checkpoints using validate() from training loop.
Loads best model, runs validate(), then runs DC gain eval — same as train_v3.py end section."""
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
    cfg = load_config(os.path.join(exp, 'original_config.yaml'))
    parsed = parse_training_config(cfg)
    ns = argparse.Namespace(**parsed)
    dp = Path(cfg['data']['path'])

    # Load & normalize (same as train_v3.py)
    tb = load_prebatched_variant(dp / 'train', 0)
    vb = load_prebatched_variant(dp / 'val', 0)
    vm, vs = compute_vdc_normalization(dp, tb, 'zscore', 1.8)
    cm, cs = compute_current_normalization([tb])
    # Preprocessing (same as train_v3.py lines 280-295)
    add_ss_node_targets(tb)
    add_ss_node_targets(vb)
    add_vth_node_targets(tb)
    add_vth_node_targets(vb)
    add_region_node_targets(tb)
    add_region_node_targets(vb)
    add_mosfet_gt_vov(tb)
    add_mosfet_gt_vov(vb)

    has_ss = any(hasattr(b, 'mosfet_drain_mask') and b.mosfet_drain_mask.any() for b in tb[:3])
    gm_m, gm_s, gds_m, gds_s = (0., 1., 0., 1.)
    if has_ss:
        gm_m, gm_s, gds_m, gds_s = compute_ss_normalization([tb])
    normalize_batches_vdc(tb, vm, vs)
    normalize_batches_vdc(vb, vm, vs)
    normalize_batches_current(tb, cm, cs)
    normalize_batches_current(vb, cm, cs)
    if has_ss:
        normalize_batches_ss(tb, gm_m, gm_s, gds_m, gds_s)
        normalize_batches_ss(vb, gm_m, gm_s, gds_m, gds_s)
    dcv = [x for b in tb if hasattr(b, 'ac_dc_gain') for x in b.ac_dc_gain.tolist()]
    dc_m = float(np.mean(dcv)) if dcv else 0.
    dc_s = max(float(np.std(dcv)), 1e-6) if dcv else 1.

    vl = PrebatchedLoader(vb, shuffle=False)
    s = tb[0]
    dim = s.x.shape[1] + s.type_tens.shape[1]

    # Recreate model from SAVED config (not current code defaults)
    # Force disable new features that didn't exist when model was trained
    if 'gm_id_head_config' in parsed:
        parsed['gm_id_head_config'] = {'enabled': False}
    if 'ss_head_config' in parsed:
        parsed['ss_head_config']['predict_gm_id'] = False

    model, mc = create_model_from_args(argparse.Namespace(**parsed), dim, a.device)
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(vm, vs, gm_m, gm_s, gds_m, gds_s, current_mean=cm, current_std=cs)

    mp = os.path.join(exp, 'best_model.pt')
    st = torch.load(mp, map_location=a.device, weights_only=False)
    sd = st['model_state_dict'] if isinstance(st, dict) and 'model_state_dict' in st else st
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        # Filter out known non-critical missing keys (buffers etc)
        critical = [k for k in missing if 'buf' not in k and '_cached' not in k]
        if critical:
            print(f"WARNING: Missing critical keys: {critical}", file=sys.stderr)

    def log(msg):
        print(msg); print(msg, file=sys.stderr)

    log(f"\n=== Evaluation: {os.path.basename(exp)} ===")

    # Run validate() — exact same as training
    pc = getattr(ns, 'predict_currents', True)
    r = validate(model, vl, a.device, vm, vs, cm, cs,
                 predict_currents=pc,
                 current_weight=getattr(ns, 'current_weight', 1.0),
                 ss_gm_mean=gm_m, ss_gm_std=gm_s, ss_gds_mean=gds_m, ss_gds_std=gds_s,
                 ss_gm_loss_weight=getattr(ns, 'ss_gm_loss_weight', 0.),
                 ss_gds_loss_weight=getattr(ns, 'ss_gds_loss_weight', 0.),
                 dc_gain_loss_weight=getattr(ns, 'dc_gain_loss_weight', 0.),
                 dc_gain_mean=dc_m, dc_gain_std=dc_s)

    # Unpack validate results
    val_loss, mae_mv, v_loss, c_loss, c_mae = r[0], r[1], r[2], r[3], r[4]
    acc80, acc50, acc20, acc10 = r[5], r[6], r[7], r[8]
    c50, c20, c10, c5 = r[9], r[10], r[11], r[12]
    rel = r[28]  # rel_metrics dict
    ss_m = rel.get('ss_metrics') if rel else None

    log(f"Voltage MAE: {mae_mv:.2f} mV")
    log(f"Current MAE: {c_mae:.1f} µA")
    log(f"V Acc @80={acc80:.1f}% @50={acc50:.1f}% @20={acc20:.1f}% @10={acc10:.1f}%")
    log(f"I Acc @50={c50:.1f}% @20={c20:.1f}% @5={c10:.1f}% @2={c5:.1f}%")
    if ss_m:
        log(f"gm  MAE: {ss_m['gm_log_mae']:.3f} log10 (median rel: {ss_m['gm_median']:.1f}%)")
        log(f"gds MAE: {ss_m['gds_log_mae']:.3f} log10 (median rel: {ss_m['gds_median']:.1f}%)")
        log(f"gm @10%: {ss_m['gm_acc'][10]:.1f}% | gds @10%: {ss_m['gds_acc'][10]:.1f}%")

    # DC gain eval (same as train_v3.py end section)
    dcw = getattr(ns, 'dc_gain_loss_weight', 0.)
    if dcw > 0:
        errs = []
        pv, gv = [], []
        model.eval()
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(a.device)
                od = model(batch)
                dp_ = od.get('dc_gain_pred')
                if dp_ is None:
                    log("DC gain pred is None"); break
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
