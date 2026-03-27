"""
Training and validation loop functions.
"""

import torch
import numpy as np

from src.training.losses import compute_loss, compute_combined_loss
from src.training.data_loading import get_prediction_mask
from src.training.metrics import (
    compute_voltage_accuracy,
    compute_current_accuracy,
    denormalize_voltage,
    denormalize_current,
)


def build_voltage_node_weights(
    batch,
    stage2_nodes: list = None,
    stage2_weight: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build per-node weights for voltage loss. Stage 2 nodes get extra weight.

    Supports mixed-topology batches by iterating per-graph.

    Args:
        batch: Batch with node_names and train_mask
        stage2_nodes: List of stage 2 node names to upweight
        stage2_weight: Weight multiplier for stage 2 nodes (1.0 = no extra weight)
        device: Device to create tensor on

    Returns:
        Tensor of weights matching train_mask True count, or None if disabled
    """
    if stage2_weight <= 1.0 or stage2_nodes is None:
        return None

    if not hasattr(batch, 'node_names') or batch.node_names is None:
        return None

    stage2_set = {n.lower() for n in stage2_nodes}
    ptr = batch.ptr
    num_graphs = len(ptr) - 1

    weights = []
    for g in range(num_graphs):
        start, end = ptr[g].item(), ptr[g + 1].item()
        mask_g = batch.train_mask[start:end]
        names_g = batch.node_names[g]
        for j, name in enumerate(names_g):
            if mask_g[j]:
                w = stage2_weight if name.lower() in stage2_set else 1.0
                weights.append(w)

    return torch.tensor(weights, device=device, dtype=torch.float32)


def train_epoch(model, loader, optimizer, gradient_clip, device, scaler=None,
                predict_currents=False, current_weight=1.0, voltage_weight=1.0, kcl_weight=0.0,
                current_mean=0.0, current_std=1.0,
                loss_type='mse', huber_delta=1.0, kcl_min_current=1e-9,
                constraint_weight=0.0, amp_dtype=None,
                vdc_mean=0.0, vdc_std=1.0, lambda_n=0.05,
                stage2_nodes=None, stage2_weight=1.0,
                use_terminal_voltage_loss=False,
                gm_physics_loss_weight=0.0, gm_physics_min_vov=0.0, gm_physics_use_clm=False,
                gm_physics_use_gt_voltages=True,
                ss_gm_mean=0.0, ss_gm_std=1.0,
                ss_gds_mean=0.0, ss_gds_std=1.0,
                ac_loss_weight=0.0, ac_mean=None, ac_std=None, ac_components=None,
                ss_gm_loss_weight=0.0, ss_gds_loss_weight=0.0,
                ss_huber_delta=0.0,
                triode_physics_loss_weight=0.0, triode_physics_config=None,
                cutoff_physics_loss_weight=0.0, cutoff_physics_n_nmos=1.5, cutoff_physics_n_pmos=2.0,
                region_loss_weight=0.0, kcl_mode='logsumexp', kcl_exclusive=False,
                kcl_detach_backbone=False, kcl_violation_threshold=0.0,
                kcl_conservation=False, kcl_skip_two_term=False, kcl_only_two_term=False,
                kcl_huber_delta=0.0, kcl_gt_filter=0.1,
                device_consistency_weight=0.0,
                intermediate_v_weight=0.0,
                kcl_intermediate_weight=0.0,
                vov_loss_weight=0.0,
                vth_loss_weight=0.0,
                uncertainty_weights=None,
                ss_region_stats=None,
                iv_id_loss_weight=0.0,
                dc_gain_loss_weight=0.0, dc_gain_mean=0.0, dc_gain_std=1.0):
    """
    Train for one epoch.

    Args:
        model: GNN model to train
        loader: DataLoader or PrebatchedLoader yielding batches
        optimizer: PyTorch optimizer
        gradient_clip: Max gradient norm (0 to disable)
        device: Device to train on
        scaler: GradScaler for AMP (None to disable)
        predict_currents: Whether to predict currents
        current_weight: Weight for current loss term
        kcl_weight: Weight for KCL physics loss (0 to disable)
        current_mean: Mean for current denormalization (log scale)
        current_std: Std for current denormalization (log scale)
        loss_type: 'mse' or 'huber'
        huber_delta: Delta for Huber loss
        kcl_min_current: Minimum total current for KCL nets (filters noise)
        constraint_weight: Weight for physics constraint losses (0 to disable)
        amp_dtype: AMP dtype (torch.float16 or torch.bfloat16) or None to disable
        vdc_mean: Mean for voltage denormalization
        vdc_std: Std for voltage denormalization
        lambda_n: Channel length modulation parameter
        stage2_nodes: List of stage 2 node names to upweight (e.g., ['vout', 'vout_stage1', 'vg2', 'vc'])
        stage2_weight: Weight multiplier for stage 2 nodes (1.0 = disabled)
        ac_loss_weight: Weight for AC prediction loss (0 to disable)
        ac_mean: [N] mean for enabled AC components
        ac_std: [N] std for enabled AC components
        ac_components: List of enabled AC component names (e.g. ['ugbw', 'pm', 'am'])
        ss_gm_loss_weight: Weight for supervised gm loss (0 to disable)
        ss_gds_loss_weight: Weight for supervised gds loss (0 to disable)

    Returns:
        Tuple of (avg_loss, mae_norm, avg_voltage_loss, avg_current_loss, current_mae_ua,
                  avg_kcl_loss, avg_diff_pair_loss, avg_mirror_loss, avg_output_stage_loss,
                  avg_lambda_mirror_loss, avg_gm_loss, avg_ac_loss, avg_ss_gm_loss, avg_ss_gds_loss)
    """
    model.train()
    total_loss = 0
    total_voltage_loss = 0
    total_current_loss = 0
    total_mae = 0
    total_count = 0
    total_current_mae = 0
    total_current_count = 0
    total_kcl_loss = 0
    total_diff_pair_loss = 0
    total_mirror_loss = 0
    total_output_stage_loss = 0
    total_lambda_mirror_loss = 0
    total_gm_physics_loss = 0
    total_ac_loss = 0
    total_ac_component_losses = {}  # per-component: ugbw, pm, am
    total_ss_gm_loss = 0
    total_ss_gds_loss = 0
    total_triode_physics_loss = 0
    total_triode_eq1_loss = 0
    total_triode_eq2_loss = 0
    total_triode_eq3_loss = 0
    total_cutoff_physics_loss = 0
    total_region_loss = 0
    total_vov_loss = 0
    total_vth_loss = 0
    total_dc_gain_loss = 0
    max_grad_norm = 0.0
    sum_grad_norm = 0.0
    grad_norm_count = 0

    use_amp = amp_dtype is not None

    for batch in loader:
        needs_transfer = str(batch.x.device).split(':')[0] != str(device).split(':')[0]
        if needs_transfer:
            batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype if amp_dtype else torch.float16):
            out_dict = model(batch)
            out = out_dict['node_voltages']

            # Select voltage loss targets
            if use_terminal_voltage_loss:
                mask = batch.terminal_train_mask
                pred = out[mask]
                target = batch.terminal_vdc.flatten()
            else:
                mask = get_prediction_mask(batch)
                pred = out[mask]
                target = batch.vdc.flatten()

            if pred.shape != target.shape:
                raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

            out_currents = out_dict.get('node_currents') if predict_currents else None
            current_mask = batch.has_current_mask if predict_currents and out_currents is not None else None

            # Per-device current: use device_current_mask (one terminal per device) to avoid double-counting
            if current_mask is not None and 'device_current_mask' in out_dict:
                current_mask = current_mask & out_dict['device_current_mask']

            # For voltage-derived currents, use intersection of has_current_mask and voltage_derived_current_mask
            # This ensures we only compute loss on terminals with both targets and predictions
            if current_mask is not None and 'voltage_derived_current_mask' in out_dict:
                voltage_derived_mask = out_dict['voltage_derived_current_mask']
                current_mask = current_mask & voltage_derived_mask

            # Build voltage node weights for stage2 upweighting (only for net-node loss)
            voltage_node_weights = build_voltage_node_weights(
                batch, stage2_nodes, stage2_weight, device=device
            ) if not use_terminal_voltage_loss else None

            loss, voltage_loss, current_loss, kcl_loss, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss, gm_physics_loss, ac_loss, ss_gm_loss, ss_gds_loss, triode_physics_loss, triode_eq1_loss, triode_eq2_loss, triode_eq3_loss, region_loss, cutoff_physics_loss, kcl_intermediate_loss, vov_loss, vth_loss, iv_id_loss, ac_per_component, dc_gain_loss = compute_combined_loss(
                voltage_pred=pred,
                voltage_target=target,
                current_pred=out_currents,
                current_pred_raw=out_dict.get('node_currents_raw') if predict_currents else None,
                current_target=batch.node_current_targets if predict_currents else None,
                current_mask=current_mask,
                current_weight=current_weight,
                voltage_weight=voltage_weight,
                loss_type=loss_type,
                huber_delta=huber_delta,
                kcl_weight=kcl_weight,
                edge_index=batch.edge_index,
                num_terminals=batch.num_terminals,
                train_mask=batch.train_mask,
                ptr=batch.ptr,
                terminal_current_sign=batch.terminal_current_sign if hasattr(batch, 'terminal_current_sign') else None,
                current_mean=current_mean,
                current_std=current_std,
                kcl_include_mask=batch.kcl_include_mask if hasattr(batch, 'kcl_include_mask') else None,
                kcl_min_current=kcl_min_current,
                kcl_mode=kcl_mode,
                kcl_violation_threshold=kcl_violation_threshold,
                kcl_huber_delta=kcl_huber_delta,
                kcl_gt_filter=kcl_gt_filter,
                kcl_exclusive=kcl_exclusive,
                kcl_detach_backbone=kcl_detach_backbone,
                kcl_skip_two_term=kcl_skip_two_term,
                kcl_only_two_term=kcl_only_two_term,
                kcl_conservation=kcl_conservation,
                node_embeddings=out_dict.get('node_embeddings'),
                current_head=getattr(model, 'current_head', None),
                constraint_weight=constraint_weight,
                mosfet_info=batch.mosfet_info if hasattr(batch, 'mosfet_info') else None,
                terminal_features=batch.x,
                diff_pair_constraints=batch.diff_pair_constraints if hasattr(batch, 'diff_pair_constraints') else None,
                mirror_constraints=batch.mirror_constraints if hasattr(batch, 'mirror_constraints') else None,
                output_stage_constraints=batch.output_stage_constraints if hasattr(batch, 'output_stage_constraints') else None,
                lambda_mirror_constraints=None,
                node_names=None,
                vdc_mean=vdc_mean,
                vdc_std=vdc_std,
                lambda_n=lambda_n,
                full_voltage_pred=out_dict['node_voltages'],
                voltage_node_weights=voltage_node_weights,
                gm_physics_loss_weight=gm_physics_loss_weight,
                gm_physics_min_vov=gm_physics_min_vov,
                gm_physics_use_clm=gm_physics_use_clm,
                node_mosfet_vth=batch.node_mosfet_vth if hasattr(batch, 'node_mosfet_vth') else None,
                mosfet_region_labels=batch.mosfet_region_labels if hasattr(batch, 'mosfet_region_labels') else None,
                ss_gm_mean=ss_gm_mean,
                ss_gm_std=ss_gm_std,
                ss_gds_mean=ss_gds_mean,
                ss_gds_std=ss_gds_std,
                ac_loss_weight=ac_loss_weight,
                ac_pred=out_dict.get('ac_pred'),
                ac_ugbw=batch.ac_ugbw if hasattr(batch, 'ac_ugbw') else None,
                ac_pm=batch.ac_pm if hasattr(batch, 'ac_pm') else None,
                ac_am=batch.ac_am if hasattr(batch, 'ac_am') else None,
                ac_valid=batch.ac_valid & (batch.ac_dc_gain > 0) if hasattr(batch, 'ac_valid') and hasattr(batch, 'ac_dc_gain') else (batch.ac_valid if hasattr(batch, 'ac_valid') else None),
                ac_mean=ac_mean,
                ac_std=ac_std,
                ac_components=ac_components or out_dict.get('ac_components'),
                ss_gm_loss_weight=ss_gm_loss_weight,
                ss_gds_loss_weight=ss_gds_loss_weight,
                ss_gm_pred=out_dict.get('mosfet_gm_pred'),
                ss_gds_pred=out_dict.get('mosfet_gds_pred'),
                mosfet_gm=batch.mosfet_gm if hasattr(batch, 'mosfet_gm') else None,
                mosfet_gds=batch.mosfet_gds if hasattr(batch, 'mosfet_gds') else None,
                ss_huber_delta=ss_huber_delta,
                ss_region_stats=ss_region_stats,
                triode_physics_loss_weight=triode_physics_loss_weight,
                triode_physics_config=triode_physics_config,
                cutoff_physics_loss_weight=cutoff_physics_loss_weight,
                cutoff_physics_n_nmos=cutoff_physics_n_nmos,
                cutoff_physics_n_pmos=cutoff_physics_n_pmos,
                mosfet_gt_vov=batch.mosfet_gt_vov if hasattr(batch, 'mosfet_gt_vov') else None,
                mosfet_ptr=batch.mosfet_ptr if hasattr(batch, 'mosfet_ptr') else None,
                node_voltage_targets=(batch.node_voltage_targets if hasattr(batch, 'node_voltage_targets') else None) if gm_physics_use_gt_voltages else None,
                region_loss_weight=region_loss_weight,
                region_pred=out_dict.get('mosfet_region_pred'),
                node_region_labels=batch.node_region_labels if hasattr(batch, 'node_region_labels') else None,
                mosfet_drain_mask=batch.mosfet_drain_mask if hasattr(batch, 'mosfet_drain_mask') else None,
                region_logits=out_dict.get('mosfet_region_logits'),
                device_consistency_weight=device_consistency_weight,
                batch=batch,
                kcl_intermediate_weight=kcl_intermediate_weight,
                aux_node_currents=out_dict.get('aux_node_currents'),
                vov_loss_weight=vov_loss_weight,
                vov_pred=out_dict.get('mosfet_vov_pred'),
                mosfet_vov_valid=getattr(batch, 'mosfet_vov_valid', None),
                vth_loss_weight=vth_loss_weight,
                vth_pred=out_dict.get('mosfet_vth_pred'),
                mosfet_vth=batch.mosfet_vth if hasattr(batch, 'mosfet_vth') else None,
                uncertainty_weights=uncertainty_weights,
                iv_id_loss_weight=iv_id_loss_weight,
                iv_log_abs_id=out_dict.get('mosfet_iv_log_abs_id'),
                dc_gain_loss_weight=dc_gain_loss_weight,
                dc_gain_pred=out_dict.get('dc_gain_pred'),
                dc_gain_target=batch.ac_dc_gain if hasattr(batch, 'ac_dc_gain') else None,
                dc_gain_mean=dc_gain_mean,
                dc_gain_std=dc_gain_std,
            )

            # Intermediate voltage auxiliary loss
            if intermediate_v_weight > 0 and 'intermediate_voltages' in out_dict:
                int_v_pred = out_dict['intermediate_voltages'][mask]
                int_v_loss = torch.nn.functional.mse_loss(int_v_pred, target)
                loss = loss + intermediate_v_weight * int_v_loss

            # Track current loss and MAE
            if current_mask is not None and current_mask.any():
                current_batch_size = current_mask.sum().item()
                total_current_loss += current_loss.detach().float() * current_batch_size

                current_pred_masked = out_currents[current_mask]
                current_target_masked = batch.node_current_targets[current_mask]


                current_pred_orig, current_target_orig = denormalize_current(
                    current_pred_masked, current_target_masked, current_mean, current_std
                )
                total_current_mae += (current_pred_orig - current_target_orig).abs().sum().detach() * 1e6
                total_current_count += current_batch_size

        if scaler is not None:
            # float16 AMP requires gradient scaling
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            # bfloat16 or no AMP - no scaling needed
            loss.backward()
            if gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            optimizer.step()
        max_grad_norm = max(max_grad_norm, grad_norm)
        sum_grad_norm += grad_norm
        grad_norm_count += 1

        batch_size = len(target)
        total_loss += loss.detach().float() * batch_size
        total_voltage_loss += voltage_loss.detach().float() * batch_size
        total_mae += (pred - target).abs().sum().detach()
        total_count += batch_size
        total_kcl_loss += kcl_loss.detach().float() * batch_size
        if constraint_weight > 0:
            total_diff_pair_loss += diff_pair_loss.detach().float() * batch_size
            total_mirror_loss += mirror_loss.detach().float() * batch_size
            total_output_stage_loss += output_stage_loss.detach().float() * batch_size
            total_lambda_mirror_loss += lambda_mirror_loss.detach().float() * batch_size
        total_gm_physics_loss += gm_physics_loss.detach().float() * batch_size
        if ac_loss_weight > 0:
            total_ac_loss += ac_loss.detach().float() * batch_size
            for comp, comp_loss in ac_per_component.items():
                total_ac_component_losses[comp] = total_ac_component_losses.get(comp, 0) + comp_loss.detach().float() * batch_size
        if ss_gm_loss_weight > 0:
            total_ss_gm_loss += ss_gm_loss.detach().float() * batch_size
        if ss_gds_loss_weight > 0:
            total_ss_gds_loss += ss_gds_loss.detach().float() * batch_size
        total_triode_physics_loss += triode_physics_loss.detach().float() * batch_size
        total_triode_eq1_loss += triode_eq1_loss.detach().float() * batch_size
        total_triode_eq2_loss += triode_eq2_loss.detach().float() * batch_size
        total_triode_eq3_loss += triode_eq3_loss.detach().float() * batch_size
        total_cutoff_physics_loss += cutoff_physics_loss.detach().float() * batch_size
        if region_loss_weight > 0:
            total_region_loss += region_loss.detach().float() * batch_size
        if vov_loss_weight > 0:
            total_vov_loss += vov_loss.detach().float() * batch_size
        if vth_loss_weight > 0:
            total_vth_loss += vth_loss.detach().float() * batch_size
        if dc_gain_loss_weight > 0:
            total_dc_gain_loss += dc_gain_loss.detach().float() * batch_size

    avg_loss = (total_loss / total_count).item()
    avg_voltage_loss = (total_voltage_loss / total_count).item()
    avg_current_loss = (total_current_loss / total_current_count).item() if total_current_count > 0 else 0.0
    mae_norm = (total_mae / total_count).item()
    current_mae_ua = (total_current_mae / total_current_count).item() if total_current_count > 0 else 0.0
    avg_kcl_loss = (total_kcl_loss / total_count).item()
    avg_diff_pair_loss = (total_diff_pair_loss / total_count).item() if constraint_weight > 0 else 0.0
    avg_mirror_loss = (total_mirror_loss / total_count).item() if constraint_weight > 0 else 0.0
    avg_output_stage_loss = (total_output_stage_loss / total_count).item() if constraint_weight > 0 else 0.0
    avg_lambda_mirror_loss = (total_lambda_mirror_loss / total_count).item() if constraint_weight > 0 else 0.0
    avg_gm_physics_loss = (total_gm_physics_loss / total_count).item()
    avg_ac_loss = (total_ac_loss / total_count).item() if ac_loss_weight > 0 else 0.0
    avg_ac_component_losses = {comp: (v / total_count).item() for comp, v in total_ac_component_losses.items()} if ac_loss_weight > 0 else {}
    avg_ss_gm_loss = (total_ss_gm_loss / total_count).item() if ss_gm_loss_weight > 0 else 0.0
    avg_ss_gds_loss = (total_ss_gds_loss / total_count).item() if ss_gds_loss_weight > 0 else 0.0
    avg_triode_physics_loss = (total_triode_physics_loss / total_count).item()
    avg_triode_eq1_loss = (total_triode_eq1_loss / total_count).item()
    avg_triode_eq2_loss = (total_triode_eq2_loss / total_count).item()
    avg_triode_eq3_loss = (total_triode_eq3_loss / total_count).item()
    avg_cutoff_physics_loss = (total_cutoff_physics_loss / total_count).item()
    avg_region_loss = (total_region_loss / total_count).item() if region_loss_weight > 0 else 0.0
    avg_vov_loss = (total_vov_loss / total_count).item() if vov_loss_weight > 0 else 0.0
    avg_vth_loss = (total_vth_loss / total_count).item() if vth_loss_weight > 0 else 0.0
    avg_dc_gain_loss = (total_dc_gain_loss / total_count).item() if dc_gain_loss_weight > 0 else 0.0
    avg_grad_norm = sum_grad_norm / max(grad_norm_count, 1)

    return avg_loss, mae_norm, avg_voltage_loss, avg_current_loss, current_mae_ua, avg_kcl_loss, avg_diff_pair_loss, avg_mirror_loss, avg_output_stage_loss, avg_lambda_mirror_loss, avg_gm_physics_loss, avg_ac_loss, avg_ss_gm_loss, avg_ss_gds_loss, avg_triode_physics_loss, avg_triode_eq1_loss, avg_triode_eq2_loss, avg_triode_eq3_loss, avg_cutoff_physics_loss, avg_region_loss, avg_vov_loss, avg_vth_loss, avg_ac_component_losses, avg_dc_gain_loss, max_grad_norm, avg_grad_norm


def validate(model, loader, device, vdc_mean, vdc_std, current_mean, current_std,
             predict_currents=False, current_weight=1.0, voltage_weight=1.0, kcl_weight=0.0,
             loss_type='mse', huber_delta=1.0, kcl_min_current=1e-9,
             constraint_weight=0.0, lambda_n=0.05,
             stage2_nodes=None, stage2_weight=1.0,
             use_terminal_voltage_loss=False,
             gm_physics_loss_weight=0.0, gm_physics_min_vov=0.0, gm_physics_use_clm=False,
             gm_physics_use_gt_voltages=True,
             ss_gm_mean=0.0, ss_gm_std=1.0,
             ss_gds_mean=0.0, ss_gds_std=1.0,
             ac_loss_weight=0.0, ac_mean=None, ac_std=None, ac_components=None,
             ss_gm_loss_weight=0.0, ss_gds_loss_weight=0.0,
             ss_huber_delta=0.0,
             triode_physics_loss_weight=0.0, triode_physics_config=None,
             cutoff_physics_loss_weight=0.0, cutoff_physics_n_nmos=1.5, cutoff_physics_n_pmos=2.0,
             region_loss_weight=0.0, kcl_mode='logsumexp', kcl_exclusive=False,
             kcl_detach_backbone=False, kcl_violation_threshold=0.0,
             kcl_conservation=False, kcl_skip_two_term=False, kcl_only_two_term=False,
             kcl_huber_delta=0.0, kcl_gt_filter=0.1, amp_dtype=None,
             device_consistency_weight=0.0,
             kcl_intermediate_weight=0.0,
             vov_loss_weight=0.0,
             vth_loss_weight=0.0,
             ss_region_stats=None,
             iv_id_loss_weight=0.0,
             dc_gain_loss_weight=0.0, dc_gain_mean=0.0, dc_gain_std=1.0):
    """
    Validate the model.

    Args:
        stage2_nodes: List of stage 2 node names to upweight (e.g., ['vout', 'vout_stage1', 'vg2', 'vc'])
        stage2_weight: Weight multiplier for stage 2 nodes (1.0 = disabled)

    Returns:
        Tuple of (avg_loss, mae_mv, avg_voltage_loss, avg_current_loss, current_mae_ua,
                  acc80, acc50, acc20, acc10, current_acc50, current_acc20, current_acc10, current_acc5,
                  avg_kcl_loss, avg_diff_pair_loss, avg_mirror_loss, avg_output_stage_loss,
                  avg_lambda_mirror_loss, avg_gm_physics_loss, avg_ac_loss, avg_ss_gm_loss, avg_ss_gds_loss,
                  avg_triode_physics_loss, avg_triode_eq1/2/3_loss, avg_cutoff_physics_loss, avg_region_loss)
    """
    model.eval()
    total_loss = 0
    total_voltage_loss = 0
    total_current_loss = 0
    total_kcl_loss = 0
    total_diff_pair_loss = 0
    total_mirror_loss = 0
    total_output_stage_loss = 0
    total_lambda_mirror_loss = 0
    total_gm_physics_loss = 0
    total_ac_loss = 0
    total_ac_component_losses = {}
    total_ss_gm_loss = 0
    total_ss_gds_loss = 0
    total_triode_physics_loss = 0
    total_triode_eq1_loss = 0
    total_triode_eq2_loss = 0
    total_triode_eq3_loss = 0
    total_cutoff_physics_loss = 0
    total_region_loss = 0
    total_vov_loss = 0
    total_vth_loss = 0
    total_dc_gain_loss = 0
    total_mae = 0
    total_count = 0
    total_current_mae = 0
    total_current_count = 0
    all_errors = []
    all_rel_errors = []
    all_current_errors = []
    all_current_preds = []
    all_current_targets = []
    all_i_rel_errors = []
    all_gm_rel = []
    all_gds_rel = []
    all_gm_log_errors = []
    all_gds_log_errors = []
    # MoE diagnostics
    moe_region_probs_sum = None  # running sum of predicted region probs [3]
    moe_region_correct = 0
    moe_region_total = 0
    moe_num_batches = 0
    moe_per_region_gm_errors = {0: [], 1: [], 2: []}  # cutoff, triode, saturation
    moe_per_region_gds_errors = {0: [], 1: [], 2: []}

    # Autograd SS needs autograd for torch.autograd.grad — cannot use inference_mode
    use_autograd = getattr(model, '_needs_autograd', False)

    for batch in loader:
        needs_transfer = str(batch.x.device).split(':')[0] != str(device).split(':')[0]
        if needs_transfer:
            batch = batch.to(device, non_blocking=True)

        if use_autograd:
            with torch.enable_grad():
                out_dict = model(batch)
        else:
            with torch.inference_mode():
                out_dict = model(batch)
        out = out_dict['node_voltages']

        # Select voltage loss targets
        if use_terminal_voltage_loss:
            mask = batch.terminal_train_mask
            pred = out[mask]
            target = batch.terminal_vdc.flatten()
        else:
            mask = get_prediction_mask(batch)
            pred = out[mask]
            target = batch.vdc.flatten()

        out_currents = out_dict.get('node_currents') if predict_currents else None
        current_mask = batch.has_current_mask if predict_currents and out_currents is not None else None

        # Per-device current: use device_current_mask (one terminal per device) to avoid double-counting
        if current_mask is not None and 'device_current_mask' in out_dict:
            current_mask = current_mask & out_dict['device_current_mask']

        # For voltage-derived currents, use intersection of has_current_mask and voltage_derived_current_mask
        if current_mask is not None and 'voltage_derived_current_mask' in out_dict:
            voltage_derived_mask = out_dict['voltage_derived_current_mask']
            current_mask = current_mask & voltage_derived_mask

        # Build voltage node weights for stage2 upweighting (only for net-node loss)
        voltage_node_weights = build_voltage_node_weights(
            batch, stage2_nodes, stage2_weight, device=device
        ) if not use_terminal_voltage_loss else None

        loss, voltage_loss, current_loss, kcl_loss, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss, gm_physics_loss, ac_loss, ss_gm_loss, ss_gds_loss, triode_physics_loss, triode_eq1_loss, triode_eq2_loss, triode_eq3_loss, region_loss, cutoff_physics_loss, kcl_intermediate_loss, vov_loss, vth_loss, iv_id_loss, ac_per_component, dc_gain_loss = compute_combined_loss(
            voltage_pred=pred,
            voltage_target=target,
            current_pred=out_currents,
            current_pred_raw=out_dict.get('node_currents_raw') if predict_currents else None,
            current_target=batch.node_current_targets if predict_currents else None,
            current_mask=current_mask,
            current_weight=current_weight,
            voltage_weight=voltage_weight,
            loss_type=loss_type,
            huber_delta=huber_delta,
            kcl_weight=kcl_weight,
            edge_index=batch.edge_index,
            num_terminals=batch.num_terminals,
            train_mask=batch.train_mask,
            ptr=batch.ptr,
            terminal_current_sign=batch.terminal_current_sign if hasattr(batch, 'terminal_current_sign') else None,
            current_mean=current_mean,
            current_std=current_std,
            kcl_include_mask=batch.kcl_include_mask if hasattr(batch, 'kcl_include_mask') else None,
            kcl_min_current=kcl_min_current,
            kcl_mode=kcl_mode,
            kcl_violation_threshold=kcl_violation_threshold,
            kcl_huber_delta=kcl_huber_delta,
            kcl_gt_filter=kcl_gt_filter,
            kcl_exclusive=kcl_exclusive,
            kcl_detach_backbone=kcl_detach_backbone,
            kcl_skip_two_term=kcl_skip_two_term,
            kcl_only_two_term=kcl_only_two_term,
            kcl_conservation=kcl_conservation,
            node_embeddings=out_dict.get('node_embeddings'),
            current_head=getattr(model, 'current_head', None),
            constraint_weight=constraint_weight,
            mosfet_info=batch.mosfet_info if hasattr(batch, 'mosfet_info') else None,
            terminal_features=batch.x,
            diff_pair_constraints=batch.diff_pair_constraints if hasattr(batch, 'diff_pair_constraints') else None,
            mirror_constraints=batch.mirror_constraints if hasattr(batch, 'mirror_constraints') else None,
            output_stage_constraints=batch.output_stage_constraints if hasattr(batch, 'output_stage_constraints') else None,
            lambda_mirror_constraints=None,
            node_names=None,
            vdc_mean=vdc_mean,
            vdc_std=vdc_std,
            lambda_n=lambda_n,
            full_voltage_pred=out_dict['node_voltages'],
            voltage_node_weights=voltage_node_weights,
            gm_physics_loss_weight=0.0,  # Don't include physics regularizer in val loss
            gm_physics_min_vov=gm_physics_min_vov,
            gm_physics_use_clm=gm_physics_use_clm,
            node_mosfet_vth=batch.node_mosfet_vth if hasattr(batch, 'node_mosfet_vth') else None,
            mosfet_region_labels=batch.mosfet_region_labels if hasattr(batch, 'mosfet_region_labels') else None,
            ss_gm_mean=ss_gm_mean,
            ss_gm_std=ss_gm_std,
            ss_gds_mean=ss_gds_mean,
            ss_gds_std=ss_gds_std,
            ac_loss_weight=ac_loss_weight,
            ac_pred=out_dict.get('ac_pred'),
            ac_ugbw=batch.ac_ugbw if hasattr(batch, 'ac_ugbw') else None,
            ac_pm=batch.ac_pm if hasattr(batch, 'ac_pm') else None,
            ac_am=batch.ac_am if hasattr(batch, 'ac_am') else None,
            ac_valid=batch.ac_valid & (batch.ac_dc_gain > 0) if hasattr(batch, 'ac_valid') and hasattr(batch, 'ac_dc_gain') else (batch.ac_valid if hasattr(batch, 'ac_valid') else None),
            ac_mean=ac_mean,
            ac_std=ac_std,
            ac_components=ac_components or out_dict.get('ac_components'),
            ss_gm_loss_weight=ss_gm_loss_weight,
            ss_gds_loss_weight=ss_gds_loss_weight,
            ss_gm_pred=out_dict.get('mosfet_gm_pred'),
            ss_gds_pred=out_dict.get('mosfet_gds_pred'),
            mosfet_gm=batch.mosfet_gm if hasattr(batch, 'mosfet_gm') else None,
            mosfet_gds=batch.mosfet_gds if hasattr(batch, 'mosfet_gds') else None,
            ss_huber_delta=ss_huber_delta,
            ss_region_stats=ss_region_stats,
            triode_physics_loss_weight=0.0,  # Don't include triode physics regularizer in val loss
            triode_physics_config=triode_physics_config,
            cutoff_physics_loss_weight=0.0,  # Don't include cutoff physics regularizer in val loss
            cutoff_physics_n_nmos=cutoff_physics_n_nmos,
            cutoff_physics_n_pmos=cutoff_physics_n_pmos,
            mosfet_gt_vov=batch.mosfet_gt_vov if hasattr(batch, 'mosfet_gt_vov') else None,
            mosfet_ptr=batch.mosfet_ptr if hasattr(batch, 'mosfet_ptr') else None,
            node_voltage_targets=(batch.node_voltage_targets if hasattr(batch, 'node_voltage_targets') else None) if gm_physics_use_gt_voltages else None,
            region_loss_weight=region_loss_weight,
            region_pred=out_dict.get('mosfet_region_pred'),
            node_region_labels=batch.node_region_labels if hasattr(batch, 'node_region_labels') else None,
            mosfet_drain_mask=batch.mosfet_drain_mask if hasattr(batch, 'mosfet_drain_mask') else None,
            region_logits=out_dict.get('mosfet_region_logits'),
            device_consistency_weight=device_consistency_weight,
            batch=batch,
            kcl_intermediate_weight=0.0,  # Don't include intermediate KCL in val loss
            aux_node_currents=out_dict.get('aux_node_currents'),
            vov_loss_weight=vov_loss_weight,
            vov_pred=out_dict.get('mosfet_vov_pred'),
            vth_loss_weight=vth_loss_weight,
            vth_pred=out_dict.get('mosfet_vth_pred'),
            mosfet_vth=batch.mosfet_vth if hasattr(batch, 'mosfet_vth') else None,
            iv_id_loss_weight=iv_id_loss_weight,
            iv_log_abs_id=out_dict.get('mosfet_iv_log_abs_id'),
            dc_gain_loss_weight=dc_gain_loss_weight,
            dc_gain_pred=out_dict.get('dc_gain_pred'),
            dc_gain_target=batch.ac_dc_gain if hasattr(batch, 'ac_dc_gain') else None,
            dc_gain_mean=dc_gain_mean,
            dc_gain_std=dc_gain_std,
        )

        # Track current loss and MAE
        if current_mask is not None and current_mask.any():
            current_batch_size = current_mask.sum().item()
            total_current_loss += current_loss.float() * current_batch_size

            current_pred_masked = out_currents[current_mask]
            current_target_masked = batch.node_current_targets[current_mask]
            current_pred_orig, current_target_orig = denormalize_current(
                current_pred_masked, current_target_masked, current_mean, current_std
            )
            current_errors_ua = (current_pred_orig - current_target_orig).abs() * 1e6
            total_current_mae += current_errors_ua.sum()
            total_current_count += current_batch_size
            all_current_errors.extend(current_errors_ua.cpu().tolist())
            all_current_preds.extend(current_pred_orig.cpu().tolist())
            all_current_targets.extend(current_target_orig.cpu().tolist())
            # Relative error for currents above 1nA floor
            above_floor = current_target_orig.abs() > 1e-9
            if above_floor.any():
                i_rel = (current_errors_ua[above_floor] * 1e-6) / current_target_orig[above_floor].abs() * 100
                all_i_rel_errors.extend(i_rel.cpu().tolist())

        batch_size = len(target)
        total_loss += loss.float() * batch_size
        total_voltage_loss += voltage_loss.float() * batch_size
        total_mae += (pred - target).abs().sum()
        total_count += batch_size
        total_kcl_loss += kcl_loss.float() * batch_size
        if constraint_weight > 0:
            total_diff_pair_loss += diff_pair_loss.float() * batch_size
            total_mirror_loss += mirror_loss.float() * batch_size
            total_output_stage_loss += output_stage_loss.float() * batch_size
            total_lambda_mirror_loss += lambda_mirror_loss.float() * batch_size
        total_gm_physics_loss += gm_physics_loss.float() * batch_size
        if ac_loss_weight > 0:
            total_ac_loss += ac_loss.float() * batch_size
            for comp, comp_loss in ac_per_component.items():
                total_ac_component_losses[comp] = total_ac_component_losses.get(comp, 0) + comp_loss.float() * batch_size
        if ss_gm_loss_weight > 0 or ss_gds_loss_weight > 0:
            total_ss_gm_loss += ss_gm_loss.float() * batch_size
            total_ss_gds_loss += ss_gds_loss.float() * batch_size
            # Collect SS relative errors for accuracy metrics (per-MOSFET predictions)
            gm_pred = out_dict.get('mosfet_gm_pred')
            gds_pred = out_dict.get('mosfet_gds_pred')
            mosfet_gm = getattr(batch, 'mosfet_gm', None)
            mosfet_gds = getattr(batch, 'mosfet_gds', None)
            if gm_pred is not None and mosfet_gm is not None:
                valid = mosfet_gm > 1e-12
                if valid.any():
                    # Denormalize predictions: per-region if available, else global
                    if ss_region_stats is not None and hasattr(batch, 'mosfet_region_labels'):
                        labels = batch.mosfet_region_labels.to(gm_pred.device)[valid]
                        gm_pred_log = torch.empty_like(gm_pred[valid])
                        gds_pred_log = torch.empty_like(gds_pred[valid])
                        # Default to global stats
                        gm_pred_log[:] = gm_pred[valid] * ss_gm_std + ss_gm_mean
                        gds_pred_log[:] = gds_pred[valid] * ss_gds_std + ss_gds_mean
                        for r in range(3):
                            rmask = (labels == r)
                            if rmask.any():
                                gm_pred_log[rmask] = gm_pred[valid][rmask] * ss_region_stats[r]['gm_std'] + ss_region_stats[r]['gm_mean']
                                gds_pred_log[rmask] = gds_pred[valid][rmask] * ss_region_stats[r]['gds_std'] + ss_region_stats[r]['gds_mean']
                    else:
                        gm_pred_log = gm_pred[valid] * ss_gm_std + ss_gm_mean
                        gds_pred_log = gds_pred[valid] * ss_gds_std + ss_gds_mean
                    gm_tgt_log = torch.log10(mosfet_gm[valid].to(gm_pred.device))
                    gds_tgt_log = torch.log10(mosfet_gds[valid].to(gds_pred.device).clamp(min=1e-20))
                    gm_rel = ((torch.pow(10, gm_pred_log) - torch.pow(10, gm_tgt_log)).abs() / torch.pow(10, gm_tgt_log).clamp(min=1e-15) * 100)
                    gds_rel = ((torch.pow(10, gds_pred_log) - torch.pow(10, gds_tgt_log)).abs() / torch.pow(10, gds_tgt_log).clamp(min=1e-15) * 100)
                    all_gm_rel.extend(gm_rel.cpu().tolist())
                    all_gds_rel.extend(gds_rel.cpu().tolist())
                    all_gm_log_errors.extend((gm_pred_log - gm_tgt_log).abs().cpu().tolist())
                    all_gds_log_errors.extend((gds_pred_log - gds_tgt_log).abs().cpu().tolist())

                    # MoE diagnostics: per-region errors and region head accuracy
                    region_probs = out_dict.get('mosfet_region_probs')
                    region_labels = getattr(batch, 'mosfet_region_labels', None)
                    if region_probs is not None and region_labels is not None:
                        region_labels = region_labels.to(gm_pred.device)
                        # Region head accuracy
                        pred_regions = region_probs.argmax(dim=-1)
                        moe_region_correct += (pred_regions == region_labels).sum().item()
                        moe_region_total += region_labels.shape[0]
                        # Average routing probabilities
                        batch_probs = region_probs.mean(dim=0).cpu()
                        if moe_region_probs_sum is None:
                            moe_region_probs_sum = batch_probs
                        else:
                            moe_region_probs_sum = moe_region_probs_sum + batch_probs
                        moe_num_batches += 1
                        # Per-region gm/gds log errors (using valid mask)
                        gm_log_err = (gm_pred_log - gm_tgt_log).abs()
                        gds_log_err = (gds_pred_log - gds_tgt_log).abs()
                        valid_labels = region_labels[valid]
                        for r in range(3):
                            rmask = (valid_labels == r)
                            if rmask.any():
                                moe_per_region_gm_errors[r].extend(gm_log_err[rmask].cpu().tolist())
                                moe_per_region_gds_errors[r].extend(gds_log_err[rmask].cpu().tolist())
        total_triode_physics_loss += triode_physics_loss.float() * batch_size
        total_triode_eq1_loss += triode_eq1_loss.float() * batch_size
        total_triode_eq2_loss += triode_eq2_loss.float() * batch_size
        total_triode_eq3_loss += triode_eq3_loss.float() * batch_size
        total_cutoff_physics_loss += cutoff_physics_loss.float() * batch_size
        if region_loss_weight > 0:
            total_region_loss += region_loss.float() * batch_size
        if vov_loss_weight > 0:
            total_vov_loss += vov_loss.float() * batch_size
        if vth_loss_weight > 0:
            total_vth_loss += vth_loss.float() * batch_size
        if dc_gain_loss_weight > 0:
            total_dc_gain_loss += dc_gain_loss.float() * batch_size

        # Denormalize for error analysis — always use net-node predictions
        # for fair comparison across runs (regardless of terminal loss setting)
        if use_terminal_voltage_loss:
            net_mask = get_prediction_mask(batch)
            net_pred = out[net_mask]
            net_target = batch.vdc.flatten()
            pred_mv, target_mv = denormalize_voltage(net_pred, net_target, vdc_mean, vdc_std)
        else:
            pred_mv, target_mv = denormalize_voltage(pred, target, vdc_mean, vdc_std)
        errors_mv = (pred_mv - target_mv).abs()
        all_errors.extend(errors_mv.cpu().tolist())
        rel_errors_pct = errors_mv / torch.clamp(target_mv.abs(), min=10.0) * 100
        all_rel_errors.extend(rel_errors_pct.cpu().tolist())

    all_errors = np.array(all_errors)
    all_rel_errors = np.array(all_rel_errors)
    all_i_rel_errors = np.array(all_i_rel_errors) if all_i_rel_errors else np.array([])

    avg_loss = (total_loss / total_count).item()
    avg_voltage_loss = (total_voltage_loss / total_count).item()
    avg_current_loss = (total_current_loss / total_current_count).item() if total_current_count > 0 else 0.0
    # MAE from all_errors (always net-node based, in mV)
    mae_mv = all_errors.mean()
    current_mae_ua = (total_current_mae / total_current_count).item() if total_current_count > 0 else 0.0

    # Accuracy metrics
    acc80, acc50, acc20, acc10 = compute_voltage_accuracy(all_errors)
    if all_current_preds:
        current_acc50, current_acc20, current_acc10, current_acc5 = compute_current_accuracy(
            np.array(all_current_preds), np.array(all_current_targets)
        )
    else:
        current_acc50, current_acc20, current_acc10, current_acc5 = 0.0, 0.0, 0.0, 0.0
    avg_kcl_loss = (total_kcl_loss / total_count).item()
    avg_diff_pair_loss = (total_diff_pair_loss / total_count).item() if constraint_weight > 0 else 0.0
    avg_mirror_loss = (total_mirror_loss / total_count).item() if constraint_weight > 0 else 0.0
    avg_output_stage_loss = (total_output_stage_loss / total_count).item() if constraint_weight > 0 else 0.0
    avg_lambda_mirror_loss = (total_lambda_mirror_loss / total_count).item() if constraint_weight > 0 else 0.0
    avg_gm_physics_loss = (total_gm_physics_loss / total_count).item()
    avg_ac_loss = (total_ac_loss / total_count).item() if ac_loss_weight > 0 else 0.0
    avg_ac_component_losses = {comp: (v / total_count).item() for comp, v in total_ac_component_losses.items()} if ac_loss_weight > 0 else {}
    avg_ss_gm_loss = (total_ss_gm_loss / total_count).item() if ss_gm_loss_weight > 0 else 0.0
    avg_ss_gds_loss = (total_ss_gds_loss / total_count).item() if ss_gds_loss_weight > 0 else 0.0
    avg_triode_physics_loss = (total_triode_physics_loss / total_count).item()
    avg_triode_eq1_loss = (total_triode_eq1_loss / total_count).item()
    avg_triode_eq2_loss = (total_triode_eq2_loss / total_count).item()
    avg_triode_eq3_loss = (total_triode_eq3_loss / total_count).item()
    avg_cutoff_physics_loss = (total_cutoff_physics_loss / total_count).item()
    avg_region_loss = (total_region_loss / total_count).item() if region_loss_weight > 0 else 0.0
    avg_vov_loss = (total_vov_loss / total_count).item() if vov_loss_weight > 0 else 0.0
    avg_vth_loss = (total_vth_loss / total_count).item() if vth_loss_weight > 0 else 0.0
    avg_dc_gain_loss = (total_dc_gain_loss / total_count).item() if dc_gain_loss_weight > 0 else 0.0

    # Relative error metrics
    v_rel_median = float(np.median(all_rel_errors)) if len(all_rel_errors) > 0 else 0.0
    v_rel_mean = float(all_rel_errors.mean()) if len(all_rel_errors) > 0 else 0.0
    v_rel_acc = {t: float((all_rel_errors < t).mean() * 100) for t in [1, 5, 10, 20]}
    i_rel_median = float(np.median(all_i_rel_errors)) if len(all_i_rel_errors) > 0 else 0.0
    i_rel_mean = float(all_i_rel_errors.mean()) if len(all_i_rel_errors) > 0 else 0.0
    i_rel_acc = {t: float((all_i_rel_errors < t).mean() * 100) for t in [1, 5, 10, 20]} if len(all_i_rel_errors) > 0 else {t: 0.0 for t in [1, 5, 10, 20]}
    all_current_errors_np = np.array(all_current_errors) if all_current_errors else np.array([])
    i_abs_acc = {t: float((all_current_errors_np < t).mean() * 100) for t in [50, 20, 5, 2]} if len(all_current_errors_np) > 0 else {t: 0.0 for t in [50, 20, 5, 2]}
    # SS accuracy metrics
    ss_metrics = None
    if all_gm_rel:
        gm_rel_arr = np.array(all_gm_rel)
        gds_rel_arr = np.array(all_gds_rel)
        gm_log_arr = np.array(all_gm_log_errors)
        gds_log_arr = np.array(all_gds_log_errors)
        ss_metrics = {
            'gm_acc': {t: float((gm_rel_arr < t).mean() * 100) for t in [10, 20, 50]},
            'gds_acc': {t: float((gds_rel_arr < t).mean() * 100) for t in [10, 20, 50]},
            'gm_median': float(np.median(gm_rel_arr)),
            'gds_median': float(np.median(gds_rel_arr)),
            'gm_log_mae': float(gm_log_arr.mean()),
            'gm_log_median': float(np.median(gm_log_arr)),
            'gds_log_mae': float(gds_log_arr.mean()),
            'gds_log_median': float(np.median(gds_log_arr)),
        }
        # MoE diagnostics
        if moe_region_total > 0:
            region_names = ['cutoff', 'triode', 'saturation']
            ss_metrics['moe_region_acc'] = moe_region_correct / moe_region_total
            if moe_region_probs_sum is not None and moe_num_batches > 0:
                avg_probs = moe_region_probs_sum / moe_num_batches
                ss_metrics['moe_avg_probs'] = avg_probs.tolist()
            else:
                ss_metrics['moe_avg_probs'] = None
            for r, name in enumerate(region_names):
                if moe_per_region_gm_errors[r]:
                    ss_metrics[f'moe_gm_mae_{name}'] = float(np.mean(moe_per_region_gm_errors[r]))
                    ss_metrics[f'moe_gds_mae_{name}'] = float(np.mean(moe_per_region_gds_errors[r]))

    rel_metrics = {
        'v_rel_median': v_rel_median, 'v_rel_mean': v_rel_mean, 'v_rel_acc': v_rel_acc,
        'i_rel_median': i_rel_median, 'i_rel_mean': i_rel_mean, 'i_rel_acc': i_rel_acc,
        'i_abs_acc': i_abs_acc,
        'ss_metrics': ss_metrics,
    }

    return avg_loss, mae_mv, avg_voltage_loss, avg_current_loss, current_mae_ua, acc80, acc50, acc20, acc10, current_acc50, current_acc20, current_acc10, current_acc5, avg_kcl_loss, avg_diff_pair_loss, avg_mirror_loss, avg_output_stage_loss, avg_lambda_mirror_loss, avg_gm_physics_loss, avg_ac_loss, avg_ss_gm_loss, avg_ss_gds_loss, avg_triode_physics_loss, avg_triode_eq1_loss, avg_triode_eq2_loss, avg_triode_eq3_loss, avg_cutoff_physics_loss, avg_region_loss, rel_metrics, avg_vov_loss, avg_vth_loss, avg_ac_component_losses, avg_dc_gain_loss


def validate_simple(model, loader, device, vdc_mean, vdc_std, current_mean, current_std,
                    predict_currents=False, current_weight=1.0):
    """
    Simplified validation for hyperparam search (returns fewer metrics).

    Returns:
        Dict with val_loss, val_mae, acc80, current_mae
    """
    model.eval()
    val_volt_loss_sum = 0
    val_volt_count = 0
    val_current_loss_sum = 0
    val_current_count = 0
    all_errors = []
    all_current_errors = []

    with torch.inference_mode():
        for batch in loader:
            out_dict = model(batch)
            pred = out_dict['node_voltages']
            mask = get_prediction_mask(batch)
            pred_masked = pred[mask]
            target = batch.vdc.flatten()

            val_volt_loss_sum += compute_loss(pred_masked, target).item() * len(target)
            val_volt_count += len(target)

            # Denormalize for voltage error
            pred_mv, target_mv = denormalize_voltage(pred_masked, target, vdc_mean, vdc_std)
            all_errors.extend((pred_mv - target_mv).abs().cpu().tolist())

            # Current loss and error
            if predict_currents and 'node_currents' in out_dict:
                current_mask = batch.has_current_mask
                if current_mask.any():
                    current_pred = out_dict['node_currents'][current_mask]
                    current_target = batch.node_current_targets[current_mask]
                    val_current_loss_sum += compute_loss(current_pred, current_target).item() * len(current_target)
                    val_current_count += len(current_target)
                    pred_current, target_current = denormalize_current(current_pred, current_target, current_mean, current_std)
                    current_err_ua = (pred_current - target_current).abs() * 1e6
                    all_current_errors.extend(current_err_ua.cpu().tolist())

    val_volt_loss = val_volt_loss_sum / val_volt_count
    val_current_loss = val_current_loss_sum / val_current_count if val_current_count > 0 else 0.0
    val_loss = val_volt_loss + current_weight * val_current_loss
    all_errors = np.array(all_errors)
    val_mae = all_errors.mean()
    acc80, _, _ = compute_voltage_accuracy(all_errors)
    current_mae = np.mean(all_current_errors) if all_current_errors else 0.0

    return {
        'val_loss': val_loss,
        'val_mae': val_mae,
        'acc80': acc80,
        'current_mae': current_mae,
    }
