"""Loss functions
source: https://gist.github.com/ranftlr/1d6194db2e1dffa0a50c9b0a9549cbd2
"""
# Script for finetuning losses

import torch

from chrislib.loss import compute_scale_and_shift

EPSILON = 1e-6
_dtype = torch.float32

# ===================== MiDaS Losses =======================


def reduction_batch_based(image_loss, M):
    """Average of all valid pixels of the batch. Avoid division by 0.

    params:
        image_loss (TODO): TODO
        M (TODO): TODO

    returns:
        (float): TODO
    """
    divisor = torch.sum(M)

    if divisor == 0:
        ret = 0
    else:
        ret = torch.sum(image_loss) / divisor
    return ret


def reduction_image_based(image_loss, M):
    """Average of all valid pixels of an image. Avoid division by 0.

    params:
        image_loss (TODO): TODO
        M (TODO): TODO

    returns:
        (float): TODO
    """
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def midas_mse_loss(pred_depth, gt_depth, mask, reduction):
    """TODO DESCRIPTION

    params:
        pred_depth (TODO): TODO
        gt_depth (TODO): TODO
        mask (TODO): TODO
        reduction (TODO): TODO

    returns:
        (TODO): TODO
    """
    M = torch.sum(mask, (1, 2))
    res = pred_depth - gt_depth
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def midas_gradient_loss(prediction, target, mask, reduction):
    """TODO DESCRIPTION

    params:
        prediction (TODO): TODO
        target (TODO): TODO
        mask (TODO): TODO
        reduction (TODO): TODO

    returns:
        (TODO): TODO
    """
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def normalize_prediction_robust(target, mask):
    """TODO DESCRIPTION

    params:
        target (TODO): TODO
        mask (TODO): TODO

    returns:
        (TODO): TODO
    """
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum).to(_dtype)
    s = torch.ones_like(ssum).to(_dtype)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values

    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))

    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))


def trimmed_mae_loss(prediction, target, mask, reduction, trim=0.2):
    """TODO DESCRIPTION

    params:
        prediction (TODO): TODO
        target (TODO): TODO
        mask (TODO): TODO
        reduction (TODO): TODO
        trim (float) optional: TODO (default 0.2)

    returns:
        (TODO): TODO
    """
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)
    trimmed = trimmed[: int(len(res.view(-1)) * (1.0 - trim))]

    return reduction(trimmed, 2 * M)


def ssi_mse_loss(pred_depth, gt_depth, mask, reduction, do_scale=True, do_shift=True):
    """TODO DESCRIPTION

    params:
        pred_depth (TODO): TODO
        gt_depth (TODO): TODO
        mask (TODO): TODO
        reduction (TODO): TODO
        do_scale (bool) optional: TODO (default True)
        do_shift (bool) optional: TODO (default True)

    returns:
        ssi_mae_term (TODO): TODO
        grad_term (TODO): TODO
    """
    # compute scale and shift values
    scale, shift = compute_scale_and_shift(pred_depth, gt_depth, mask)

    scale = torch.nn.functional.relu(scale)
    pred_ssi = pred_depth

    if do_scale:
        pred_ssi = pred_ssi * scale.view(-1, 1, 1)

    if do_shift:
        pred_ssi = pred_ssi + shift.view(-1, 1, 1)

    ssi_mse_term = midas_mse_loss(pred_ssi, gt_depth, mask, reduction)

    grad_term = 0.0
    for ds in range(4):
        scale = pow(2, ds)

        pred_scale = pred_ssi[:, ::scale, ::scale]
        gt_scale = gt_depth[:, ::scale, ::scale]
        mask_scale = mask[:, ::scale, ::scale]

        grad_term += midas_gradient_loss(pred_scale,
                                         gt_scale, mask_scale, reduction)

    return ssi_mse_term, grad_term


def trim_mae_loss(pred_depth, gt_depth, mask, reduction):
    """TODO DESCRIPTION

    params:
        pred_depth (TODO): TODO
        gt_depth (TODO): TODO
        mask (TODO): TODO
        reduction (TODO): TODO

    returns:
        trim_mae_term (TODO): TODO
        grad_term (TODO): TODO
    """
    # normalized disparity -- zero translation and unit scaling
    pred_ssi = normalize_prediction_robust(pred_depth, mask)
    gt_ssi = normalize_prediction_robust(gt_depth, mask)

    trim_mae_term = trimmed_mae_loss(
        pred_ssi, gt_ssi, mask, reduction, trim=0.2)

    grad_term = 0.0

    for ds in range(4):
        scale = pow(2, ds)

        pred_scale = pred_ssi[:, ::scale, ::scale]
        gt_scale = gt_ssi[:, ::scale, ::scale]
        mask_scale = mask[:, ::scale, ::scale]

        grad_term += midas_gradient_loss(pred_scale,
                                         gt_scale, mask_scale, reduction)

    return trim_mae_term, grad_term


# loss is considered in the disparity space for MiDaS
def midas_loss(
        gt_depth,
        pred_depth,
        mask,
        loss_type='trim_mae',
        reduction='batch',
        alpha=0.5):
    """TODO DESCRIPTION

    params:
        gt_depth (TODO): TODO
        pred_depth (TODO): TODO
        mask (TODO): TODO
        loss_type (str) optional: TODO (default "trim_mae")
        reduction (TODO) optional: TODO (default "batch")
        alpha (float) optional: TODO (default 0.5)

    returns:
        total_loss (TODO): TODO
    """
    reduction = reduction_batch_based if reduction == 'batch' else reduction_image_based

    # downscaling image, gt depth and pred depth
    # mask = torch.isfinite(gt_depth).int()

    # gradient loss
    grad_term = 0.0
    ssi_mse_term = 0.0
    trim_mae_term = 0.0
    total_loss = 0.0

    if loss_type == 'ssi_mse':
        # scale and shift invariant mae loss
        ssi_mse_term, grad_term = ssi_mse_loss(
            pred_depth, gt_depth, mask, reduction)
        total_loss = ssi_mse_term
    elif loss_type == 'trim_mae':
        # scale and shift normalized trim loss
        trim_mae_term, grad_term = trim_mae_loss(
            pred_depth, gt_depth, mask, reduction)
        total_loss = trim_mae_term
    elif loss_type == 'si_mse':
        ssi_mse_term, grad_term = ssi_mse_loss(
            pred_depth, gt_depth, mask, reduction, do_shift=False)
        total_loss = ssi_mse_term
    elif loss_type == 'mse':
        ssi_mse_term, grad_term = ssi_mse_loss(
            pred_depth, gt_depth, mask, reduction, do_shift=False, do_scale=False)
        total_loss = ssi_mse_term
    else:
        pass

    if alpha > 0:
        total_loss += alpha * grad_term

    return total_loss
