import torch
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable

from adet.utils.comm import compute_locations, aligned_bilinear

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels):
    """
    Binary Lovasz hinge loss
        logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

def lovasz_loss(x, target):
    eps = 1e-6
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)

    x = torch.clamp(x, min=eps, max=1-eps)
    x = torch.log(x) - torch.log(1 - x)

    losses = []
    for i in range(n_inst):
        losses.append(lovasz_hinge(x[i], target[i]))
    loss = torch.stack(losses)

    return loss

def build_mask_pred(cfg):
    return MaskPred(cfg)


class MaskPred(nn.Module):
    def __init__(self, cfg):
        super(MaskPred, self).__init__()
        self.in_channels = cfg.MODEL.EMBEDMASK.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.EMBEDMASK.MASK_OUT_STRIDE

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        self.register_buffer("_iter", torch.zeros([1]))

        self.mask_loss_type = cfg.MODEL.EMBEDMASK.MASK_LOSS_TYPE
        self.mask_loss_alpha = cfg.MODEL.EMBEDMASK.MASK_LOSS_ALPHA

    def __call__(self, pixel_embed, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=pixel_embed.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = pixel_embed.sum() * 0 + pred_instances.proposal_embed.sum() * 0 + pred_instances.proposal_margin.sum() * 0
                losses["loss_mask"] = dummy_loss
            else:
                mask_prob = self.compute_mask_prob(pred_instances, pixel_embed, mask_feat_stride)

                if self.mask_loss_type == "Dice":
                    mask_losses = dice_coefficient(mask_prob, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                elif self.mask_loss_type == "Lovasz":
                    mask_losses = lovasz_loss(mask_prob, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                losses["loss_mask"] = loss_mask * self.mask_loss_alpha

            return losses
        else:
            if len(pred_instances) > 0:
                mask_prob = self.compute_mask_prob(pred_instances, pixel_embed, mask_feat_stride)
                pred_instances.pred_global_masks = mask_prob

            return pred_instances

    def compute_mask_prob(self, instances, pixel_embed, mask_feat_stride):
        proposal_embed = instances.proposal_embed
        proposal_margin = instances.proposal_margin
        im_inds = instances.im_inds

        dim, m_h, m_w = pixel_embed.shape[-3:]
        obj_num = proposal_embed.shape[0]
        pixel_embed = pixel_embed.permute(0, 2, 3, 1)[im_inds]

        proposal_embed = proposal_embed.view(obj_num, 1, 1, -1).expand(-1, m_h, m_w, -1)
        proposal_margin = proposal_margin.view(obj_num, 1, 1, dim).expand(-1, m_h, m_w, -1)
        mask_var = (pixel_embed - proposal_embed) ** 2
        mask_prob = torch.exp(-torch.sum(mask_var * proposal_margin, dim=3))

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_prob = aligned_bilinear(mask_prob.unsqueeze(1), int(mask_feat_stride / self.mask_out_stride))

        return mask_prob
