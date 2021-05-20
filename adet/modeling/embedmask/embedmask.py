# -*- coding: utf-8 -*-
import logging
from skimage import color

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .mask_pred import build_mask_pred
from .mask_branch import build_mask_branch

from adet.utils.comm import aligned_bilinear

import math

__all__ = ["EmbedMask"]


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class EmbedMask(nn.Module):
    """
    Main class for EmbedMask architectures
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        self.mask_pred = build_mask_pred(cfg)

        self.mask_out_stride = cfg.MODEL.EMBEDMASK.MASK_OUT_STRIDE

        self.max_proposals = cfg.MODEL.EMBEDMASK.MAX_PROPOSALS
        self.topk_proposals_per_im = cfg.MODEL.EMBEDMASK.TOPK_PROPOSALS_PER_IM

        self.mask_th = cfg.MODEL.EMBEDMASK.MASK_TH

        # build proposal head
        in_channels = self.proposal_generator.in_channels_to_top_module

        self.proposal_head = ProposalHead(cfg, in_channels)

        # build pixel head
        self.pixel_head = EmbedHead(cfg, cfg.MODEL.EMBEDMASK.MASK_BRANCH.OUT_CHANNELS)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        original_images = [x["image"].to(self.device) for x in batched_inputs]

        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)

        features = self.backbone(images_norm.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.add_bitmasks(gt_instances, images_norm.tensor.size(-2), images_norm.tensor.size(-1))
        else:
            gt_instances = None

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)
        pixel_embed = self.pixel_head(mask_feats, self.mask_branch.out_stride)

        proposals, proposal_losses = self.proposal_generator(
            images_norm, features, gt_instances, self.proposal_head, top_module_with_level=True
        )

        if self.training:
            mask_losses = self._forward_mask_heads_train(proposals, pixel_embed, gt_instances)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update(mask_losses)
            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, pixel_embed)

            padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images_norm.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w, mask_threshold=self.mask_th
                )

                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, pixel_embed, gt_instances):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances), device=pixel_embed.device).long()
                logger.info("clipping proposals from {} to {}".format(
                    len(pred_instances), self.max_proposals
                ))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)

                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)

            pred_instances = Instances.cat(kept_instances)

        pred_instances.proposal_embed = pred_instances.top_feats[:, :pixel_embed.size(1)]
        pred_instances.proposal_margin = pred_instances.top_feats[:, pixel_embed.size(1):]

        loss_mask = self.mask_pred(
            pixel_embed, self.mask_branch.out_stride,
            pred_instances, gt_instances
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, pixel_embed):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.proposal_embed = pred_instances.top_feat[:, :pixel_embed.size(1)]
        pred_instances.proposal_margin = pred_instances.top_feat[:, pixel_embed.size(1):]

        pred_instances_w_masks = self.mask_pred(
            pixel_embed, self.mask_branch.out_stride, pred_instances
        )

        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results


class ProposalHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        embed_dim = cfg.MODEL.EMBEDMASK.EMBED_DIM
        prior_margin = cfg.MODEL.EMBEDMASK.PRIOR_MARGIN
        init_margin_bias = math.log(-math.log(0.5) / (prior_margin ** 2))

        self.margin_reduce_factor = cfg.MODEL.EMBEDMASK.MARGIN_REDUCE_FACTOR
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.proposal_head = cfg.MODEL.EMBEDMASK.PROPOSAL_HEAD

        if self.proposal_head == "Conv3":
            # proposal margin
            self.spatial_margin_head = nn.Conv2d(
                in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            torch.nn.init.normal_(self.spatial_margin_head.weight, std=0.01)
            torch.nn.init.constant_(self.spatial_margin_head.bias, init_margin_bias)

            self.free_margin_head = nn.Conv2d(
                in_channels, embed_dim - 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            torch.nn.init.normal_(self.free_margin_head.weight, std=0.01)
            torch.nn.init.constant_(self.free_margin_head.bias, init_margin_bias)

            self.margin_scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

            # proposal embed
            self.embed_head = EmbedHead(cfg, in_channels)
        elif self.proposal_head == "Conv1_Conv3":
            # proposal margin
            self.margin_conv = nn.Conv2d(
                in_channels, 128, kernel_size=1, stride=1, padding=0, bias=True
            )
            torch.nn.init.normal_(self.margin_conv.weight, std=0.01)
            torch.nn.init.constant_(self.margin_conv.bias, 0)

            self.spatial_margin_head = nn.Conv2d(
                128, 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            torch.nn.init.normal_(self.spatial_margin_head.weight, std=0.01)
            torch.nn.init.constant_(self.spatial_margin_head.bias, init_margin_bias)

            self.free_margin_head = nn.Conv2d(
                128, embed_dim - 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            torch.nn.init.normal_(self.free_margin_head.weight, std=0.01)
            torch.nn.init.constant_(self.free_margin_head.bias, init_margin_bias)

            self.margin_scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

            # proposal embed
            self.embed_conv = nn.Conv2d(
                in_channels, 128, kernel_size=1, stride=1, padding=0, bias=True
            )
            torch.nn.init.normal_(self.embed_conv.weight, std=0.01)
            torch.nn.init.constant_(self.embed_conv.bias, 0)

            self.embed_head = EmbedHead(cfg, 128)

    def forward(self, x, l):
        margin_x = x / self.margin_reduce_factor

        if self.proposal_head == "Conv1_Conv3":
            margin_x = self.margin_conv(margin_x)
            embed_x = self.embed_conv(x)
        else:
            embed_x = x

        spatial_margin = self.spatial_margin_head(margin_x)
        spatial_margin = torch.exp(self.margin_scales[l](spatial_margin))
        free_margin = self.free_margin_head(margin_x)
        free_margin = torch.exp(free_margin)

        proposal_embed = self.embed_head(embed_x, self.fpn_strides[l])

        proposal_feat = torch.cat([proposal_embed, spatial_margin, free_margin], dim=1)

        return proposal_feat

class EmbedHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        embed_dim = cfg.MODEL.EMBEDMASK.EMBED_DIM

        self.spatial_embed_head = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        torch.nn.init.normal_(self.spatial_embed_head.weight, std=0.01)
        torch.nn.init.constant_(self.spatial_embed_head.bias, 0)

        self.free_embed_head = nn.Conv2d(
            in_channels, embed_dim - 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        torch.nn.init.normal_(self.free_embed_head.weight, std=0.01)
        torch.nn.init.constant_(self.free_embed_head.bias, 0)

    def forward(self, x, stride):
        spatail_embed = self.spatial_embed_head(x)
        free_embed = self.free_embed_head(x)

        location_map = self.compute_location_map(
            x.size(2), x.size(3),
            stride=stride, device=x.device
        ).contiguous()
        location_map = location_map / 100.0
        spatail_embed = spatail_embed + location_map

        return torch.cat([spatail_embed, free_embed], dim=1)

    def compute_location_map(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        location_map = torch.stack((shift_x, shift_y), dim=0) + stride // 2

        return location_map

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale