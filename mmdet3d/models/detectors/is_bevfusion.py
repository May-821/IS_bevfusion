import mmcv
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class IS_BEVFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(ISFusionDetector, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                               pts_middle_encoder, pts_fusion_layer,
                                               img_backbone, pts_backbone, img_neck, pts_neck,
                                               pts_bbox_head, img_roi_head, img_rpn_head,
                                               train_cfg, test_cfg, pretrained, init_cfg, **kwargs)

        self.detach = kwargs.get('detach', False)

        out_size_factor = kwargs.get('out_size_factor', None)
        voxel_size = kwargs.get('voxel_size', None)
        self.pc_range = kwargs.get('pc_range', None)
        self.pillar_size = [voxel_size[0]*out_size_factor, voxel_size[1]*out_size_factor, self.pc_range[5]-self.pc_range[2]]

        self.pts_pillar_layer = Voxelization(
            max_num_points=self.fusion_encoder.num_points_in_pillar,
            voxel_size=self.pillar_size,
            max_voxels=(30000, 60000),
            point_cloud_range=self.pc_range)

        self.only_camera = kwargs.get('only_camera', False)
        self,only_lidar = kwargs.get('only_lidar', False)
        self.fusion = kwargs.get('fusion', False)

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
        **kwargs,
    ) -> torch.Tensor:
        x = x.data[0]
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = x.half()
        x = x.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        kwargs.update(dict(pts_backbone=self.bevfeats_processor['backbone']))

        x, ins_heatmap = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
            **kwargs,
        )
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    def isfusion(self, pts, pts_feats, img_feats, img_metas, batch_size, **kwargs):

        # create BEV space
        pillars, pillars_num_points, pillar_coors = self.voxelize(pts, voxel_type='pillar')
        pts_metas = {}
        pts_metas['pillars'] = pillars
        pts_metas['pillars_num_points'] = pillars_num_points
        pts_metas['pillar_coors'] = pillar_coors
        pts_metas['pts'] = pts
        pts_metas['pillar_size'] = self.pillar_size

        kwargs.update(dict(pts_metas=pts_metas))
        kwargs.update(dict(img_metas=img_metas))
        kwargs.update(dict(pts_backbone=self.pts_backbone))

        x = self.fusion_encoder(img_feats, pts_feats, batch_size, **kwargs)

        return x

    # def extract_pts_feat(self, pts, img_feats, img_metas, **kwargs):
    #     """Extract features of points."""
    #     if not self.with_pts_bbox:
    #         return None

    #     voxels, coors = self.dynamic_voxelize(pts)
    #     voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors, pts, img_feats, img_metas)
    #     batch_size = coors[-1, 0].item() + 1
    #     # x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, **kwargs)

    #     x, ins_heatmap = self.isfusion(pts, voxel_features, img_feats, img_metas, batch_size, **kwargs)

    #     if self.with_pts_neck:
    #         x = self.pts_neck(x)

    #     if self.training:
    #         return x, ins_heatmap
    #     else:
    #         return x

    # @torch.no_grad()
    # @force_fp32()
    # def dynamic_voxelize(self, points):
    #     """Apply dynamic voxelization to points.

    #     Args:
    #         points (list[torch.Tensor]): Points of each sample.

    #     Returns:
    #         tuple[torch.Tensor]: Concatenated points and coordinates.
    #     """
    #     coors = []
    #     # dynamic voxelization only provide a coors mapping
    #     for res in points:
    #         res_coors = self.pts_voxel_layer(res)
    #         coors.append(res_coors)
    #     points = torch.cat(points, dim=0)
    #     coors_batch = []

    #     for i, coor in enumerate(coors):
    #         coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
    #         coors_batch.append(coor_pad)
    #     coors_batch = torch.cat(coors_batch, dim=0)
    #     return points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # def extract_feat(self, points, img, img_metas, **kwargs):
    #     """Extract features from images and points."""
    #     img_feats = self.extract_img_feat(img, img_metas)
    #     pts_feats = self.extract_pts_feat(points, img_feats, img_metas, **kwargs)
    #     return (img_feats, pts_feats)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        if self.training:
            torch.cuda.empty_cache()

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)

        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if len(pts_feats) == 2:  # instance heatmap loss
            outs = self.pts_bbox_head(pts_feats[0], img_feats, img_metas)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, pts_feats[1]]
        else:
            outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        torch.cuda.empty_cache()
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox

        torch.cuda.empty_cache()
        return bbox_list