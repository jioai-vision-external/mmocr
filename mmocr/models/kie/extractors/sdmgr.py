# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
from mmdet.core import bbox2roi
from torch import nn
from torch.nn import functional as F

from mmocr.core import imshow_edge, imshow_node
from mmocr.models.builder import DETECTORS, build_roi_extractor
from mmocr.models.common.detectors import SingleStageDetector
from mmocr.utils import list_from_file

import torch
import dgl


@DETECTORS.register_module()
class SDMGR(SingleStageDetector):
    """The implementation of the paper: Spatial Dual-Modality Graph Reasoning
    for Key Information Extraction. https://arxiv.org/abs/2103.14470.

    Args:
        visual_modality (bool): Whether use the visual modality.
        class_list (None | str): Mapping file of class index to
            class name. If None, class index will be shown in
            `show_results`, else class name.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extractor=dict(
                     type='mmdet.SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=7),
                     featmap_strides=[1]),
                 visual_modality=False,
                 train_cfg=None,
                 test_cfg=None,
                 class_list=None,
                 init_cfg=None,
                 openset=False):
        super().__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg=init_cfg)
        self.visual_modality = visual_modality
        if visual_modality:
            self.extractor = build_roi_extractor({
                **extractor, 'out_channels':
                self.backbone.base_channels
            })
            self.maxpool = nn.MaxPool2d(extractor['roi_layer']['output_size'])
        else:
            self.extractor = None
        self.class_list = class_list
        self.openset = openset

    def forward_train(self, img, img_metas, relations, texts, gt_bboxes,
                      gt_labels,  len_of_nodes): #src_and_dst_nodes,
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details of the values of these keys,
                please see :class:`mmdet.datasets.pipelines.Collect`.
            relations (list[tensor]): Relations between bboxes.
            texts (list[tensor]): Texts in bboxes.
            gt_bboxes (list[tensor]): Each item is the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[tensor]): Class indices corresponding to each box.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        
        x = self.extract_feat(img, gt_bboxes)
        graphs=[]
        device="cuda"
        for ind, no_of_nodes in enumerate(len_of_nodes):
            G = dgl.DGLGraph()
            G.add_nodes(no_of_nodes.cpu())
            src, dst = img_metas[ind]["src_and_dst_nodes"][0], img_metas[ind]["src_and_dst_nodes"][1]
            G.add_edges(src, dst)
            G=G.to(device)
            graphs.append(G)
        g = dgl.batch(graphs)
        node_preds, edge_preds = self.bbox_head.forward(relations, texts, g, x)
        
        return self.bbox_head.loss(node_preds, edge_preds, gt_labels, len_of_nodes)

    def forward_test(self,
                     img,
                     img_metas,
                     relations,
                     texts,
                     gt_bboxes,
                     len_of_nodes,
                     rescale=False):
        
        print("LEN OF NODES IN TEST", len_of_nodes)
        x = self.extract_feat(img, gt_bboxes)
        graphs=[]
        
        # START code for inference
        device="cpu"
        # print(img_metas)
        img_metas=img_metas[0]
        # END code for inference

        # # START code for training
        # device="cuda"
        # # END code for training

        ind=0
        no_of_nodes=len_of_nodes
        G = dgl.DGLGraph()
        G.add_nodes(no_of_nodes.cpu())
        
        try:
            src, dst = img_metas[ind]["src_and_dst_nodes"][0], img_metas[ind]["src_and_dst_nodes"][1]
        except:
            print("ISSSSUEEEEEEE HERRERRERRERERE")
            print("IMG METAs", img_metas)
        G.add_edges(src, dst)
        G=G.to(device)
        graphs.append(G)
        g = dgl.batch(graphs)
        node_preds, edge_preds = self.bbox_head.forward(relations, texts, g, x)
        temp_edges = torch.cat(
            [rel.view(-1, rel.size(-1)) for rel in relations])
        return [
            dict(
                img_metas=img_metas,
                nodes=F.softmax(node_preds, -1),
                edges=F.softmax(edge_preds, -1))
        ]

    def extract_feat(self, img, gt_bboxes):
        if self.visual_modality:
            x = super().extract_feat(img)[-1]
            feats = self.maxpool(self.extractor([x], bbox2roi(gt_bboxes)))
            return feats.view(feats.size(0), -1)
        return None

    def show_result(self,
                    img,
                    result,
                    boxes,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Draw `result` on `img`.

        Args:
            img (str or tensor): The image to be displayed.
            result (dict): The results to draw on `img`.
            boxes (list): Bbox of img.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The output filename.
                Default: None.

        Returns:
            img (tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        idx_to_cls = {}
        if self.class_list is not None:
            for line in list_from_file(self.class_list):
                class_idx, class_label = line.strip().split()
                idx_to_cls[class_idx] = class_label

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if self.openset:
            img = imshow_edge(
                img,
                result,
                boxes,
                show=show,
                win_name=win_name,
                wait_time=wait_time,
                out_file=out_file)
        else:
            img = imshow_node(
                img,
                result,
                boxes,
                idx_to_cls=idx_to_cls,
                show=show,
                win_name=win_name,
                wait_time=wait_time,
                out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

        return img
