# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv import rescale_size
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle, to_tensor

try:
    from ...utils import get_beta_skeleton
except:
    print("Unable to load `beta_skeleton` utility")
    get_beta_skeleton = None

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

@PIPELINES.register_module()
class ResizeNoImg:
    """Image resizing without img.

    Used for KIE.
    """

    def __init__(self, img_scale, keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        w, h = results['img_info']['width'], results['img_info']['height']
        if self.keep_ratio:
            (new_w, new_h) = rescale_size((w, h),
                                          self.img_scale,
                                          return_scale=False)
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            (new_w, new_h) = self.img_scale

        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img_shape'] = (new_h, new_w, 1)
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = True

        return results


@PIPELINES.register_module()
class KIEFormatBundle(DefaultFormatBundle):
    """Key information extraction formatting bundle.

    Based on the DefaultFormatBundle, itt simplifies the pipeline of formatting
    common fields, including "img", "proposals", "gt_bboxes", "gt_labels",
    "gt_masks", "gt_semantic_seg", "relations" and "texts".
    These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - proposals: (1) to tensor, (2) to DataContainer
    - gt_bboxes: (1) to tensor, (2) to DataContainer
    - gt_bboxes_ignore: (1) to tensor, (2) to DataContainer
    - gt_labels: (1) to tensor, (2) to DataContainer
    - gt_masks: (1) to tensor, (2) to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1) unsqueeze dim-0 (2) to tensor,
                       (3) to DataContainer (stack=True)
    - relations: (1) scale, (2) to tensor, (3) to DataContainer
    - texts: (1) to tensor, (2) to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        super().__call__(results)
        if 'ann_info' in results:
            for key in ['relations', 'texts']:
                value = results['ann_info'][key]
                if key == 'relations' and 'scale_factor' in results:
                    scale_factor = results['scale_factor']
                    if isinstance(scale_factor, float):
                        sx = sy = scale_factor
                    else:
                        sx, sy = results['scale_factor'][:2]
                    r = sx / sy
                    factor = np.array([sx, sy, r, 1, r]).astype(np.float32)
                    value = value * factor[None, None]
                results[key] = DC(to_tensor(value))
        return results

    def __repr__(self):
        return self.__class__.__name__

def viz_edge_data(results, prefix, out_path='/data/jioai/indu/graph-neural-network/externals/open-mmlab/mmocr/model_edge_rev2/'):

    img_path = results["filename"]
    img=cv2.imread(img_path)
    dim = results['img_shape'][:2]
    img = cv2.resize(img, [dim[1], dim[0]], interpolation = cv2.INTER_AREA)
    img1=img.copy()
    img_name=os.path.basename(img_path)
    no_of_nodes = results["len_of_nodes"]
    for c, [s, d] in enumerate(zip(results["src_and_dst_nodes"][0], results["src_and_dst_nodes"][1])):

        scale=1
        box1 = [int(a) for a in results["gt_bboxes"][s]]
        box2 = [int(a) for a in results["gt_bboxes"][d]]
        # print("BOX!< BOX2", box1, box2)
        if results['gt_labels'][no_of_nodes+c]==1:
            cv2.line(img, (box1[0]*scale, box1[1]*scale), (box2[0]*scale, box2[1]*scale), (0, 255, 0), 2)
        if results['gt_labels'][no_of_nodes+c]==0:
            cv2.line(img1, (box1[0]*scale, box1[1]*scale), (box2[0]*scale, box2[1]*scale), (0, 0, 255), 2)
    
    cv2.imwrite(out_path+prefix+img_name, img)
    cv2.imwrite(out_path+prefix+'1L'+img_name, img1)
        

@PIPELINES.register_module()
class BetaSkeletonGraph:
    def __init__(self, beta, max_neighbors=150):
        self.beta = beta
        self.max_neighbors = max_neighbors

    def __call__(self, results):
        
        # print("Results", results.keys())
        # print("ANN INFO", results["ann_info"].keys())
        # print(results["ann_info"]['labels'].shape, results["gt_labels"].shape)
        assert (results["ann_info"]['labels'] == results["gt_labels"]).all()
        # print("START BETA EDGES")
        src_ids, dst_ids =[], []
        bbs = results["ann_info"]["bboxes"].astype(np.uint16)
        if get_beta_skeleton is not None:
            edges = get_beta_skeleton(
                bbs=bbs,
                beta=self.beta,
                max_neighbors=self.max_neighbors,
            )
            beta_edges = [(i, j) for (i, j, _) in edges]
            beta_edges = set(beta_edges)
            results["beta_edges"] = [[i, j] for (i, j, _) in edges]
            beta_edges=results["beta_edges"]
            # print("EDGES", edges)
            labels = results["ann_info"]["labels"]
            # print("LABELS NCALSESES, nwords, _nclasses_", labels.shape)
            nwords, _nclasses_ = labels.shape
            nclasses = _nclasses_ - nwords
            # nodes = labels[:, :nclasses]
            # edges = labels[:, nclasses:]
            nodes = labels[:, :1]
            edges = labels[:, 1:]
            for i in range(nwords):
                for j in range(nwords):
                    
                    # if i==j:
                    #     src_ids.append(i)
                    #     dst_ids.append(j)
                    if [i, j] not in beta_edges:
                        edges[i, j] = -1
                        
                    else:
                        src_ids.append(i)
                        dst_ids.append(j)
                        # pass
                    
            labels = np.concatenate([nodes, edges], axis=-1)
            results['ann_info']['labels'] = labels
        else:
            # logger.warning("BETA EDGES: FALLBACK TO NxN connection")
            results["beta_edges"] = [
                [i, j] for i in range(len(bbs)) for j in range(len(bbs))
            ]
        
        # # print(len(src_ids), len(results["beta_edges"]))
        # assert len(src_ids)==len(beta_edges)
        # assert len(src_ids)==len(results["beta_edges"])
        
        results["src_and_dst_nodes"]=[src_ids, dst_ids]
        results["len_of_nodes"]=len(nodes)
        # print("END BETA EDGES")
        return results

import cv2
import os
@PIPELINES.register_module()
class UpdateRelationsAndGtlabelsUsingBetaEdges():

    # def __init__():
        # pass
    
    def __call__(self, results):

        beta_edges = results["beta_edges"]
        visualize=False
        if visualize:
            img_path = results["filename"]
            img=cv2.imread(img_path)
            dim = results['img_shape'][:2]
            img = cv2.resize(img, [dim[1], dim[0]], interpolation = cv2.INTER_AREA)
            img1=img.copy()
            img_name=os.path.basename(img_path)

        updated_relations =np.array([], dtype=np.int32)
        updated_gt_labels = np.array([], dtype=np.int32)
        updated_gt_labels= np.append(updated_gt_labels, results['gt_labels'][:,:1])
        
        for beta_edge in beta_edges:
            row_id, col_id = beta_edge[0], beta_edge[1]
            updated_relations = np.append(updated_relations, results['ann_info']["relations"][row_id][col_id])
            updated_gt_labels = np.append(updated_gt_labels, results['gt_labels'][row_id][col_id+1])
            if visualize:
                scale=1
                box1 = [int(a) for a in results["gt_bboxes"][row_id]]
                box2 = [int(a) for a in results["gt_bboxes"][col_id]]
                
                if results['gt_labels'][row_id][col_id+1]==1:
                    cv2.line(img, (box1[0]*scale, box1[1]*scale), (box2[0]*scale, box2[1]*scale), (0, 255, 0), 2)
                if results['gt_labels'][row_id][col_id+1]==0:
                    cv2.line(img1, (box1[0]*scale, box1[1]*scale), (box2[0]*scale, box2[1]*scale), (0, 0, 255), 2)
       
        updated_relations = updated_relations.reshape(len(beta_edges), 5)
        
        results['ann_info']["relations"]=updated_relations
        assert len(beta_edges) == len(updated_relations)
        results['gt_labels']=updated_gt_labels
        
        if visualize:
            cv2.imwrite('./mmocr/model_edge_rev2/'+img_name, img)
            cv2.imwrite('./mmocr/model_edge_rev2/1L'+img_name, img1)
        
        return results

# # g = dgl.graph((src_ids, dst_ids), num_nodes=len(nodes))
# G = dgl.DGLGraph()
# G.add_nodes(len(nodes))
# G.add_edges(src_ids, dst_ids)
# # G = dgl.add_self_loop(G)
# results["graph"]=G