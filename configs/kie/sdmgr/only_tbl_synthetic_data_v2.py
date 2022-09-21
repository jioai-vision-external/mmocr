img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale, min_scale = 1024, 512

NUM_CLASSES=2
# dict(type='BetaSkeletonGraph', beta=0.5, max_neighbors=150),
train_pipeline = [
    
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(max_scale, min_scale), keep_ratio=True),
    dict(type='BetaSkeletonGraph', beta=0.5, max_neighbors=150),
    dict(type='UpdateRelationsAndGtlabelsUsingBetaEdges'),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels',  'len_of_nodes'],
        meta_keys=[
            'img_norm_cfg', 'img_shape', 'ori_filename', 'filename',
            'ori_texts', 'src_and_dst_nodes'
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(max_scale, min_scale), keep_ratio=True),
    dict(type='BetaSkeletonGraph', beta=0.5, max_neighbors=150),
    dict(type='UpdateRelationsAndGtlabelsUsingBetaEdges'),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes',  'len_of_nodes'],
        meta_keys=[
            'img_norm_cfg', 'img_shape', 'ori_filename', 'filename',
            'ori_texts', 'src_and_dst_nodes'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(max_scale, min_scale), keep_ratio=True),
    dict(type='BetaSkeletonGraph', beta=0.5, max_neighbors=150),
    dict(type='UpdateRelationsAndGtlabelsUsingBetaEdges'),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels',  'len_of_nodes'],
        meta_keys=[
            'img_norm_cfg', 'img_shape', 'ori_filename', 'filename',
            'ori_texts', 'src_and_dst_nodes'
        ])
]

dataset_type = 'KIEDataset'
# dataset_type = 'OpensetKIEDataset'
# data_root = '/data/jioai/indu/datasets/only_tbl_synthetic_data_v2'
data_root = '/mnt/only_tbl_synthetic_data_v2'
loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/only_tbl_synth_train_v2.txt',
    pipeline=train_pipeline,
    img_prefix="/mnt/only_tbl_synthetic_data_v2/",
    # img_prefix='/data/jioai/indu/datasets/only_tbl_synthetic_data_v2/',
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=False)
val = dict(
    type=dataset_type,
    ann_file=f'{data_root}/only_tbl_synth_val_v2.txt',
    pipeline=val_pipeline,
    img_prefix="/data/jioai/indu/datasets/only_tbl_synthetic_data_v2/",
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=False)
test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/only_tbl_synth_test_v2.txt',
    pipeline=test_pipeline,
    img_prefix="/data/jioai/indu/datasets/only_tbl_synthetic_data_v2/",
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=True)

# train2.txt tbl data on one document
# real_tbl_data.txt
# real_data_w_only_tbl_cropped.txt
real = dict(
    type=dataset_type,
    ann_file=f'{data_root}/real_data_1_cropped.txt',
    pipeline=test_pipeline,
    img_prefix="/data/jioai/indu/datasets/only_tbl_synthetic_data_v2/",#"/data/jioai/indu/datasets/only_tbl_synthetic_data_v1_cropped/",
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=True)

# real = dict(
#     type=dataset_type,
#     ann_file=f'/data/jioai/indu/datasets/annotated_data_1k_v2/real_data_v1_all.txt',
#     pipeline=test_pipeline,
#     img_prefix='/data/jioai/indu/datasets/6k_test_data_all/image_files',
#     loader=loader,
#     dict_file=f'{data_root}/dict.txt',
#     test_mode=True)

tmp = dict(
    type=dataset_type,
    ann_file=f'{data_root}/train4.txt',
    pipeline=test_pipeline,
    img_prefix="/mnt/only_tbl_synthetic_data_v2/",
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=True)

# tmp = dict(
#     type=dataset_type,
#     ann_file=f'/data/jioai/indu/datasets/annotated_data_1k_v2/test_v2.txt',
#     pipeline=test_pipeline,
#     img_prefix="/data/",
#     loader=loader,
#     dict_file=f'{data_root}/dict.txt',
#     test_mode=True)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=3),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=train,
    val=val,
    test=test,
    real=real,
    tmp=tmp)

evaluation = dict(
    interval=1,
    metric='macro_f1',
    save_best = 'auto',
    rule="greater",
    metric_options=dict(
        macro_f1=dict(
            ignores=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25])))

model = dict(
    type='SDMGR',
    backbone=dict(type='UNet', base_channels=16),
    bbox_head=dict(
        type='SDMGRHead', visual_dim=16, num_chars=92, num_classes=NUM_CLASSES),
    visual_modality=True,
    train_cfg=None,
    test_cfg=None,
    class_list=f'{data_root}/closed_class_list.txt')

optimizer = dict(type='Adam', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1,
    step=[40, 50])
total_epochs = 2

checkpoint_config = dict(interval=total_epochs -2)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# find_unused_parameters = True
find_unused_parameters = False

