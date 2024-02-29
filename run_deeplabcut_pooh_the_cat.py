# %% import stuff
import deeplabcut

# %% load/create project

config_path = r"D:\\deeplabcut-projects\\pooh-the-cat-jsm-2024-02-28\\config.yaml"
pooh_videos = [
    r"D:\deeplabcut-projects\pooh-the-cat-jsm-2024-02-28\videos\GX010034_clip1.mp4"
]
# %% Extract frames

deeplabcut.extract_frames(
    config_path, mode="automatic", algo="uniform", userfeedback=False, crop=False
)

# %% Label frames -

deeplabcut.label_frames(config_path)

# %% Check labels (print image with labels overlaid)

deeplabcut.check_labels(config_path, visualizeindividuals=True)

# %% Create training dataset

deeplabcut.create_training_dataset(
    config_path, net_type="resnet_50", augmenter_type="imgaug"
)


# %% Train network

deeplabcut.train_network(
    config_path,
    shuffle=1,
    trainingsetindex=0,
    max_snapshots_to_keep=5,
    autotune=False,
    displayiters=10,
    saveiters=1000,
    maxiters=300000,
    allow_growth=True,
)

# %% Evaluate network
deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=True)

# %% Filter predictions

deeplabcut.filterpredictions(
    config_path,
    pooh_videos,
    shuffle=1,
    trainingsetindex=0,
    filtertype="arima",
    p_bound=0.01,
    ARdegree=3,
    MAdegree=1,
    alpha=0.01,
)

# %% Plot trajectories

deeplabcut.plot_trajectories(config_path, pooh_videos)

# %% Analyze videos
deeplabcut.analyze_videos(config_path, pooh_videos, save_as_csv=True)

# %% Create labeled video
deeplabcut.create_labeled_video(config_path, pooh_videos, videotype='.mp4')

# %% Extract outlier frames
deeplabcut.extract_outlier_frames(config_path, pooh_videos)

# %% Refine labels
deeplabcut.refine_labels(config_path)

# %% Merge datasets
deeplabcut.merge_datasets(config_path)

# #################
# all_joints:
# - - 0
# - - 1
# - - 2
# - - 3
# - - 4
# - - 5
# - - 6
# - - 7
# - - 8
# - - 9
# - - 10
# - - 11
# - - 12
# - - 13
# - - 14
# - - 15
# - - 16
# - - 17
# - - 18
# - - 19
# - - 20
# - - 21
# - - 22
# - - 23
# - - 24
# - - 25
# - - 26
# - - 27
# - - 28
# - - 29
# - - 30
# - - 31
# - - 32
# - - 33
# - - 34
# - - 35
# - - 36
# - - 37
# - - 38
# - - 39
# - - 40
# - - 41
# - - 42
# - - 43
# - - 44
# all_joints_names:
# - nose
# - lip_upper
# - lip_lower
# - right_mouth_end
# - right_jaw_end
# - left_mouth_end
# - left_jaw_end
# - right_eye_inner
# - right_eye_outer
# - right_ear_base
# - right_ear_tip
# - left_eye_inner
# - left_eye_outer
# - left_ear_base
# - left_ear_tip
# - skull_base
# - spine_t1
# - spine_l1
# - sacrum
# - tail_base
# - tail_mid_1
# - tail_mid_2
# - tail_tip
# - sternum_top
# - xyphoid_process
# - right_shoulder
# - right_elbow
# - right_wrist
# - right_front_paw_joint
# - right_front_paw_tip
# - left_shoulder
# - left_elbow
# - left_wrist
# - left_front_paw_joint
# - left_front_paw_tip
# - right_hip
# - right_knee
# - right_ankle
# - right_hind_paw_joint
# - right_hind_paw_tip
# - left_hip
# - left_knee
# - left_ankle
# - left_hind_paw_joint
# - left_hind_paw_tip
# alpha_r: 0.02
# apply_prob: 0.5
# batch_size: 1
# contrast:
#   clahe: true
#   claheratio: 0.1
#   histeq: true
#   histeqratio: 0.1
# convolution:
#   edge: false
#   emboss:
#     alpha:
#     - 0.0
#     - 1.0
#     strength:
#     - 0.5
#     - 1.5
#   embossratio: 0.1
#   sharpen: false
#   sharpenratio: 0.3
# cropratio: 0.4
# dataset: 
#   training-datasets\iteration-0\UnaugmentedDataSet_pooh-the-catFeb28\pooh-the-cat_jsm95shuffle1.mat
# dataset_type: imgaug
# decay_steps: 30000
# display_iters: 1000
# global_scale: 1.0
# init_weights: 
#   C:\Users\jonma\github_repos\jonmatthis\DeepLabCut\deeplabcut\pose_estimation_tensorflow\models\pretrained\resnet_v1_50.ckpt
# intermediate_supervision: false
# intermediate_supervision_layer: 12
# location_refinement: true
# locref_huber_loss: true
# locref_loss_weight: 0.05
# locref_stdev: 7.2801
# lr_init: 0.0005
# max_input_size: 15000
# metadataset: 
#   training-datasets\iteration-0\UnaugmentedDataSet_pooh-the-catFeb28\Documentation_data-pooh-the-cat_95shuffle1.pickle
# min_input_size: 64
# mirror: false
# multi_stage: false
# multi_step:
# - - 0.005
#   - 10000
# - - 0.02
#   - 430000
# - - 0.002
#   - 730000
# - - 0.001
#   - 1030000
# net_type: resnet_50
# num_joints: 45
# pairwise_huber_loss: false
# pairwise_predict: false
# partaffinityfield_predict: false
# pos_dist_thresh: 17
# project_path: D:/deeplabcut-projects/pooh-the-cat-jsm-2024-02-28
# rotation: 25
# rotratio: 0.4
# save_iters: 50000
# scale_jitter_lo: 0.5
# scale_jitter_up: 1.25
