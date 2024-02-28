# %% import stuff
from pathlib import Path
import deeplabcut

# %% load/create project

config_path = r"D:\\deeplabcut-projects\\pooh-the-cat-jsm-2024-02-28\\config.yaml"

if not Path(config_path).exists():  # noqa: F821
    response = deeplabcut.create_pretrained_project(
        "pooh-the-cat",
        "jsm",
        [r"C:\Users\jonma\Sync\videos\2024-02-28_pooh_gopros\GX010034_clip1.mp4"],
        model="superanimal_quadruped",
        working_directory=r"D:\deeplabcut-projects",
        copy_videos=True,
        videotype=".mp4",
        analyzevideo=True,
        filtered=True,
        createlabeledvideo=True,
        trainFraction=None,
    )
    print("-----------------------------")
    print(f"Created project: \n\n {response}")
    print("-----------------------------")

# %% Extract frames
    
deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', userfeedback=False, crop=False)

# %% Label frames - 

deeplabcut.label_frames(config_path)

# %%







#################

#     # Project definitions (do not edit)
# Task: pooh-the-cat
# scorer: jsm
# date: Feb28
# multianimalproject: false
# identity:

#     # Project path (change when moving around)
# project_path: D:/deeplabcut-projects/pooh-the-cat-jsm-2024-02-28

#     # Annotation data set configuration (and individual video cropping parameters)
# video_sets:
#   D:\deeplabcut-projects\pooh-the-cat-jsm-2024-02-28\videos\GX010034_clip1.mp4:
#     crop: 0, 2704, 0, 1520
# bodyparts:

# # skull
# - nose
# - lip_upper
# - lip_lower

# - right_mouth_end
# - right_jaw_end

# - left_mouth_end
# - left_jaw_end

# - right_eye_inner
# - right_eye_pupil
# - right_eye_outer
# - right_ear_base
# - right_ear_tip

# - left_eye_inner
# - left_eye_pupil
# - left_eye_outer
# - left_ear_base
# - left_ear_tip
# - skull_base

# # spine 
# - spine_t1
# - spine_l1
# - sacrum
# - tail_base
# - tail_mid_1
# - tail_mid_2
# - tail_tip

# # thorax
# - sternum_top
# - xyphoid_process

# # right arm
# - right_shoulder
# - right_elbow
# - right_wrist
# - right_front_paw_joint
# - right_front_paw_tip

# # left arm
# - left_shoulder
# - left_elbow
# - left_wrist
# - left_front_paw_joint
# - left_front_paw_tip

# # right leg
# - right_hip
# - right_knee
# - right_ankle
# - right_hind_paw_joint
# - right_hind_paw_tip



#     # Fraction of video to start/stop when extracting frames for labeling/refinement
# start: 0
# stop: 1
# numframes2pick: 20

#     # Plotting configuration
# skeleton:
# - - nose
#   - lip_upper
# - - lip_upper
#   - right_mouth_end
# - - lip_lower
#   - right_mouth_end
# - - lip_upper
#   - left_mouth_end
# - - lip_lower
#   - left_mouth_end
# - - lip_lower
#   - left_jaw_end
# - - lip_lower
#   - right_jaw_end
# - - left_jaw_end
#   - skull_base
# - - right_jaw_end
#   - skull_base

# - - right_eye_inner
#   - right_eye_pupil
# - - right_eye_pupil
#   - right_eye_outer
# - - right_eye_outer
#   - right_ear_base
# - - right_ear_base
#   - right_ear_tip

# - - left_eye_inner
#   - left_eye_pupil
# - - left_eye_pupil
#   - left_eye_outer
# - - left_eye_outer
#   - left_ear_base
# - - left_ear_base
#   - left_ear_tip

# - - skull_base
#   - spine_t1
# - - spine_t1
#   - spine_l1
# - - spine_l1
#   - sacrum
# - - sacrum
#   - tail_base
# - - tail_base
#   - tail_mid_1
# - - tail_mid_1
#   - tail_mid_2
# - - tail_mid_2
#   - tail_tip

# - - spine_t1
#   - sternum_top
# - - sternum_top
#   - xyphoid_process

# - - right_shoulder
#   - right_elbow
# - - right_elbow
#   - right_wrist
# - - right_wrist
#   - right_front_paw_joint
# - - right_front_paw_joint
#   - right_front_paw_tip

# - - left_shoulder
#   - left_elbow
# - - left_elbow
#   - left_wrist
# - - left_wrist
#   - left_front_paw_joint
# - - left_front_paw_joint
#   - left_front_paw_tip

# - - right_hip
#   - right_knee
# - - right_knee
#   - right_ankle
# - - right_ankle
#   - right_hind_paw_joint
# - - right_hind_paw_joint
#   - right_hind_paw_tip

# skeleton_color: black
# pcutoff: 0.6
# dotsize: 6
# alphavalue: 0.7
# colormap: rainbow

#     # Training,Evaluation and Analysis configuration
# TrainingFraction:
# - 0.95
# iteration: 0
# default_net_type: resnet_50
# default_augmenter: imgaug
# snapshotindex: -1
# batch_size: 8

#     # Cropping Parameters (for analysis and outlier frame detection)
# cropping: false
#     #if cropping is true for analysis, then set the values here:
# x1: 0
# x2: 640
# y1: 277
# y2: 624

#     # Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
# corner2move2:
# - 50
# - 50
# move2corner: true
