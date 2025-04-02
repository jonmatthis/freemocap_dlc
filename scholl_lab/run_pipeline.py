from dlc_utils.create_dlc_config import create_new_project
from dlc_utils.create_dlc_project_data import fill_in_labelled_data_folder
from pathlib import Path
import pandas as pd
from datetime import datetime

import deeplabcut

path_to_output_csv_from_labeller = Path(r'C:\Users\Aaron\Downloads\output.csv')
body_parts = ['left_ear', 'left_eye_inner', 'left_eye_outer', 'nose', 'right_ear', 'right_eye_inner', 'right_eye_outer']

path_to_create_dlc_project_folder = Path(r"C:\Users\Aaron\Documents")
project_name = "dlc_test"
scorer_name = "Aaron" #we can see if we can just get rid of this at some point
net_type = "resnet_50" 

path_to_freemocap_recording = Path(r"C:\Users\Aaron\FreeMocap_Data\recording_sessions\freemocap_test_data")

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
dlc_project_name = f"{project_name}_{scorer_name}_{timestamp}"



config = create_new_project(
    project = dlc_project_name,
    experimenter= scorer_name,
    working_directory=path_to_create_dlc_project_folder,
    bodyparts=body_parts
)

fill_in_labelled_data_folder(path_to_recording=path_to_freemocap_recording,
    path_to_dlc_project_folder=path_to_create_dlc_project_folder / dlc_project_name,
    path_to_image_labels_csv=path_to_output_csv_from_labeller,
    scorer_name=scorer_name)

deeplabcut.create_training_dataset(
    config= str(path_to_create_dlc_project_folder / dlc_project_name / "config.yaml"),
)

deeplabcut.train_network(
    config= str(path_to_create_dlc_project_folder / dlc_project_name / "config.yaml"),
)