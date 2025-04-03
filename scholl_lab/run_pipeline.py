import os
import logging
from pathlib import Path
from datetime import datetime
import deeplabcut
import numpy as np
from dlc_utils.create_dlc_config import create_new_project
from dlc_utils.create_dlc_project_data import fill_in_labelled_data_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_dlc_pipeline(
    project_name, 
    bodyparts,
    source_data_path,
    labels_csv_path,
    output_directory=None,
    experimenter="user",
    skeleton=None,
):
    """
    Run the complete DeepLabCut pipeline from project creation to training.
    
    Parameters
    ----------
    project_name : str
        Name for the DeepLabCut project
    bodyparts : list
        List of bodyparts/keypoints to track
    source_data_path : str or Path
        Path to the directory containing the videos
    labels_csv_path : str or Path
        Path to the CSV file with labeled frames
    output_directory : str or Path, optional
        Directory to create the project in (default: current directory)
    experimenter : str, optional
        Name of the experimenter (default: "user")
    skeleton : list, optional
        List of connections between bodyparts for visualization
    
    Returns
    -------
    dict
        Information about the created project
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    full_project_name = f"{project_name}_{experimenter}_{timestamp}"
    
    # Create paths
    if output_directory is None:
        output_directory = Path.cwd()
    else:
        output_directory = Path(output_directory)
    
    source_data_path = Path(source_data_path)
    labels_csv_path = Path(labels_csv_path)
    
    # Make sure output directory exists
    output_directory.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Starting DLC pipeline for project: {full_project_name}")
    
    # Step 1: Create project
    logger.info("Creating project structure...")
    config_path, project_name = create_new_project(
        project=full_project_name,
        experimenter=experimenter,
        working_directory=str(output_directory),
        bodyparts=bodyparts,
        skeleton=skeleton
    )
    
    project_path = output_directory / full_project_name
    
    # Step 2: Fill in labeled data
    logger.info("Processing labeled frames...")
    labeled_frames = fill_in_labelled_data_folder(
        path_to_recording=source_data_path,
        path_to_dlc_project_folder=project_path,
        path_to_image_labels_csv=labels_csv_path,
        scorer_name=experimenter
    )
    
    # Step 3: Create training dataset
    logger.info("Creating training dataset...")
    deeplabcut.create_training_dataset(
        config=config_path,
    )
    
    # Step 4: Train network
    logger.info("Training network...")
    deeplabcut.train_network(
        config=config_path,
        epochs=2,
    )
    
    logger.info(f"Pipeline completed for project: {full_project_name}")
    logger.info(f"Project path: {project_path}")
    
    return {
        "project_name": full_project_name,
        "config_path": config_path,
        "project_path": str(project_path),
        "labeled_frames": labeled_frames
    }

if __name__ == "__main__":
    # # Example usage
    # bodyparts = [
    #     'left_ear', 'left_eye_inner', 'left_eye_outer', 
    #     'nose', 'right_ear', 'right_eye_inner', 'right_eye_outer'
    # ]
    
    # # Create skeleton connections
    # skeleton = [
    #     ['left_ear', 'left_eye_outer'],
    #     ['left_eye_outer', 'left_eye_inner'],
    #     ['left_eye_inner', 'nose'],
    #     ['nose', 'right_eye_inner'],
    #     ['right_eye_inner', 'right_eye_outer'],
    #     ['right_eye_outer', 'right_ear']
    # ]

    bodyparts = [
        "nose",
        "right_eye_inner",
        "left_eye_inner"
    ]

    skeleton = [
        ["nose", "right_eye_inner"],
        ["nose", "left_eye_inner"]
    ]
    
    # Run the pipeline
    project_info = run_dlc_pipeline(
        project_name="freemocap_sample_data_test",
        bodyparts=bodyparts,
        source_data_path=(Path.home() / "freemocap_data/recording_sessions/freemocap_test_data"),
        labels_csv_path=Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/skellyclicker_data/2025-04-02_15-02-27_skellyclicker_output.csv"),
        output_directory=Path("/Users/philipqueen/DLCtest/"),
        experimenter="philip",
        skeleton=skeleton
    )
    
    print(f"Project created: {project_info['project_name']}")
    print(f"\tConfig path: {project_info['config_path']}")