import deeplabcut
from pathlib import Path

config_path = Path(r"D:\sfn\michael_wobble\wobble_board_project_3-Aaron-2024-02-12\config.yaml")    
path_to_video = Path(r"D:\sfn\michael_wobble\recording_10_03_16_gmt-5\synchronized_videos")
dest_folder = Path(r"D:\sfn\michael_wobble\recording_10_03_16_gmt-5\dlc_data")
deeplabcut.extract_outlier_frames(config=config_path, videos=[str(path_to_video)], destfolder=dest_folder, frames2use=10, outlieralgorithm='manual')

# deeplabcut.refine_labels(config_path)