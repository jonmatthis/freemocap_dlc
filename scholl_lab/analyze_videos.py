from pathlib import Path
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
import pandas as pd

def merge_csvs_for_skellyclicker(csv_folder_path: str | Path, output_path: str | Path):
    dataframe_list = []
    for csv in Path(csv_folder_path).glob("*.csv"):
        df = pd.read_csv(csv)

        video_name = Path(csv).stem.split("DLC_")[0]

        bodyparts = df.iloc[0, :].unique()[1:]

        column_names = ["frame"]
        for bodypart in bodyparts:
            column_names.extend([f"{bodypart}_x", f"{bodypart}_y", f"{bodypart}_likelihood"])

        df.columns = column_names

        # remove first two rows
        df = df.iloc[2:, :]

        df = df.drop(columns=[f"{bodypart}_likelihood" for bodypart in bodyparts])

        df["video"] = video_name

        # set frames and video as multi index
        df = df.set_index(["video", "frame"])

        print(df.head())

        dataframe_list.append(df)

    df = pd.concat(dataframe_list)
    print(df)

    df.to_csv(output_path)
    print(f"Saved skellyclicker compatible CSV to {output_path}")


def analyze_videos(config_path: str | Path, path_to_recording_folder: str | Path | None = None):
    config = auxiliaryfunctions.read_config(config_path)
    if path_to_recording_folder is None:
        path_to_recording_folder = config["skellyclicker_folder_of_videos"]
        print(f"Using default path to recording folder: {path_to_recording_folder}")
    deeplabcut.analyze_videos(
        config=str(config_path),
        videos=[str(path_to_recording_folder)],
        videotype=".mp4",
        save_as_csv=True,
    )

    merge_csvs_for_skellyclicker(path_to_recording_folder, Path(config["project_path"]) / f"skellyclicker_output_iteration_{config['iteration']}.csv")

if __name__ == "__main__":
    config_path = Path("/Users/philipqueen/DLCtest/sample_data_test2_user_20250403/config.yaml")
    analyze_videos(config_path=config_path)