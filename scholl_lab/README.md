Workflow:

1. Label Videos
    1. Run skellyclicker `__main__.py`
        1. set `video_paths` to synchronized videos you want to label
        2. set `data_path` to `TRACKED_POINTS_JSON_PATH`
            1. Make sure points in `tracked_points.json` match what you want to label
        3. set `machine_labels_path` to `None`
2. Train initial model
    1. `run_pipeline.py`
    2. Need to fill out information in configs: ProjectConfig, DataConfig, TrainingConfig
        1. In DataConfig:
            1. set `folder_of_videos` to synchronized videos folder used for labelling in (1)
            2. set `labels_csv_path` to output of skellyclicker in (1)
    3. Make sure "bodyparts" and "skeleton" match data labelled from skellyclicker (must match `tracked_points.json`)
3. Analyze videos
    1. `analyze_videos.py`
    2. Set `annotate_videos=True` to save out videos with machine labels on them
        1. Takes more time and videos will need to be moved from "synchronized_videos" folder before continuing to label
4. Relabel based on machine label predictions
    1. Point skellyclicker to machine label csv and most recent csv used for training (at bottom of `__main__.py`)
        1. keep `video_paths` the same
        2. set `data_path` to the skellyclicker data from last iteration
        3. set `machine_labels_path` to output csv from 3 (in `model_outputs_iteration_N` in the project folder)
    2. Rerun skellyclicker `__main__.py` with updated CSV paths
5. Iterate model
    1. Run `iterate_model.py` with `config.yaml` path and path to skellyclicker training from (4)
6. Repeat steps (3) through (6)