import deeplabcut
from pathlib import Path

config_path = Path(r"D:\sfn\michael_wobble\wobble_board_project_3-Aaron-2024-02-12\config.yaml")
# deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', userfeedback=False)

deeplabcut.label_frames(config_path)

# deeplabcut.create_training_dataset(config_path)

# deeplabcut.check_labels(config_path, visualizeindividuals=True)
# deeplabcut.train_network(config_path, maxiters=100000, saveiters=50000,displayiters=1000)

# deeplabcut.evaluate_network(config_path, plotting=True)

# deeplabcut.analyze_videos()
