import deeplabcut


config_path = r"D:\sfn\michael wobble\Wobble Board Proect-Aaron-2024-01-31\config.yaml"
# deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', userfeedback=False)

# deeplabcut.label_frames(config_path)

# deeplabcut.create_training_dataset(config_path)

# deeplabcut.check_labels(config_path, visualizeindividuals=True)
# deeplabcut.train_network(config_path)

# deeplabcut.evaluate_network(config_path, plotting=True)

deeplabcut.analyze_videos()
