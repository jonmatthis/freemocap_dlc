from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple



@dataclass
class ProjectConfig:
    project_name: str
    experimenter_name: str
    video_directory: Path
    working_directory: Path
    copy_videos_to_dlc_folder: bool
