"""
Configuration classes for DeepLabCut pipeline
---------------------------------------------
Classes to organize and manage configuration settings for the DeepLabCut pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import os


@dataclass
class ProjectConfig:
    """Configuration for project setup"""
    
    # Basic project information
    name: str
    experimenter: str = "user"
    working_directory: Optional[Union[Path, str]] = None
    
    # Anatomical configuration
    bodyparts: List[str] = field(default_factory=list)
    skeleton: Optional[List[List[str]]] = None
        
    def __post_init__(self):
        if isinstance(self.working_directory, str):
            self.working_directory = Path(self.working_directory)
        elif self.working_directory is None:
            self.working_directory = Path.cwd()
            
        self.working_directory.mkdir(exist_ok=True, parents=True)


@dataclass
class DataConfig:
    """Configuration for data processing"""
    
    folder_of_videos: Path
    labels_csv_path: Path
    
    def __post_init__(self):
        if isinstance(self.folder_of_videos, str):
            self.folder_of_videos = Path(self.folder_of_videos)
        if isinstance(self.labels_csv_path, str):
            self.labels_csv_path = Path(self.labels_csv_path)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Network settings
    model_type: str = "resnet_50"
    
    # Training settings
    epochs: int = 200, #this is the new equivalent of 'maxiters' for PyTorch (200 is their default)
    save_epochs: int = 20, #this is the new equivalent of 'save_iters' for PyTorch
    batch_size: int = 1 #this seems to be similar to batch/multi processing (higher number = faster if your gpu can handle it?)
