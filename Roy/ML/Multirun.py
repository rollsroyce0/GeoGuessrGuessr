import Geoguessrmodel_Trainer_silent
from rich.progress import track
import torch
import gc
import importlib
for i in track(range(10), description="Running Geoguessrmodel_Trainer_silent..."):
    Geoguessrmodel_Trainer_silent.main()
    print(f"Run {i+1} completed.")