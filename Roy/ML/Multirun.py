import Geoguessrmodel_Trainer_silent
from rich.progress import track
import torch

for i in track(range(5)):
    Geoguessrmodel_Trainer_silent
    # clear GPU memory
    torch.cuda.empty_cache()
    print(f"Run {i+1} completed.")