import Geoguessrmodel_Trainer_silent
from rich.progress import track

for i in track(range(5)):
    Geoguessrmodel_Trainer_silent.main()
    print(f"Run {i+1} completed.")