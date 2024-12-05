import glob

import numpy as np

metrics = ["RotErr", "TransErr", "CamMC"]
for folder in glob.glob("results/*/*"):

    with open(f"{folder}/merge.csv", "w") as f:
        f.write("Name,Time," + ",".join(metrics) + "\n")

        data: dict[str, list] = {}
        for file in glob.glob(f"{folder}/trial_*.csv"):

            lines = open(file, "r").readlines()[1:]

            for line in lines:
                name, *metric = line.strip().split(",")
                if name not in data:
                    data[name] = []
                data[name].append(tuple(map(eval, metric)))

        for name, metric in data.items():
            f.write(name + "," + ",".join(map(str, np.mean(metric, axis=0))) + "\n")
