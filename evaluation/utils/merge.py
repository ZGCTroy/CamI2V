import glob
from itertools import chain
import numpy as np

metrics = ["RotErr", "TransErr_rel", "CamMC_rel", "TransErr_abs", "CamMC_abs"]
for folder in glob.glob("results/*/*"):

    with open(f"{folder}/merge.csv", "w") as f:
        f.write("Name," + ",".join(metrics) + "\n")

        data: dict[str, list] = {}
        for file in glob.glob(f"{folder}/trial_*.csv"):

            lines = open(file, "r").readlines()[1:]

            for line in lines:
                name, time, *metric = line.strip().split(",")
                if name not in data:
                    data[name] = []
                data[name].append(tuple(map(eval, metric)))

        data_list = list(chain.from_iterable(data.values()))
        data_mean = np.mean(data_list, axis=0)
        data_std = np.std(data_list, axis=0)

        for name, metric in data.items():
            valid = []
            for entry in metric:
                flag = True
                for item, mean, std in zip(entry, data_mean, data_std):
                    if abs(item - mean) > 10 * std:
                        flag = False
                        break
                if flag:
                    valid.append(entry)
            
            if valid:
                f.write(name + "," + ",".join(map(str, np.mean(valid, axis=0))) + "\n")
