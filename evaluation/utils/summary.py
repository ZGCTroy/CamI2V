import glob
import os
import re

import numpy as np
import pandas as pd

regs = {int: r"(0|[1-9]\d*)", float: r"(\d+(\.\d+)?)", bool: r"(True|False)", str: r"([a-z]+)"}
configs = {
    "iters": ("", "k", int, 0),
    "ImageTextcfg": ("ImageTextcfg", "", float, 7.5),
    "CameraCondition": ("CameraCondition", "", bool, False),
    "CameraCfg": ("CameraCfg", "", float, 1.0),
    "eta": ("eta", "", float, 1.0),
    "guidanceRescale": ("guidanceRescale", "", float, 0.7),
    "cfgScheduler": ("cfgScheduler=", "", str, "constant"),
    "frameStride": ("frameStride", "", int, 8),
}


def order(file):
    def capture(before, after, kind, default):
        cap = re.search(f"{before}{regs[kind]}{after}", setting)
        if cap is None:
            return default
        item = cap.group(1)
        return eval(item) if kind in (int, float, bool) else item

    _, method, setting, _ = file.split("/")
    return method, [capture(*v) for v in configs.values()]


metrics = ["RotErr", "TransErr", "CamMC"]
with open("summary.csv", "w") as f:
    f.write("Method,ImageTextcfg,CameraCfg,Time," + ",".join(metrics) + "\n")

summary = []
for file in sorted(glob.glob("results/*/*/merge.csv"), key=order):

    _, method, setting, _ = file.split("/")
    method = re.sub(r"_\d+(_\dgpu)?$", "", method.removeprefix("test_256_"))
    ticfg = re.search(rf"ImageTextcfg({regs[float]})", setting).group(1)
    ccfg = re.search(rf"CameraCfg({regs[float]})", setting).group(1)

    data = pd.read_csv(file, skiprows=[0]).iloc[:, 1:].mean(axis=0).values.tolist()

    summary.append([method, ticfg, ccfg] + data)

    print(list(map(lambda x: round(x, 4), data)))

pd.DataFrame(summary).to_csv("summary.csv", mode="a", header=False, index=False)
