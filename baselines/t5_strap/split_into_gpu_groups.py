import os
import sys

import glob


folder = sys.argv[1]
NUM_GPUS = int(sys.argv[2])
paths = sorted(glob.glob(f"{folder}/*/"))

# split into NUM_GPUS groups

groups = [[] for _ in range(NUM_GPUS)]
for i, path in enumerate(paths):
    groups[i % NUM_GPUS].append(path)

for i, group in enumerate(groups):
    with open(f"gpu_{i}.txt", "w") as f:
        for path in group:
            f.write(path + "\n")
