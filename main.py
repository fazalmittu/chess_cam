#Piece Detection
import os
import numpy as np
import pandas as pd

os.chdir("./yolov7/")
os.system('python3 detect.py --weights runs/train/exp5/weights/best.pt --conf 0.1 --source board.jpg')

print("---------COORDS----")
coords = np.load("coords.npy")
coords_df = pd.DataFrame(coords, columns=["c1", "c2", "c3", "c4", "confidence", "label"])
print(coords_df.head())

#Grid Mapping

