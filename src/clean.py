import glob
import os

files = glob.glob("out/*")
for file in files:
    os.remove(file)
