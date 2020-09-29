import os
import glob

files = glob.glob("out/*")
for file in files:
  os.remove(file)