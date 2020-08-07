import os
import glob
import re

cwd = os.getcwd().replace('/', '-')
paths = glob.glob('results/*/cache/*/joblib/')

for path in paths:
    old_path = glob.glob(path + '*')[0]
    part = old_path[len(path):]
    groups = list(re.search(r'(__main__-)(.*)(-results.*)', part).groups())
    groups[1] = cwd
    new_path = path + ''.join(groups)
    os.rename(old_path, new_path)
