# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:33:14 2023

@author: zhuhe
"""

import requests
from urllib.request import urlretrieve
import re
import os

# REPLACE WITH YOUR PATH
TARGET_PATH = r'D:\Dropbox\Vasp_home\Machine_learning on TP\Kyoto-phonopy'

BASE_URL = 'http://phonondb.mtl.kyoto-u.ac.jp/_downloads/'
PATTERN = r"mp-\d+-20180417.tar.lzma"

req = requests.get(BASE_URL)
all_files = re.findall(PATTERN, req.text)
all_files = set(all_files)
for f in all_files:
    download_url = os.path.join(BASE_URL, f)
    target_f = os.path.join(TARGET_PATH, f)
    if os.path.exists(target_f):
        print('existed file:' ,target_f)
        continue
    urlretrieve(download_url, target_f)