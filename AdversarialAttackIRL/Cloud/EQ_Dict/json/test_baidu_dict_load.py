import requests
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import json
import pathlib

current_path = pathlib.Path(__file__).resolve().parent
label_dict_path = current_path / './baidu_dict.json'
with label_dict_path.open(mode='r', encoding="utf-8") as f:
    label_dict = json.loads(f.read())
    print(label_dict)
