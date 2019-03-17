# pytorch-practice

![](https://img.shields.io/badge/version-1.0-brightgreen.svg)
![](https://img.shields.io/github/license/TubatuBD/pytorch-practice.svg)

``` python
# notebook 环境配置
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# 导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
import time
import json
import argparse
import ast
from os import listdir

# 爬虫
import requests
from bs4 import BeautifulSoup
import re
import MySQLdb
from pymongo import MongoClient
```
