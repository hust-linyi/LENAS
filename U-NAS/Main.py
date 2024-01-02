from preprocess import preprocess
from search import Searching
from train import Training
from prediction import Prediction

'''
Configurations are set in config.yml.
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Preprocessing
# preprocess()

# Searching process
s = Searching()
gene = s.search()

