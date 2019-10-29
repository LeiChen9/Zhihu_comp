import os
import random
import json5
import numpy as np
import tensorflow as tf
from datetime import datetime
from pprint import pformat
from .utils.loader import load_data
from .utils.logger import Logger
# from .utils.params import validate_params
# from .model import Model
# from .interface import Interface

class Trainer:
    def __init__(self, args):
        self.args = args
        self.log = Logger(self.args)
    
    def train(self):
        start_time = datetime.now()
        train = load_data(self.args.data_dir, 'train')