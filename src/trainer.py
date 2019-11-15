import os
import random
import json5
import numpy as np
import tensorflow as tf
from datetime import datetime
from pprint import pformat
from .utils.loader import load_data
from .utils.logger import Logger
from .utils.params import validate_params
from .model import Model
from .interface import Interface

class Trainer:
    """
    __init__: Load args and define logger
    
    
    train:
        Split train and dev set;
        Set up tf graph
        build session
        mode, interface, states = self.build_model(sess)


    build_model:
        states = {}
        Define interface
        model = Model(args, sess)
    """
    def __init__(self, args):
        self.args = args
        self.log = Logger(self.args)
    
    def train(self):
        # Setup train set and dev set
        start_time = datetime.now()
        train = load_data(self.args.data_dir, 'train')  # looks like 'data/snli/train.txt'
        dev = load_data(self.args.data_dir, self.args.eval_file)  # looks like 'data/snli/test.txt'
        self.log(f'train ({len(train)}) | {self.args.eval_file} ({len(dev)})')

        # Setup tf graph
        tf.reset_default_graph()
        with tf.Graph().as_default():
            config = tf.ConfigProto()  # build the session and set parameters
            # config.gpu_options.allow_growth = True  
            # config.allow_soft_replacement = True
            sess = tf.Session(config=config)
            with sess.as_default():
                model, interface, states = self.build_model(sess)
            # to be continued


    def build_model(self, sess):
        states = {}
        interface = Interface(self.args, self.log)
        # import pdb; pdb.set_trace()
        self.log(f'#classes: {self.args.num_classes}; #vocab: {self.args.num_vocab}')
        if self.args.seed:  # Set seed to random, np and tf
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            tf.set_random_seed(self.args.seed)
        
        model = Model(self.args, sess)
        import pdb; pdb.set_trace()
        sess.run(tf.global_variables_initializer())
        embeddings = interface.load_embeddings()
        
