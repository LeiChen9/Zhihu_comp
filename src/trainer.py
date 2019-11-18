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
                train_batches = interface.pre_process(train)
                dev_batches = interface.pre_process(dev, training=False)
                self.log('setup complete: {}s'.format(str(datetime.now() - start_time).split(".")[0]))
                try:
                    for epoch in range(states['start_epoch'], self.args.epochs + 1):
                        states['epoch'] = epoch
                        self.log.set_epoch(epoch)

                        batches = interface.shuffle_batch(train_batches)
                        for batch_id, batch in enumerate(batches):
                            stats = model.update(sess, batch)  # get new stats: updates, loss, lr, gnorm, summary
                            self.log.update(stats)
                            eval_per_updates = self.eval_per_updates \
                                if model.updates > self.args.eval_warmup_steps else self.args.eval_per_updates_warmup
                            if model.updates % eval_per_updates == 0 \
                                    or (self.args.eval_epoch and batch_id + 1 == len(batches)):
                                score, dev_stats = model.evaluate(sess, dev_batches)
                                if score > states['best_eval']:
                                    states['best_eval'], states['best_epoch'], states['best_step'] = \
                                        score, epoch, model.updates 
                                    if self.args.save:
                                        model.save(states, name=model.best_model_name)
                                self.log.log_eval(dev_stats)
                                if self.args.save_all:
                                    model.save(states)
                                    model.save(states, name='last')
                                if model.updates - states['best_step'] > self.args.early_stopping \
                                    and model.updates > self.args.min_steps:
                                    raise EarlyStop('[Tolerance reached. Training is stopped early.]')
                            if states['loss'] > self.args.max_loss:
                                raise EarlyStop('[Loss exceeds tolerance. Unstable training is stopped early.]')
                            if states['lr'] < self.args.min_lr - 1e-6:
                                raise EarlyStop('[Learning rate has decayed below min_lr. Training is stopped early.]')
                        self.log.newline()
                    self.log('Training complete.')
                except KeyboardInterrupt:
                    self.log.newline()
                    self.log(f'Training interupted. Stopped early')
                except EarlyStop as e:
                    self.log.newline()
                    self.log(str(e))
                self.log(f'best dev score {states["best_eval"]} at step {states["best_step"]} '
                         f'(epoch {states["best_epoch"]}).')
                self.log(f'best eval stats [{self.log.best_eval_str}]')
                training_time = str(datetime.now() - start_time).split('.')[0]
                self.log(f'Training time: {training_time}.')
        states['start_time'] = str(start_time).split('.')[0]
        states['training_time'] = training_time
        return states 



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
        model.set_embeddings(sess, embeddings)

        # set initial states
        states['start_epoch'] = 1
        states['best_eval'] = 0. 
        states['best_epoch'] = 0
        states['best_step'] = 0 

        self.log(f'trainable params: {model.num_parameters():,d}')
        self.log(f'trainable parameters (exclude embeddings): {model.num_parameters(exclude_embedding=True):,d}')
        validate_params(self.args)
        with open(os.path.join(self.args.summary_dir, 'args.json5'), 'w') as f:
            args = {k: v for k, v in vars(self.args).items() if not k.startswith('_')}
            json5.dump(args, f, indent=2)  # indent: print across multiple lines
        self.log(pformat(vars(self.args), indent=2, width=120))
        return model, interface, states 
        
class EarlyStop(Exception):
    pass