import os 
import re 
import sys 
import random 
import pickle
import numpy as np 
import tensorflow as tf  
from tensorflow.python.ops.lookup_ops import HashTable 
from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer
from tensorflow.python.client import device_lib 

from .network import Network
from .utils.vocab import Vocab
# from .utils.metrics import registry as metrics

class Model:
    prefix = 'checkpoint'
    best_model_name = 'best'

    def __init__(self, args, session, updates=None):
        self.args = args
        self.session = session

        # updates
        if not updates:
            updates = 0
        self.global_step = tf.get_variable('global_step', shape=(), dtype=tf.float32,
                                            initializer=tf.constant_initializer(updates), trainable=False)
        self.step = tf.assign_add(self.global_step, 1)  # update self.global by add 1 to it

        # placeholders
        table = HashTable(TextFileIdTableInitializer(filename=os.path.join(args, output_dir, 'vocab.txt')),
                            default_value=Vocab.unk())  # string to id table, generates one key-value pair per line
        
        self.q1_string = tf.placeholder(tf.string, [None, None], name='q1_str')
        self.q2_string = tf.placeholder(tf.string, [None, None], name='q2_str')
        self.q1 = tf.placeholder_with_default(table.lookup(self.q1_string), [None, None], name='q1')
        self.q2 = tf.placeholder_with_default(table.lookup(self.q2_string), [None, None], name='q2')
        self.q1_len = tf.placeholder(tf.int32, [None], name='q1_len')
        self.q2_len = tf.placeholder(tf.int32, [None], name='q2_len')
        self.y = tf.placeholder(tf.int32, [None], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, (), name='dropout_keep_prob')

        q1_mask = tf.expand_dim(tf.sequence_mask(self.q1_len, dtype=tf.float32), dim=-1)  # 返回一个表示每个单元的前N个位置的mask张量
        q2_mask = tf.expand_dim(tf.sequence_mask(self.q2_len, dtype=tf.float32), dim=-1)

        devices = self.get_variable_gpus() or ['/device:CPU:0']
        if not args.multi_gpu:
            devices = devices[:1]
        if len(devices) == 1:
            splits = 1
        else:
            splits = [tf.shape(self.q1)[0] // len(devices)] * (len(devices) - 1) + [-1]  # handle uneven split
        
        q1 = tf.split(self.q1, splits)
        q2 = tf.split(self.q2, splits)
        q1_mask = tf.split(q1_mask, splits)
        q2_mask = tf.split(q1_mask, splits)
        y = tf.split(self.y, splits)

        # network
        self.network = Network(args)

        # optimizer
        lr = tf.get_variable('lr', shape=(), dtype=tf.float32,
                                initializer=tf.constant_initializer(args.lr), trainable=False)
        lr_next = tf.cond(self.global_step < args.lr_warmup_steps,
                            true_fn=lambda: args.min_lr + 
                                    (args.lr - args.min_lr) / (1, args.lr_warmup_steps) * self.global_step,
                            false_fn=lambda: tf.maximum(args.min_lr, args.lr * args.lr_decay_rate ** tf.floor(
                                (self.global_step - args.lr_warmup_steps) / args.lr_decay_steps)))

        # 在一个计算图中，可以通过集合（collection）来管理不同类别的资源
        # tf.GraphKeys.UPDATE_OPS: ops的集合(图表运行时执行的操作,如乘法,ReLU等),而不是变量
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assgin(lr, lr_next, name='update_lr'))
        self.lr = lr 
        self.opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=args.beta1, beta2=args.beta2)

        # training graph
        tower_names = ['tower-{}'.format(i) for i in range(len(devices))] if len(devices) > 1 else ['']
        tower_logits = []
        tower_grads = []
        summaries = []
        loss = 0

        # 当系统检测到我们用了一个之前已经定义的变量时，就开启共享，否则就重新创建变量
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for i, device in enumerate(devices):
                with tf.device(device):
                    with tf.name_scope(tower_names[i]) as scope:
                        logits = self.network(q1[i], q2[i], q1_mask[i], q2_mask[i], self.dropout_keep_prob)
                        tower_logits.append(logits)
                        loss = self.get_loss(logits, y[i])
                        tf.get_variable_scope().reuse_variables() 
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = self.opt.compute_gradients(loss)
                        tower_grads.append(grads)
        
        gradients = []
        variables = []
        for grad_and_vars in zip(*tower_grads):  # *list: 将一组行转换为一组列
            if grad_and_vars[0][0] is None:
                msg = 'WARNING: trainable variable {} receives no grad.\n'.format(grad_and_vars[0][1].op.name)
                sys.stderr.write(msg)  # output error msg to window
                continue
            grad = tf.stack([g for g, _ in grad_and_vars])
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]  # use the 1st tower's pointer to the (shared) variable
            gradients.append(grad)
            variables.append(v)
        
        gradients, self.gnorm = tf.clip_by_global_norm(gradients, self.args.grad_clipping)
        """
        tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
        t_list 是梯度张量， clip_norm 是截取的比率，和上面的 clip_gradient是同一个东西。 这个函数返回截取过的梯度张量和一个所有张量的全局范数。
        这样就保证了在一次迭代更新中，所有权重的梯度的平方和在一个设定范围以内
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        """
        该函数接受的参数control_inputs，是Operation或者Tensor构成的list。返回的是一个上下文管理器，该上下文管理器用来控制在该上下文中的操作的依赖。
        也就是说，上下文管理器下定义的操作是依赖control_inputs中的操作的，control_dependencies用来控制control_inputs中操作执行后，
        才执行上下文管理器中定义的操作
        """
        with tf.control_dependencies(update_ops): # 先执行update_ops
            self.train_op = self.opt.apply_gradients(zip(gradients, variables))
        
        logits = tf.concat(tower_logits, 0)
        self.prob = tf.nn.softmax(logits, dim=1, name='prob')
        self.pred = tf.argmax(input=logits, aixs=1, name='pred')
        self.loss = tf.identify(loss, name='loss')
        summaries.append(tf.summary.scalar('training/lr', lr))
        summaries.append(tf.summary.scalar('training/gnomr', self.gnorm))
        summaries.append(tf.summary.scalar('training/loss', self.loss))

        # add summary
        self.summary = tf.summary.merge(summaries)

        # saver
        self.saver = tf.train.Saver([var for var in tf.global_variables() if 'Adam' not in var.name],
                                    max_to_keep=args.max_checkpoints)


    def update(self, sess, batch):
        feed_dict = self.process_data(batch, training=True)
        _, gnorm, loss, summary, lr = sess.run(
            [self.train_op, self.gnorm, self.loss, self.summary, ]
        )        




        