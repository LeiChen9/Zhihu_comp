import os
import sys
import logging

'''
默认情况下，logging模块将日志打印到屏幕上(stdout)，日志级别为WARNING(即只有日志级别高于WARNING的日志信息才会输出)

Output:
WARNING:root:warn message   日志级别：Logger实例名称：日志消息内容
ERROR:root:error message 
CRITICAL:root:critical message  

Logger 记录器，暴露了应用程序代码能直接使用的接口。
Handler 处理器，将（记录器产生的）日志记录发送至合适的目的地。
Filter 过滤器，提供了更好的粒度控制，它可以决定输出哪些日志记录。
Formatter 格式化器，指明了最终输出中日志记录的布局。

'''

class Logger:
    """
    Logger to log the info into summary_dir in models dir
    handler:
        fh: INFO level log
        ch: DEBUG level log
    formatter:
        time + message
    
    __call__:
        can call the logger itself to write down INFO level log

    """
    def __init__(self, args):
        log = logging.getLogger(args.summary_dir)  # build a logger
        # handler send log to appropriate output
        if not log.handlers:
            log.setLevel(logging.DEBUG)
            fh = logging.FileHandler(os.path.join(args.summary_dir, args.log_file))
            fh.setLevel(logging.INFO)
            ch = ProgressHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            log.addHandler(fh)
            log.addHandler(ch)
        self.log = log
        # Setup Tensorboard
        if args.tensorboard:
            import tensorflow as tf 
            from tensorboardX import SummaryWriter
            summary_dir = os.path.join(args.summary_dir, 'viz')
            self.writer = tf.summary.FileWriter(summary_dir) # write to models folder
            
            self.eval_writer = SummaryWriter(summary_dir)
            self.log.info(f'TensorBoard activated')
        else:
            self.writer = None
            self.eval_writer = None
        self.log_per_updates = args.log_per_updates
        self.grad_clipping = args.grad_clipping
        self.clips = 0
        self.train_meters = {}
        self.epoch = None
        self.best_eval = 0. 
        self.best_eval_str = ''
    
    def set_epoch(self, epoch):
        self(f'Epoch: {epoch}')
        self.epoch = epoch

    @staticmethod
    def _format_number(x):
        """
        Formatting the float number
        """
        return f'{x: .4f}' if float(x) > 1e-3 else f'{x: .4e}'
    
    def update(self, stats):
        '''
        stats: model training info

        if updates % log_per_updates == 0:
            add summary to log
            update clips
            add stats_str to log
        neet to be solved after implemented the model
        '''
        updates = stats.pop('updates')
        if updates % self.log_per_updates == 0:
            summary = stats.pop('summary')
            if self.writer:
                self.writer.add_summary(summary, updates)
            self.clips += int(stats['gnorm'])
            stats_str = ' '.join(f'{key}: ' + self._format_number(val) for key, val in stats.items())
            for key, val in stats.items():
                if key not in self.train_meters:
                    self.train_meters[key] = AverageMeter()
                self.train_meters[key].update(val)
            msg = f'epoch {self.epoch} updates {updates} {stats_str}'
            if self.log_per_updates != 1:
                msg = '> ' + msg
            self.log.info(msg)
    
    def newline(self):
        self.log.debug('')
    
    def log_eval(self, valid_stats):
        """
        updates the new best eval to log
        """
        self.newline()
        updates = valid_stats.pop('updates')
        eval_score = valid_stats.pop('score')
        # report the exponential avg training stats, while reporting the full dev set stats
        if self.train_meters:
            train_stats_str = ' '.join(f'{key}: ' + self._format_number(val) for key, val in self.train_meters.items())
            train_stats_str += ' ' + f'clip: {self.clips}'
            self.log.info(f'train {train_stats_str}')
        valid_stats_str = ' '.join(f'{key}: ' + self._format_number(val) for key, val in valid_stats.items())
        if eval_score > self.best_eval:
            self.best_eval_str = valid_stats_str
            self.best_eval = eval_score
            valid_stats_str += ' [NEW BEST]'
        else:
            valid_stats_str += f' [BEST: {self._format_number(self.best_eval)}]'
        self.log.info(f'valid {valid_stats_str}')
        if self.eval_writer:
            for key in valid_stats.keys():
                group = {'valid': valid_stats[key]}
                if self.train_meters and key in self.train_meters:
                    group['train'] = float(self.train_meters[key])
                self.eval_writer.add_scalars(f'valid/{key}', group, updates)
        self.train_meters = {}
        self.clips = 0
    
    def __call__(self, msg):
        self.log.info(msg)


class ProgressHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):  # Set logger's level to NOTSET, default is WARNING
        super().__init__(level)
    
    def emit(self, record):
        log_entry = self.format(record)
        if record.message.startswith('> '):
            sys.stdout.write('{}\r'.format(log_entry.rstrip()))
            sys.stdout.flush()
        else:
            sys.stdout.write('{}\n'.format(log_entry))
            

class AverageMeter(object):
    '''keep exponential weighted averages'''
    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0. 
        self.value = 0. 
        self.t = 0. 
    
    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.value = self.moment / (1 - self.beta ** self.t)
    
    def __format__(self, spec):
        return format(self.value, spec)
    
    def __float__(self):
        return self.value 
