import os
import math
import shutil
import pdb
from datetime import datetime
import json5

class Object:
    """
    @DynamicAttrs
    """
    pass


def parse(config_file):
    '''
<<<<<<< HEAD
    Parse config file and return a list of tuples of objects which holds training parameters as attributes

    Args:
        $config_file.json5
    
    Returns:
        configs: list of tuples
                tuple[0]:obj   param info
                    dir(tuple[0]):   
                        ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', 
                        '__hash__', '__index__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__parents__', 
                        '__reduce__', '__reduce_ex__', '__repeat__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 
                        '__weakref__', 'alignment', 'batch_size', 'beta1', 'beta2', 'blocks', 'connection', 'data_dir', 'dropout_keep_prob', 
                        'early_stopping', 'embedding_dim', 'embedding_mode', 'enc_layers', 'epochs', 'eval_epoch', 'eval_file', 'eval_per_samples', 
                        'eval_per_samples_warmup', 'eval_per_updates', 'eval_per_updates_warmup', 'eval_warmup_samples', 'eval_warmup_steps', 
                        'fusion', 'grad_clipping', 'hidden_size', 'kernel_size', 'log_file', 'log_per_samples', 'log_per_updates', 'lower_case', 
                        'lr', 'lr_decay_rate', 'lr_decay_samples', 'lr_decay_steps', 'lr_warmup_samples', 'lr_warmup_steps', 'max_checkpoints', 
                        'max_len', 'max_loss', 'max_vocab', 'metric', 'min_df', 'min_len', 'min_lr', 'min_samples', 'min_steps', 'multi_gpu', 
                        'name', 'output_dir', 'prediction', 'pretrained_embeddings', 'save', 'save_all', 'seed', 'sort_by_len', 'summary_dir', 
                        'tensorboard', 'tolerance_samples', 'watch_metrics']

                tuple[1]:dict    file info
=======
    configs: list of tuples
                tuple[0]:obj
                tuple[1]:dict
>>>>>>> 36da86d01011ec8a19186ae5b4d2228c8f7bb4c3
                    e.g.{'name': 'benchmark-0', '__parents__': ['default', 'data/snli'], 
                        '__repeat__': 10, 'eval_file': 'test', '__index__': 0})
    '''
    root = os.path.dirname(config_file)  # __parent__ in config is a relative path
    config_group = _load_param("", config_file) 
    if type(config_group) is dict:  # list of dict
        config_group = [config_group]
    configs = []
    for config in config_group:
        try:
            choice = config.pop('__iter__')
            assert len(choice) == 1, "only support iterating over 1 variable"
            key, value = next(iter(choice.items()))
        except KeyError:
            key, value = config.popitem()
            values = [value]
        for value in values:
            config[key] = value
            repeat = config.get('__repeat__', 1)
            for idx in range(repeat):
                config_ = config.copy()
                config_['__index__'] = idx
                if repeat > 1:
                    config_['name'] += '-' + str(idx)
                args = _parse_args(root, config_)
                configs.append((args, config_))
    return configs


def _load_param(root, file: str):
<<<<<<< HEAD
    """
    Turn a json5 file to a dict, grab key and values in json5

    Args:
        file: config json5 file
    
    Returns:
        dict
    """
=======
>>>>>>> 36da86d01011ec8a19186ae5b4d2228c8f7bb4c3
    file = os.path.join(root, file)
    if not file.endswith('.json5'):
        file += '.json5'
    with open(file) as f:
        config = json5.load(f)
        return config

def _parse_args(root, config):
<<<<<<< HEAD
    """
    get the param of json5 file in a hiearchy way
    
    Args:
        root: root dir of config files
        config: json5 file, __parents__ stores the hierachy file structure
    
    Return:
        args object
    """
    args = Object()
    assert type(config) is dict
    parents = config.get('__parents__', []) 
=======
    args = Object()
    assert type(config) is dict
    parents = config.get('__parents__', [])
>>>>>>> 36da86d01011ec8a19186ae5b4d2228c8f7bb4c3
    for parent in parents:
        parent = _load_param(root, parent)
        assert type(parent) is dict, 'only top-level configs can be a sequence'
        _add_param(args, parent)
    _add_param(args, config)
    _post_process(args)
    return args 


def _add_param(args, x:dict):
<<<<<<< HEAD
    """
    Add params to the args obj as attributes

    Args:
        args: obj
        x: json5 -> dict
    
    Return:
        args with attributes
    """
    for k, v in x.items():
        if type(v) is dict:
            _add_param(args, v)  # recursively check until reach the value of attributes
        else:
            k = _validate_param(k)  # k mush be an valid name
            if hasattr(args, k):  # if the args already has the attr, attr can only be overwritten when the types are the same
=======
    for k, v in x.items():
        if type(v) is dict:
            _add_param(args, v)
        else:
            k = _validate_param(k)
            if hasattr(args, k):  # 判断对象有没有该属性
>>>>>>> 36da86d01011ec8a19186ae5b4d2228c8f7bb4c3
                previous_type = type(getattr(args, k))
                current_type = type(v)
                assert previous_type is current_type or (isinstance(None, previous_type)) or \
                    (previous_type is float and current_type is int), \
                        f'param "{k}" of type {previous_type} can not be overwritten by type {current_type}'
            setattr(args, k, v)  # setattr(object, name, value) 设置属性值，属性不一定存在


def _validate_param(name):
<<<<<<< HEAD
    """
    check the name is a vaild str name

    Args:
        str
    """
=======
>>>>>>> 36da86d01011ec8a19186ae5b4d2228c8f7bb4c3
    name = name.replace('-', '_')
    if not str.isidentifier(name):
        raise ValueError(f'Invalid param name: {name}')
    return name


def _post_process(args: Object):
<<<<<<< HEAD
    """
    Output the ['data_dir', 'min_df', 'max_vocab', 'max_len', 'min_len', 'lower_case',
                    'pretrained_embeddings', 'embedding_mode'] to models/$dataset/data_config.json5
    
    Append the args.metric to args.watch_metrics

    append steps to updates attr to args

    """
    if not args.output_dir.startswith('models'):
        args.output_dir = os.path.join('models', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)  # e.g. models/snli
=======
    if not args.output_dir.startswith('models'):
        args.output_dir = os.path.join('models', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
>>>>>>> 36da86d01011ec8a19186ae5b4d2228c8f7bb4c3
    if not args.name:
        args.name = str(datetime.now())
    args.summary_dir = os.path.join(args.output_dir, args.name)
    if os.path.exists(args.summary_dir):
        shutil.rmtree(args.summary_dir)
<<<<<<< HEAD
    os.makedirs(args.summary_dir)  # e.g. models/snli/benchmark-0
    data_config_file = os.path.join(args.output_dir, 'data_config.json5')
    if os.path.exists(data_config_file):  # already has the config file in dir
=======
    os.makedirs(args.summary_dir)
    data_config_file = os.path.join(args.output_dir, 'data_config.json5')
    if os.path.exists(data_config_file):
>>>>>>> 36da86d01011ec8a19186ae5b4d2228c8f7bb4c3
        with open(data_config_file) as f:
            config = json5.load(f)
            for k, v in config.items():
                if not hasattr(args, k) or getattr(args, k) != v:
                    print('ERROR: Data configurations are different. Please use another output_dir or'
                    'remove the older one manually.')
                    exit()
    else:
        with open(data_config_file, 'w') as f:
            keys = ['data_dir', 'min_df', 'max_vocab', 'max_len', 'min_len', 'lower_case',
                    'pretrained_embeddings', 'embedding_mode']
            json5.dump({k:getattr(args, k) for k in keys}, f)
    
<<<<<<< HEAD
    args.metric = args.metric.lower()  # grab evaluation metric
=======
    args.metric = args.metric.lower()
>>>>>>> 36da86d01011ec8a19186ae5b4d2228c8f7bb4c3
    args.watch_metrics = [m.lower() for m in args.watch_metrics]
    if args.metric not in args.watch_metrics:
        args.watch_metrics.append(args.metric)
    assert args.pretrained_embeddings, 'pretrained embedding must be provided.'

    def samples2steps(n):
        return int(math.ceil(n / args.batch_size))

    if not hasattr(args, 'log_per_updates'):
        args.log_per_updates = samples2steps(args.log_per_samples)
    if not hasattr(args, 'eval_per_updates'):
        args.eval_per_updates = samples2steps(args.eval_per_samples)
    if not hasattr(args, 'eval_per_updates_warmup'):
        args.eval_per_updates_warmup = samples2steps(args.eval_per_samples_warmup)
    if not hasattr(args, 'eval_warmup_steps'):
        args.eval_warmup_steps = samples2steps(args.eval_warmup_samples)
    if not hasattr(args, 'min_steps'):
        args.min_steps = samples2steps(args.min_samples)
    if not hasattr(args, 'early_stopping'):
        args.early_stopping = samples2steps(args.tolerance_samples)
    if not hasattr(args, 'lr_warmup_steps'):
        args.lr_warmup_steps = samples2steps(args.lr_warmup_samples)
    if not hasattr(args, 'lr_decay_steps'):
        args.lr_decay_steps = samples2steps(args.lr_decay_samples)
