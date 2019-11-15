import os 
import subprocess 
from functools import partial
import numpy as np 
from sklearn import metrics

from .registry import register

registry = {}
register = partial(register, registry=registry)


@register('acc')
def acc(outputs):
    target = outputs['target']
    pred = outputs['pred']
    return {
        'acc': metrics.accuracy_score(target, pred).item()
    }


@register('f1')
def f1(outputs):
    target = outputs['target']
    pred = outputs['pred']
    return {
        'f1': metrics.f1_score(target, pred).item()
    }

@register('auc')
def auc(outputs):
    target = outputs['target']
    prob = np.array(outputs['prob'])
    return {
        'auc': metrics.roc_auc_score(target, prob[:, 1]).item()
    }

@register('map')
@register('mrr')
def ranking(outputs):
    args = outputs['args']
    prediction = [o[1] for o in outputs['prob']]
    ref_file = os.path.join(args.data_dir, '{}.ref'.format(args.eval_file))  # 写字板格式
    rank_file = os.path.join(args.data_dir, '{}.rank'.format(args.eval_file))
    tmp_file = os.path.join(args.summmary_dir, 'tmp-pred.txt')

    with open(rank_file) as f:
        prefix = []
        for line in f:
            prefix.append(line.strip().split())
        assert len(prefix) == len(prediction), \
            'prefix {}, while prediction {}'.format(len(prefix, len(prediction)))
    
    with open(tmp_file, 'w') as f:
        for prefix, pred in zip(prefix, prediction):
            prefix[-2] = str(pred)
            f.write(' '.join(prefix) + '\n')

    """subprocess: fork一个子进程，让这个子进程exec另外一个程序
        Popen: Popen对象创建后，主程序不会自动等待子进程完成，必须调用对象的wait()方法,父进程才会等待
    """
    sp = subprocess.Popen('./resources/trec_eval {} {} | egrep "map|recip_rank"'.format(ref_file, tmp_file),
                            shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()  # type(stdout): bytes
    stdout, stderr = stdout.decode(), stderr.decode()  # turn bytes to str
    os.remove(tmp_file)
    map_, mrr = [float(s[-6:]) for s in stdout.strip().split('\n')]
    return {
        'map': map_,
        'mrr': mrr
    }