import os
import sys
import json5
import pdb
from pprint import pprint
from src.utils import params
from src.trainer import Trainer

def main():
    """
    takes 2 args
        train.py
        $config.json5
    """
    argv = sys.argv
    if len(argv) == 2:
        arg_groups = params.parse(sys.argv[1])  # list of tuples of obj and dict
        for args, config in arg_groups:
            trainer = Trainer(args)
            # pdb.set_trace()
            states = trainer.train()
            with open('models/log.jsonl', 'a') as f:
                f.write(json5.dump({
                    'data': os.path.basename(args.data_dir),
                    'params': config,
                    'state': states
                }))
                f.write('\n')
    elif len(argv) == 3 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print('Usage: "python train.py configs/xxx.json5"')
       


if __name__ == "__main__":
    main()
