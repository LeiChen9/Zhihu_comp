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
    #this message is from mac        


if __name__ == "__main__":
    main()
