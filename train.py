import os
import sys
import json5
import pdb
from pprint import pprint
from src.utils import params
from src.trainer import Trainer

def main():
    argv = sys.argv
    if len(argv) == 2:
        arg_groups = params.parse(sys.argv[1])
        for args, config in arg_groups:
            trainer = Trainer(args)
            pdb.set_trace()


if __name__ == "__main__":
    main()