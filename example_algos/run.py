import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

from example_algos.util.factory import AlgoFactory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='predict')
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--recipe', type=str, default=None)
    args = parser.parse_args()

    run_mode = args.run_mode
    model_type = args.model_type
    recipe = args.recipe
    assert run_mode in ['train', 'predict', 'validate', 'statistics']
    assert model_type in ['unet', 'zcae', 'zunet', None]
    assert recipe in ['origin', 'predict', 'mask', 'rotate', 'split_rotate', None]

    af = AlgoFactory()
    algo = af.getAlgo(run_mode=run_mode, model_type=model_type, recipe=recipe)
    algo.run(algo)

if __name__ == '__main__':
    main()