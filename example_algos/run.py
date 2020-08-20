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
    parser.add_argument('--other', type=str, nargs='+')
    args = parser.parse_args()

    run_mode = args.run_mode
    model_type = args.model_type
    recipe = args.recipe                            # list
    other = args.other
    assert run_mode in ['train', 'predict', 'validate', 'statistics']
    assert model_type in ['unet', 'zcae', 'zunet', None]
    assert recipe in ['origin', 'predict', 'rotate', 'split_rotate', 'mask', None]
    if other != None:
        if type(other) == str:
            assert other in ['resolution', 'change_loss']
        elif type(recipe) == list:
            for x in other:
                assert x in ['resolution', 'change_loss']        
    else: other = []

    af = AlgoFactory()
    algo = af.getAlgo(run_mode=run_mode, model_type=model_type, recipe=recipe, other=other)
    algo.run()


def auto_predict():
    from example_algos.util.configure import TEST_DATASET_DIR, TRAIN_DATASET_DIR
    import os
    af = AlgoFactory()

    log_dir = os.path.join(TRAIN_DATASET_DIR, 'log')
    basic_kws = {
        'logger': 'tensorboard',
        'test_dir': TEST_DATASET_DIR,
        'load': True,
    }

    for dir_name in os.listdir(log_dir):
        if len(dir_name) >= 20: continue
        print(f'dir_name: {dir_name}')

        basic_kws['name'] = dir_name
        basic_kws['log_dir'] = log_dir
        basic_kws['load_path'] = os.path.join(log_dir, dir_name, 'checkpoint')
        algo = af.getAlgo(run_mode='predict', basic_kws=basic_kws)
        algo.run(num=20, return_rec=False)


def auto_validate():
    from example_algos.util.configure import TEST_DATASET_DIR
    import os
    test_dir = os.path.join(TEST_DATASET_DIR, 'eval')
    af = AlgoFactory()
    for dir_name in os.listdir(test_dir):
        print(f'dir_name: {dir_name}')
        algo = af.getAlgo(run_mode='validate')
        algo.name = dir_name
        algo.run()


def auto_statistics():
    from example_algos.util.configure import TEST_DATASET_DIR
    import os
    test_dir = os.path.join(TEST_DATASET_DIR, 'eval')
    af = AlgoFactory()
    for dir_name in os.listdir(test_dir):
        print(f'dir_name: {dir_name}')
        algo = af.getAlgo(run_mode='statistics')
        algo.name = dir_name
        algo.run()


if __name__ == '__main__':
    main()
    # auto_predict()
    # auto_statistics()