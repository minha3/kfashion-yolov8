import argparse
import os
import kfashion


os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    if args.prepare:
        kfashion.prepare('/opt/datasets/kfashion', os.path.expanduser('~/datasets/kfashion'))
    elif args.train:
        kfashion.train(data=os.path.expanduser('~/datasets/kfashion/kfashion.yaml'), epochs=7)
