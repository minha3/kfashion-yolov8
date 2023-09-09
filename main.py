import os
import kfashion


os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    kfashion.prepare('/opt/datasets/kfashion', os.path.expanduser('~/datasets/kfashion'))
    kfashion.train(data=os.path.expanduser('~/datasets/kfashion/kfashion.yaml'), epochs=7)
