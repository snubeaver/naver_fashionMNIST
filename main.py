from threading import Thread
import sys
sys.path.append('../')
from plotter import plot
from cnn import calc
from utils import recursive_mkdir
import os


epochs = 50
norm='maxmin'
gpus = 4
data = 'fashion_mnist'
configs = {
    'optimizer': ['SGD', 'Adam'],
    'lr': [1e-3, 1e-4],
    'batch_size': [128, 1024]
}

def calc_and_plot(data, optimizer, norm, lr, epochs, iteration, batch_size):
    data_dir = f'{batch_size}-weights-{optimizer}-{lr}-{norm}'
    calc(data, optimizer, norm, lr, epochs, iteration, batch_size)
    plot_parent_dir = './plots'
    for i in range(0, 4):
        data_path = f'{data_dir}/store-{i}.pkl'
        plot_path = f'{plot_parent_dir}/{data_dir}/layer{i}'
        recursive_mkdir(plot_path)
        plot(plot_path, data_path)

def main():
    iteration = 0
    for batch_size in configs['batch_size']:
        batch_size = batch_size * gpus
        for optimizer in configs['optimizer']:
                for lr in configs['lr']:
                    iteration += 1
                    calc_and_plot(data, optimizer, norm, lr, epochs, iteration, batch_size)

if __name__ == '__main__':
    main()