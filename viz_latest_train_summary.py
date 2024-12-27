import os
import matplotlib
matplotlib.use('TkAgg')  # Or 'QtAgg', depending on your environment

import matplotlib.pyplot as plt

import pandas as pd

from czii_helper.dataset import *
from czii_helper.helper import *

def get_latest_folder(parent_directory):
    # Get a list of all entries in the parent directory
    entries = [os.path.join(parent_directory, entry) for entry in os.listdir(parent_directory)]
    
    # Filter the list to include only directories
    directories = [entry for entry in entries if os.path.isdir(entry)]
    
    if not directories:
        print("No directories found.")
        return None
    
    # Find the directory with the most recent modification time
    latest_folder = max(directories, key=os.path.getmtime)
    return latest_folder


if __name__ == '__main__':

    config = dotdict(
        load_config('config.yml')
    )

    vi = config.val_interval
    latest_metrics_folder = get_latest_folder(f'{config.output_dir}/train')

    df = pd.read_csv(f'{latest_metrics_folder}/train_log.csv')


    m4_loss_train = np.poly1d(np.polyfit(df.epoch, df.loss_train, 4))
    m4_loss_val = np.poly1d(np.polyfit(df.epoch, df.loss_val, 4))

    polyline = np.linspace(0 + 1, df.epoch.to_list()[-1] + 1, 100)

    fig, ax1 = plt.subplots(3, figsize=(9,11))

    # -- graph1. train/val loss and lb score
    ax1[0].set_title('val TS_6_6')
    ax2 = ax1[0].twinx()

    ax1[0].plot(df.epoch, df.loss_train, color='blue', alpha=.2)
    ax1[0].plot(polyline, m4_loss_train(polyline), '--', color='blue', label='train loss fit')

    ax1[0].plot(df.epoch, df.loss_val, color='red', alpha=.2)
    ax1[0].plot(polyline, m4_loss_val(polyline), '--', color='red', label='val loss fit')

    ax2.plot(df.epoch, df.lb_score, color='green', label='val lb score')

    ax1[0].set_xlabel('epoch')
    ax1[0].set_ylabel('loss')
    ax2.set_ylabel('val lb score')

    ax2.set_ylim([0, 1])

    ax1[0].legend(loc='upper left')
    ax2.legend(loc='upper right')

    # -- graph2. lr
    ax1[1].set_title(f'lr scheduler (start from {config.lr})')
    ax1[1].plot(df.epoch, df.lr)
    ax1[1].set_xlabel('epoch')
    ax1[1].set_ylabel('lr')

    # -- graph3. fbeta
    ax1[2].set_title('fbeta4 for each class')
    ax1[2].plot(df.epoch, df['fbeta4_a-fer'], label='a-fer(1)')
    ax1[2].plot(df.epoch, df['fbeta4_b-amy'], label='b-amy(0)')
    ax1[2].plot(df.epoch, df['fbeta4_b-gal'], label='b-gal(2)')
    ax1[2].plot(df.epoch, df['fbeta4_ribo'], label='ribo(1)')
    ax1[2].plot(df.epoch, df['fbeta4_thyr'], label='thyr(2)')
    ax1[2].plot(df.epoch, df['fbeta4_vlp'], label='vlp(1)')

    ax1[2].set_xlabel('epoch')
    ax1[2].set_ylabel('fbeta4')

    ax1[2].legend(loc='upper left')
    # --

    fig.tight_layout()

    plt.grid()
    plt.savefig(f'{latest_metrics_folder}/summary.png') 
    plt.show()



