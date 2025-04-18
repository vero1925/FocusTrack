import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
from tracking.plot_utils import read_avg_overlap_all_from_eval_data_pkl, draw_comp_plot
import argparse
import pickle
import torch

import warnings
import numpy as np
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

trackers = []

def parse_run_id(run_id):
    try:
        return [int(x) for x in run_id.split(',')]
    except ValueError:
        return [int(run_id)]

parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_name', type=str, help='Name of the tracker.')
parser.add_argument('--tracker_param', type=str, help='Name of config file.')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset.')
parser.add_argument('--run_id', type=parse_run_id, default=None, help='The run id.')
args = parser.parse_args()


def define_tracker(name, param, dataset_name, run_id):
    arg_list = [[dataset_name, '{}_{}'.format(param,id), name] for id in run_id]
    
    if run_id is not None:
        display_name = ["{}_{}_{}".format(name, dataset_name, id) for id in run_id]
    else:
        display_name = "{}_{}".format(name, dataset_name)

    tracker = trackerlist(name=name, parameter_name=param, dataset_name=dataset_name,
                          run_ids=run_id, display_names=display_name)

    return tracker, arg_list

dataset = get_dataset(args.dataset_name)
tracker_param = args.tracker_param

tracker_info = [
        {"name": args.tracker_name,     "param": args.tracker_param},
    ]

tracker_auc = []
tracker_list = []
for info in tracker_info:
    tracker, arg_list = define_tracker(info["name"], info["param"], args.dataset_name, args.run_id)
    for trk, arg in zip(tracker, arg_list):
        print('\n\nanalysising tracker:{}'.format(trk.display_name))
        auc_per_seq, overlap_pre_frame = print_results([trk], dataset, arg, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))   # print_results(trackers, dataset, arg_list, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
        
        tracker_auc.append(auc_per_seq)
        arg[2]='droptrack'
        tracker_list.append(arg)

def read_from_eval_data_pkl(eval_data_path):
    with open(eval_data_path, 'rb') as fh:
        eval_data = pickle.load(fh)
        tracker_names = eval_data['trackers']
        valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
        sequence_names = eval_data['sequences']
        err_overlap = eval_data['err_overlap']
        avg_overlap_all = torch.tensor(eval_data['avg_overlap_all']) * 100.0

        avg_overlap_all = avg_overlap_all[valid_sequence, :]
        
        scores_per_seqence = {k: avg_overlap_all[i] for i, k in enumerate(sequence_names)}
        
    return scores_per_seqence, err_overlap

