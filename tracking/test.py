import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker, trackerlist

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, test_epoch=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot, tnl2k).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]
    
    trackers = trackerlist(tracker_name, tracker_param, dataset_name, run_id)
    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    
    def parse_run_id(run_id):
        try:
            return [int(x) for x in run_id.split(',')]
        except ValueError:
            return [int(run_id)]
    
    parser.add_argument('--tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--run_id', type=parse_run_id, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--test_epoch', type=int, default=None)
    

    args = parser.parse_args()

    seq_name = args.sequence
    
    # try:
    #     seq_name = int(args.sequence)
    # except:
    #     seq_name = args.sequence

    if args.run_id is None and args.test_epoch is None:
        print("Please specify either run_id or test_epoch")
    if args.run_id is None and args.test_epoch is not None:
         args.run_id = args.test_epoch
    if args.test_epoch is None and args.run_id is not None:
         args.test_epoch = args.run_id
    
    print("dataset_name is %s" %args.dataset_name)
    print("tracker_param is %s" %args.tracker_param)
    
    run_tracker(args.tracker_name, args.tracker_param, args.run_id, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, test_epoch=args.test_epoch)
if __name__ == '__main__':
    main()
