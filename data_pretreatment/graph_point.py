# -*- coding: utf-8 -*-
import argparse
from multiprocessing.pool import Pool
import gc
import os.path as osp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import repeat

from src.checker import TopologyChecker
from src.extractor import GPCExtractor
from src.writer import load_json, WriteH5, WriteH5MFCAD, WriteH5MFCAD2


#######################################
# Calculate mean & std of attributes
#######################################

def check_zero_std(stat_data):
    std_face_attr = stat_data['std_face_attr']
    std_edge_attr = stat_data['std_edge_attr']
    if np.nonzero(std_face_attr)[0].shape[0] != len(std_face_attr) \
            or np.nonzero(std_edge_attr)[0].shape[0] != len(std_edge_attr):
        print('WARNING! has zero standard deviation.')


def find_standardization(data):
    """
    Find mean and standard deviation of face and edge attributes
    Args:
    data (list): [filename, graph_data]
    """

    all_face_attr = []
    all_edge_attr = []
    for one_sample in data:
        fn, graph = one_sample
        all_face_attr.extend(graph["graph_face_attr"])
        all_edge_attr.extend(graph["graph_edge_attr"])
    graph_face_attr = np.asarray(all_face_attr)
    graph_edge_attr = np.asarray(all_edge_attr)

    mean_face_attr = np.mean(graph_face_attr, axis=0)
    std_face_attr = np.std(graph_face_attr, axis=0)

    mean_edge_attr = np.mean(graph_edge_attr, axis=0)
    std_edge_attr = np.std(graph_edge_attr, axis=0)

    return {
            'mean_face_attr': mean_face_attr,
            'std_face_attr': std_face_attr,
            'mean_edge_attr': mean_edge_attr,
            'std_edge_attr': std_edge_attr,
        }


def matching_step_files(txt_file, pathname):
    with open(txt_file, 'r') as f:
        names = f.read().splitlines()

    matching_steps = []
    for step_file in pathname.glob("*.st*p"):
        if step_file.stem in names:
            matching_steps.append(step_file)

    return matching_steps


def initializer():
    """Ignore CTRL+C in the worker process"""

    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_one_file(arguments):
    fn, feature_schema = arguments

    topo_checker = TopologyChecker()
    extractor = GPCExtractor(fn, feature_schema, topo_checker)
    out = extractor.process()

    return [str(fn.stem), out]


def main(sub_data):
    step_path = Path(args.dataset_path + '/steps')
    label_path = Path(args.dataset_path + '/labels')
    output_path = Path(args.dataset_path + '/g_pc')
    partition_path = Path(args.dataset_path + '/dataset_partition')
    if not output_path.exists():
        output_path.mkdir()

    feature_list_path = None
    if args.feature_list is not None:
        feature_list_path = Path(args.feature_list)

    parent_folder = Path(__file__).parent.parent
    if feature_list_path is None:
        feature_list_path = parent_folder / "feature_lists/all.json"
    feature_schema = load_json(feature_list_path)
    step_files = matching_step_files(osp.join(partition_path, f'{sub_data}.txt'), step_path)

    pool = Pool(processes=args.num_workers, initializer=initializer)
    results = []
    try:
        results = list(tqdm(
            pool.imap(
                process_one_file, zip(step_files, repeat(feature_schema))),
            total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

    if args.dataset_name == 'MFCAD':
        writer = WriteH5MFCAD(osp.join(output_path, f'{sub_data}_graphs.h5'), label_path,
                              osp.join(output_path, f'{sub_data}_attr_stat.h5'))
    elif args.dataset_name == 'MFCAD2':
        writer = WriteH5MFCAD2(osp.join(output_path, f'{sub_data}_graphs.h5'), label_path,
                               osp.join(output_path, f'{sub_data}_attr_stat.h5'))
    else:
        writer = WriteH5(osp.join(output_path, f'{sub_data}_graphs.h5'), label_path,
                         osp.join(output_path, f'{sub_data}_attr_stat.h5'))
    writer.write_h5_file(results)

    attr_stat = find_standardization(results)
    check_zero_std(attr_stat)
    writer.write_attr_data(attr_stat)

    gc.collect()
    print(f"Processed {len(results)} files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")
    parser.add_argument("--feature_list", type=str, required=False,
                        help="Optional path to the feature lists")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of workers")
    args = parser.parse_args()

    for sub_dataset in ['train', 'val', 'test']:
        main(sub_dataset)
