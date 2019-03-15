from chainer.serializers import load_npz
from scipy import sparse
from chainer import Variable
from pathlib import Path
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import os, json, sys
from train import get_model
from chainer import function
from joblib import load, dump
from tqdm import tqdm


def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


#
def load_model(result_path):
    config_path = result_path / 'config.json'
    log_path = result_path / 'log'
    log = json.load(open(log_path, 'r'))
    config = json.load(open(config_path, 'r'), object_hook=hinted_tuple_hook)
    model = get_model(config=config, comm=None, predict=True)
    max_auc = -1
    max_iteration = -1
    key = 'val/main/roc_auc_local_lddt'
    for e_data in log:
        if key in e_data.keys():
            if max_auc < e_data[key]:
                max_auc = e_data[key]
                max_iteration = e_data['iteration']
    model_path = result_path / 'snapshot_iter_{}'.format(max_iteration)
    obj_path = 'updater/model:main/predictor/'
    load_npz(model_path, model, obj_path)
    model.to_gpu(0)
    box_width = config['box_width']
    channel = config['channel']
    label_name = config['label']
    return model, box_width, channel, label_name


def predict_file(voxel_path, out_path, model, channel, box_width, label_name, device):
    data = sparse.load_npz(voxel_path)
    data = np.reshape(data.toarray(), [data.shape[0], 14, 30, 30, 30])[:, :channel].astype(np.float32)
    data_width = data.shape[2]
    b, e = (data_width - box_width) // 2, (data_width + box_width) // 2
    data = data[:, :, b:e, b:e, b:e]
    batch_size = 16
    i = 0
    out_data = {}
    out_pred_score = np.array([]).reshape(0, len(label_name))
    while i * batch_size < data.shape[0]:
        voxel = data[i * batch_size:(i + 1) * batch_size]
        voxel = Variable(voxel)
        if device >= 0:
            voxel.to_gpu()
        with function.no_backprop_mode(), chainer.using_config('train', False):
            pred_score = F.sigmoid(model(voxel))
        pred_score = chainer.cuda.to_cpu(pred_score.data)
        out_pred_score = np.vstack([out_pred_score, pred_score])
        i += 1
    for index, i in enumerate(label_name):
        out_data.update({i: out_pred_score[:, index]})
    np.savez(out_path, **out_data)


def make_dir(in_path: Path, out_path: Path):
    [(out_path / i.relative_to(in_path)).mkdir(exist_ok=True, parents=True) for i in in_path.glob('**/')]


def get_path(in_path: Path, out_path: Path):
    in_path_set = set([str(i.relative_to(in_path)) for i in in_path.glob('**/*.npz')])
    out_path_set = set([str(i.relative_to(out_path)) for i in out_path.glob('**/*.npz')])
    path_list = list(in_path_set - out_path_set)
    return path_list


def main(rank, size, device):
    np.random.seed(0)
    derevyanko_in = Path('/gs/hs0/tga-ishidalab/sato/discriptor/test_set_georgy/voxel_30_1_18_bool/scwrl4')
    derevyanko_out = Path('/gs/hs0/tga-ishidalab/sato/evaluate/test_set_georgy/voxel_30_1_18_bool/scwrl4')

    general_test_in = Path('/gs/hs0/tga-ishidalab/sato/discriptor/test_set_not_H/voxel_30_1_18')
    general_test_out = Path('/gs/hs0/tga-ishidalab/sato/evaluate/test_set_not_H/voxel_30_1_18/smorms3_1_random')

    surface_in = Path('/gs/hs0/tga-ishidalab/sato/discriptor/test_set_georgy/scwrl4/align_native/casp11_stage_2')
    surface_out = Path('/gs/hs0/tga-ishidalab/sato/evaluate/test_set_georgy/align_native/casp11_stage_2')

    in_path = surface_in
    out_path = surface_out
    make_dir(in_path=in_path, out_path=out_path)
    path_list = get_path(in_path=in_path, out_path=out_path)
    dump(path_list, './path_predict.pkl')
    path_list = load('./path_predict.pkl')
    path_list = np.array_split(np.random.permutation(path_list), size)[rank]
    pbar = tqdm(total=len(path_list))
    result_path = Path('/gs/hs0/tga-ishidalab/sato/source/3dcnn/training/0/smorms3_1_random')
    model, box_width, channel, label_name = load_model(result_path=result_path)
    if device >= 0:
        model.to_gpu(device=device)

    for path in path_list:
        d = in_path / path
        o = out_path / path
        predict_file(voxel_path=d, model=model, out_path=o, channel=channel, box_width=box_width,
                     label_name=label_name, device=device)
        pbar.update(1)


if __name__ == '__main__':
    rank = int(sys.argv[1])
    size = int(sys.argv[2])
    main(rank=rank, size=size, device=0)
