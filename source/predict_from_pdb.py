import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import json
import argparse
from chainer.function import no_backprop_mode
from chainer.sequential import Sequential
from chainer.serializers import load_npz
from chainer import Variable
from functools import partial
from preprocessing.make_voxel import get_voxel


def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


def get_model(config):
    model = Sequential()
    layers = config['model']
    for layer in layers:
        name = layer['name']
        parameter = layer['parameter']
        if name[0] == 'L':
            add_layer = eval(name)(**parameter)
        elif name[0] == 'F':
            if len(parameter) == 0:
                add_layer = partial(eval(name))
            else:
                add_layer = partial(eval(name), **parameter)
        model.append(add_layer)
    return model


def get_predict_value(data, model, gpu):
    data = data.astype(np.float32)
    batch_size = 16
    i = 0
    out_list = []
    while i * batch_size < data.shape[0]:
        voxel = data[i * batch_size:(i + 1) * batch_size]
        voxel = Variable(voxel)
        if gpu >= 0:
            voxel.to_gpu()
        with no_backprop_mode(), chainer.using_config('train', False):
            pred_score = F.sigmoid(model(voxel))
        pred_score = chainer.cuda.to_cpu(pred_score.data).ravel()
        out_list.extend(pred_score)
        i += 1
    return np.array(out_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict ')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input_path', '-i', help='Input data path')
    parser.add_argument('--reference_fasta_path', '-r', help='Reference FASTA Sequence path')
    parser.add_argument('--model_path', '-m', help='Pre-trained model path')
    args = parser.parse_args()
    model = get_model(json.load(open('./config.json', 'r'), object_hook=hinted_tuple_hook))
    load_npz(file=args.model_path, obj=model, path='updater/model:main/predictor/')
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    data, resname, resid = get_voxel(input_path=args.input_path, target_path=args.reference_fasta_path,
                                     buffer=28, width=1)
    predict_value = get_predict_value(data=data, model=model, gpu=args.gpu)

    print('Input Data Path : {}'.format(args.input_path))
    print('Model Quality Score : {}'.format(np.mean(predict_value)))
    print('Resid\tResname\tScore')
    for i in range(len(resname)):
        print('{}\t{}\t{:.5f}'.format(resid[i], resname[i], predict_value[i]))
