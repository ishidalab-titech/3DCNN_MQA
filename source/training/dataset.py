from chainer.dataset import DatasetMixin
import six
import numpy as np
import os
from scipy.sparse import load_npz

from numba import jit


@jit
def sp_noise(data, occur_rate=0.9, sp_rate=0.5):
    noise = np.random.uniform(0, 1, data.shape)
    for i, p in enumerate(noise):
        noise[i] = data[i] if p < occur_rate else 1 if p < occur_rate + (1 - occur_rate) * sp_rate else 0
    return noise
    # noise = [data[i] if p < occur_rate else 1 if p < occur_rate + (1 - occur_rate) * sp_rate else 0 for i, p in
    #          enumerate(noise)]
    # noise = np.array(noise)


from numba import jit


class RegressionDataset(DatasetMixin):
    def __init__(self, path, index, config):
        super().__init__()
        self.path = path
        self.index = index
        self.config = config

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.path)

    @jit
    def get_data(self, path, index):
        if len(str(path).split('/')[0]) == 2:
            voxel_path = os.path.join(self.config['scop_path'], path)
            voxel = load_npz(voxel_path)[index]
            voxel = np.reshape(voxel.toarray(), [18, 30, 30, 30])[:self.config['channel']].astype(np.float32)
            data_width = voxel.shape[1]
            b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
            voxel = voxel[:, b:e, b:e, b:e]
            local_label = []
            for label_name in self.config['label']:
                local_label.append(1)
        else:
            label_path = os.path.join(self.config['label_path'], path)
            voxel_path = os.path.join(self.config['voxel_path'], path)
            voxel = load_npz(voxel_path)[index]
            voxel = np.reshape(voxel.toarray(), [18, 30, 30, 30])[:self.config['channel']].astype(np.float32)
            data_width = voxel.shape[1]
            b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
            voxel = voxel[:, b:e, b:e, b:e]
            #label = np.load(label_path)[self.config['protein']].tolist()
            label = np.load(label_path)
            local_label = []
            for label_name in self.config['label']:
                local_label.append(label[label_name][index])
        return voxel, local_label

    def get_example(self, i):
        path = self.path[i]
        index = self.index[i]
        voxel, label = self.get_data(path=path, index=index)
        threshold = self.config['local_threshold']
        label = np.array(label, dtype=np.float32)
        return voxel, label


class MultiDataset(DatasetMixin):
    def __init__(self, path, index, config):
        super().__init__()
        self.path = path
        self.index = index
        self.config = config

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.path)

    @jit
    def get_data(self, path, index):
        if len(str(path).split('/')[0]) == 2:
            voxel_path = os.path.join(self.config['scop_path'], path)
            voxel = load_npz(voxel_path)[index]
            voxel = np.reshape(voxel.toarray(), [14, 30, 30, 30])[:self.config['channel']].astype(np.float32)
            data_width = voxel.shape[1]
            b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
            voxel = voxel[:, b:e, b:e, b:e]
            local_label = []
            for label_name in self.config['label']:
                local_label.append(1)
        else:
            label_path = os.path.join(self.config['label_path'], path)
            voxel_path = os.path.join(self.config['voxel_path'], path)
            voxel = load_npz(voxel_path)[index]
            voxel = np.reshape(voxel.toarray(), [14, 30, 30, 30])[:self.config['channel']].astype(np.float32)
            data_width = voxel.shape[1]
            b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
            voxel = voxel[:, b:e, b:e, b:e]
            #label = np.load(label_path)[self.config['protein']].tolist()
            label = np.load(label_path)
            local_label = []
            for label_name in self.config['label']:
                local_label.append(label[label_name][index])
        return voxel, local_label

    def get_example(self, i):
        path = self.path[i]
        index = self.index[i]
        voxel, label = self.get_data(path=path, index=index)
        threshold = self.config['local_threshold']
        label = [1 if i > threshold else 0 for i in np.array(label, dtype=np.float32)]
        label = np.array(label)
        return voxel, label

