import numpy as np
import pandas as pd
from my_dataset import MultiDataset, RegressionDataset


def scatter_path_dict(path_data, size, rank):
    all_lst = []

    for row, item in path_data.iterrows():
        path, num = item['path'], int(item['num'])
        all_lst.extend([[path, i] for i in range(num)])
    all_lst = np.array(all_lst, dtype=object)
    # all_lst = np.random.permutation(all_lst)
    all_lst = all_lst[int(len(all_lst) / size) * rank:int(len(all_lst) / size) * (rank + 1):]
    path_dict = {}
    for item in all_lst:
        p, n = item[0], item[1]
        if not p in path_dict.keys():
            path_dict.update({p: [n]})
        else:
            path_dict[p].append(n)
    return path_dict


def scatter_path_array(path_data, size, rank):
    all_lst = []

    for row, item in path_data.iterrows():
        path, num = item['path'], int(item['num'])
        all_lst.extend([[path, i] for i in range(num)])
    all_lst = np.array(all_lst, dtype=object)
    all_lst = np.random.permutation(all_lst)
    all_lst = all_lst[int(len(all_lst) / size) * rank:int(len(all_lst) / size) * (rank + 1):]
    return all_lst[:, 0], all_lst[:, 1]


class Dataproc():
    def __init__(self, size, rank, config):
        self.verbose = 10 if rank == 0 else 0
        self.config = config
        np.random.seed(7)

        path_data = pd.read_csv(self.config['csv_path'], index_col=0)
        path_data = path_data[path_data.apply(lambda x: int(x['dir_name'][2:5]) < 759, axis=1)]
        path_data = path_data.reindex(np.random.permutation(path_data.index)).reset_index(drop=True)

        rate = self.config['train_rate']
        protein_name_list = np.random.permutation(path_data['dir_name'].unique())
        self.protein_name = {'train': protein_name_list[:int(len(protein_name_list) * rate)],
                             'hoge_test.py': protein_name_list[int(len(protein_name_list) * rate):]}

        self.data_dict = {}
        train_data = path_data.ix[path_data['dir_name'].isin(self.protein_name['train'])]
        test_data = path_data.ix[path_data['dir_name'].isin(self.protein_name['test'])]
        native_data = train_data[train_data['label_list'] == 1]
        other_data = train_data[train_data['label_list'] != 1]
        # random
        # other_data = other_data.groupby('dir_name').apply(lambda x: x.sample(frac=self.config['data_frac']))
        # upper
        # other_data = other_data.groupby('dir_name').apply(
        #     lambda x: x.sort_values('label_list')[int(x.shape[0] * (1 - self.config['data_frac'])):x.shape[0]])
        # lower
        other_data = other_data.groupby('dir_name').apply(
            lambda x: x.sort_values('label_list')[:int(x.shape[0] * self.config['data_frac'])])

        train_data = pd.concat([native_data, other_data])

        path, index = scatter_path_array(path_data=train_data, size=size, rank=rank)
        self.data_dict.update({'train': {'path': path, 'index': index}})
        path, index = scatter_path_array(path_data=test_data, size=size, rank=rank)
        self.data_dict.update({'test': {'path': path, 'index': index}})

        if self.config['scop']:
            scop_path_data = pd.read_csv('./scop_e_40_path_list.csv', index_col=0)
            path, index = scatter_path_array(
                path_data=scop_path_data, size=size, rank=rank)

            self.data_dict['train']['path'] = np.append(self.data_dict['train']['path'], path)
            self.data_dict['train']['index'] = np.append(self.data_dict['train']['index'], index)

    def get_protein_name_dict(self):
        return self.protein_name

    def get_classification_dataset(self, key):
        dataset = MultiDataset(path=self.data_dict[key]['path'], index=self.data_dict[key]['index'],
                               config=self.config)
        return dataset

    def get_regression_dataset(self, key):
        dataset = RegressionDataset(path=self.data_dict[key]['path'], index=self.data_dict[key]['index'],
                                    config=self.config)
        return dataset


if __name__ == '__main__':
    import json


    def hinted_tuple_hook(obj):
        if '__tuple__' in obj:
            return tuple(obj['items'])
        else:
            return obj


    f = open('./config_multi_label.json', 'r')
    config = json.load(f, object_hook=hinted_tuple_hook)[2]
    d = Dataproc(size=32, rank=0, config=config)
    print(len(d.data_dict['train']['path']))
