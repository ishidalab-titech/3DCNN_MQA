#hoge-MQA

## Requirement
- python==3.6
- Chainer==4.1.0
- htmd==1.13
- numpy==1.14
- scipy==1.10

## Preparation
Download pre-trained model from [here](http://www.cb.cs.titech.ac.jp/~sato/bmc2018/pretrained_model.npz).

## Usage
```bash
# Predict Model Quality Score from PDB (using GPU)
python predict_from_pdb.py -i pdb_path -m pre-trained_model_path -g 0

# Predict Model Quality Score from PDB (using CPU)
python predict_from_pdb.py -i pdb_path -m pretrained_model_path

# Preprocess input data from PDB
python make_data.py -i pdb_path -o output_path
```

