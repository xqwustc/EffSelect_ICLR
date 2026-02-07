# EffSelect: Efficient Feature Value Selection for Deep Recommender Systems with Mini-Batch Training 🚀

![KDD'26](https://img.shields.io/badge/KDD'26-Submission-blue)

This repository contains the official implementation of the KDD'26 submission:

**EffSelect: Efficient Feature Value Selection for Deep Recommender Systems with Mini-Batch Training**

EffSelect addresses two critical issues in existing feature selection methods:
1. **Low efficiency** ⏳
2. **Bad robustness** 🛡️

For detailed modifications on feature values, please refer to the `fuxictr/pytorch/layers/embeddings` directory in this repository.

## 🛠️ Code Usage (Example for DCN)

### 1. Mini-Batch Training

To train using mini-batch, follow these steps:

1. **Configure the training ratio**:
   - In the configuration file, specify `train_ratio: xxx` to set the proportion of mini-batch to be used.

2. **Run the pre-training script**:
   ```bash
   python /DCN/DCN_torch/run_expid_prefeat.py --expid <pre-training expid> --gpu 0
   ```
    This will generate the importance scores for each feature value, which will be saved in checkpoint/<expid>.

    The output file will have a header similar to:
    | feature_name | index | score |
    |--------------|-------|-------|

### 2. Re-training
Next, execute the corresponding run.sh file for your dataset. The dataset name can be one of the following: {aliccp, avazu, criteo, ipinyou}.
For example:

```bash
./DCN/DCN_torch/run_sh/<dataset_name>.sh
```

where `<dataset_name>` can be:
- aliccp
- avazu
- criteo
- ipinyou

### Appendix: possible configs

#### DCN_feat_few_avazu:
```yaml
model: DCN
dataset_id: avazu_x4
loss: 'binary_crossentropy'
metrics: [ 'logloss', 'AUC' ]
task: binary_classification
optimizer: adam
learning_rate: 1.0e-3
embedding_regularizer: 0
net_regularizer: 0
batch_size: 10000  
embedding_dim: 8
dnn_hidden_units: [ 1024, 512, 256 ]
dnn_activations: relu
num_cross_layers: 3
net_dropout: 0
batch_norm: False
epochs: 100
shuffle: True
seed: 20242025
monitor: { 'AUC': 1, 'logloss': -1 }
monitor_mode: 'max'
autofeat_mode: 'table'
need_pretrain: False
force_pretrain: False # force pretrain
score_mode: 'abs'
baseline: 'mean'
train_ratio: 0.05
```

#### DCN_feat_retrain_avazu:
```yaml
model: DCN
dataset_id: avazu_x4
loss: 'binary_crossentropy'
metrics: [ 'logloss', 'AUC' ]
task: binary_classification
optimizer: adam
learning_rate: 1.0e-3
embedding_regularizer: 0
net_regularizer: 0
batch_size: 10000 
embedding_dim: 8
dnn_hidden_units: [ 1024, 512, 256 ]
dnn_activations: relu
num_cross_layers: 3
net_dropout: 0
batch_norm: False
epochs: 100
shuffle: True
seed: 20262027
monitor: { 'AUC': 1, 'logloss': -1 }
monitor_mode: 'max'
autofeat_mode: 'retrain'
need_pretrain: True
keep_ratio: 0.1
score_version: 6
```
