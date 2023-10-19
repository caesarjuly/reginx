# Reginx
Reginx is short for recommendation engine X. I plan to build most parts of modern recommendation engine from scratch.  
Initial plan including:
1. Popular machine learning models like CF, FM, XGBoost, TwoTower, W&D, DeepFM, DCN, MaskNet, SASRec, Bert4Rec, Transformer, etc.
2. Online inference service written by Golang, including candidate generator, ranking and re-ranking layers
3. Feature engineering and preprocessing, including both online and offline part
4. Diversity approaches, like MMR, DPP
5. Deduplication approaches, like LSH or BloomFilter
6. Training data pipeline
7. Model registry, monitoring and versioning

## Supported models  
Tensorflow 2 and Google Cloud is used for model training and performance tracking. The conda environment config is [here](https://github.com/caesarjuly/reginx/tree/master/environment).  
I have a personal [blog](https://happystrongcoder.substack.com/) in substack explaining the models and I put the corresponding links in the table below.


| Model  | Paper | Code | Blog |
| ------------- | ------------- | ------------- | ------------- |
| Factorization Machines  | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/fm.py) | [Post](https://happystrongcoder.substack.com/p/from-fm-to-deepfm-the-almighty-factorization) |
| DeepFM  | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/deepfm.py) | [Post](https://happystrongcoder.substack.com/p/from-fm-to-deepfm-the-almighty-factorization) |
| XDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/wide_and_deep.py)| [Post](https://happystrongcoder.substack.com/p/xdeepfm-combining-explicit-and-implicit) |
| AutoInt  | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/autoint.py) | [Post](https://happystrongcoder.substack.com/p/autoint-automatic-feature-interaction) |
| DCN  | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/dcn.py) | [Post](https://happystrongcoder.substack.com/p/deep-and-cross-network-for-ad-click) |
| DCN V2 | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/pdf/2008.13535.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/dcn_v2.py) | [Post](https://happystrongcoder.substack.com/p/dcn-v2-improved-deep-and-cross-network) |
| DLRM | [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/dlrm.py) | [Post](https://happystrongcoder.substack.com/p/deep-learning-recommendation-model) |
| FinalMLP | [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/pdf/2304.00902.pdf)  | [DualMLP](https://github.com/caesarjuly/reginx/blob/master/trainer/models/dual_mlp.py) [FinalMLP](https://github.com/caesarjuly/reginx/blob/master/trainer/models/final_mlp.py)| [Post](https://happystrongcoder.substack.com/p/finalmlp-an-enhanced-two-stream-mlp) |
| MaskNet | [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/pdf/2102.07619.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/mask_net.py)| [Post](https://happystrongcoder.substack.com/p/dive-into-twitters-recommendation-6fc) |
| TwoTower | [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/6417b9a68bd77033d65e431bdba855563066dc8c.pdf) [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/b9f4e78a8830fe5afcf2f0452862fb3c0d6584ea.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/two_tower.py)| [Post1](https://happystrongcoder.substack.com/p/two-tower-candidate-retriever-i) [Post2](https://happystrongcoder.substack.com/p/two-tower-candidate-retriever-ii) [Post3](https://happystrongcoder.substack.com/p/two-tower-candidate-retriever-iii) |
| Wide and Deep | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/wide_and_deep.py)| [Post](https://happystrongcoder.substack.com/p/wide-and-deep-learning-for-recommender) |
| Transformer | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/transformer.py)| [Post](https://happystrongcoder.substack.com/p/transformer-with-code-part-i-positional) |
| BERT | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/common/bert.py)| [Post](https://happystrongcoder.substack.com/p/a-gentle-introduction-to-bert-pre) |
| SASRec | [Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/common/sas_rec.py)| [Post](https://happystrongcoder.substack.com/p/sasrec-self-attentive-sequential) |
| BERT4REC | [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1904.06690.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/common/bert4rec.py)| [Post](https://happystrongcoder.substack.com/p/bert4rec-sequential-recommendation) |
| ESMM | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/esmm.py)| [Post](https://happystrongcoder.substack.com/p/entire-space-multi-task-model-an) |
| MMOE | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)  | [Code](https://github.com/caesarjuly/reginx/blob/master/trainer/models/mmoe.py)| [Post](https://happystrongcoder.substack.com/p/modeling-task-relationships-in-multi) |

## Local Training
Here is an example to train a two-tower model in local machine.
### Setup Conda
Setup your conda environment using the conda config [here](https://github.com/caesarjuly/reginx/tree/master/environment).
```
conda env create -f environment.yml
conda activate tf
```
Set your PYTHONPATH to the root folder of this project. Or you can add it to your bashrc:
```
export PYTHONPATH=/your_project_folder/reginx
```

### Prepare Movielens Training Data
You can run this [script](https://github.com/caesarjuly/reginx/blob/master/trainer/preprocess/movielens.py) to generate meta and training data in your local directory.
By default, it's using the [movielens-1m](https://www.tensorflow.org/datasets/catalog/movielens#movielens1m-ratings) from TensorFlow datasets.  
And copy your dataset files to your local `/tmp/train, /tmp/test, /tmp/item` folder. Notice that the TwoTower model implementation require 3 kinds of files, train files for training, test files for test and item files for mixing global negative samples.     
If you want to use your dataset other than movielens, please prepare your own dataset and save it to your local directory.

### Check Config File
There is example config [file](https://github.com/caesarjuly/reginx/blob/master/trainer/configs/movielens_candidate_retriever.yaml) for candidate-retriever training.  
If you want to use your dataset other than movielens, please prepare your own [query](https://github.com/caesarjuly/reginx/blob/master/trainer/models/features/movielens.py#L8) and [candidate](https://github.com/caesarjuly/reginx/blob/master/trainer/models/features/movielens.py#L76) embedding class.
```
model:
  temperature: 0.05
  # specify training model under models folder
  base_model: TwoTower
  # specify query embedding model under models/features folder
  query_emb: MovieLensQueryEmb
  # specify candidate embedding model under models/features folder
  candidate_emb: MovieLensCandidateEmb
  # specify the unique key for candidates
  item_id_key: movie_id

train:
  # specify task under tasks folder
  task_name: CandidateRetrieverTrain
  epochs: 1
  batch_size: 256
  mixed_negative_batch_size: 128
  learning_rate: 0.05
  train_data: movielens/data/ratings_train
  test_data: movielens/data/ratings_test
  candidate_data: movielens/data/movies
  meta_data: trainer/meta/movie_lens.json
  model_dir: trainer/saved_models/movielens_cr
  log_dir: logs
```
### Training
Simply run the script below and specify your the config file in you activated conda environment.
```
python trainer/local_train.py -c movielens_candidate_retriever  
```
By default, the training metrics show once per 1000 training steps for faster training. You can modify the setting by tuning the [steps_per_execution](https://github.com/caesarjuly/reginx/blob/master/trainer/tasks/candidate_retriever_train.py#L37) hyperparameter while compiling model.  
After the training, evaluation will be run on the test dataset. You should see metrics like:
```
391/391 [==============================] - 50s 129ms/step - factorized_top_k/top_1_categorical_accuracy: 0.0036 - factorized_top_k/top_5_categorical_accuracy: 0.0181 - factorized_top_k/top_10_categorical_accuracy: 0.0349 - factorized_top_k/top_50_categorical_accuracy: 0.1428 - factorized_top_k/top_100_categorical_accuracy: 0.2409 - loss: 1406.8086 - regularization_loss: 7.9244 - total_loss: 1414.7329
```
