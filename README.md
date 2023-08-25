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
I also write a blog explaining the models and put the relevant links in the table below.


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
