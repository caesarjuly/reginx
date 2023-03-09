# Reginx
Reginx is short for recommendation engine X. I plan to build most part of modern recommendation engine from scratch, initial plan including:
1. Popular machine learning models like CF, FM, XGBoost, TwoTower, W&D, DeepFM, SASRec, Bert4Rec, Transformer, etc.
2. Online inference service written by Golang, including candidate generator, ranking and re-ranking layers
3. Feature engineering and preprocessing, including both online and offline part
4. Diversity approaches, like MMR, DPP
5. Deduplication approaches, like LSH or BloomFilter
6. Training data pipeline
7. Model registry, monitoring and versioning
