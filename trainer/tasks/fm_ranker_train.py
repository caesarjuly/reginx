import tensorflow as tf

from trainer.tasks.ranker_train import RankerTrain


class FMRankerTrain(RankerTrain):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

    def save(self) -> None:
        # https://github.com/tensorflow/tensorflow/issues/37439#issuecomment-596916472
        data = self.test_data.take(20).batch(20)
        for i in data.as_numpy_iterator():
            print(i["user_rating"])
        result = self.model.predict(data)
        print([i[0] for i in result])
        # Save the index.
        # https://github.com/tensorflow/models/issues/8990#issuecomment-1069733488
        self.model.save(f"/tmp/{self.hparams.model_dir}/main")
        # save linear models
        linear_user, linear_item = self.model.get_models()
        linear_user.save(f"/tmp/{self.hparams.model_dir}/linear_user")
        linear_item.save(f"/tmp/{self.hparams.model_dir}/linear_item")
        # save fm embeddings
        user_emb, item_emb = self.model.get_fm_emb()
        user_emb.save(f"/tmp/{self.hparams.model_dir}/user_emb")
        item_emb.save(f"/tmp/{self.hparams.model_dir}/item_emb")
