import argparse
import os

from google.cloud import aiplatform
from google.cloud.aiplatform.training_utils import cloud_profiler
from trainer.common.gcp import (
    BUCKET_URI,
    LOCATION,
    PROJECT_ID,
)
from trainer.util.tools import ObjectDict, prepare_hparams
from trainer.tasks import task_factory


def run(hparams: ObjectDict) -> None:
    task = task_factory.get_class(hparams.task_name)(hparams)
    task.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, help="config name")
    args = parser.parse_args()

    config_file = f"trainer/configs/{args.config}.yaml"
    hparams = prepare_hparams(config_file)
    run(hparams)
