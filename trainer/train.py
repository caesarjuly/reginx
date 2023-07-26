import argparse
import os
import tensorflow as tf

from google.cloud import aiplatform
from google.cloud.aiplatform.training_utils import cloud_profiler
from trainer.common.gcp import (
    BUCKET_URI,
    LOCATION,
    PROJECT_ID,
)
from trainer.util.tools import ObjectDict, prepare_hparams
from trainer.tasks import task_factory

# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices("GPU")
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)
# print("Compute dtype: %s" % policy.compute_dtype)
# print("Variable dtype: %s" % policy.variable_dtype)


def run(config: str, hparams: ObjectDict) -> None:
    with aiplatform.start_execution(
        schema_title="system.ContainerExecution", display_name=config
    ) as execution:
        train_artifact = aiplatform.Artifact.create(
            schema_title="system.Dataset",
            display_name=f"{config}_train",
            uri=f"{BUCKET_URI}/{hparams.train_data}",
        )
        test_artifact = aiplatform.Artifact.create(
            schema_title="system.Dataset",
            display_name=f"{config}_test",
            uri=f"{BUCKET_URI}/{hparams.test_data}",
        )
        execution.assign_input_artifacts([train_artifact, test_artifact])
        if "AIP_TENSORBOARD_LOG_DIR" in os.environ:
            hparams.log_dir = os.environ["AIP_TENSORBOARD_LOG_DIR"]
        if "AIP_MODEL_DIR" in os.environ:
            hparams.model_dir = os.environ["AIP_MODEL_DIR"]
        print("Model saved at " + hparams.model_dir)
        aiplatform.log_params(hparams)

        task = task_factory.get_class(hparams.task_name)(hparams)
        result = task.run()
        aiplatform.log_metrics(result)
        task.save()

        model_artifact = aiplatform.Artifact.create(
            schema_title="system.Model",
            display_name=f"{config}_model",
            uri=hparams.model_dir,
        )
        execution.assign_output_artifacts([model_artifact])

        aiplatform.log_metrics(
            {"lineage": execution.get_output_artifacts()[0].lineage_console_uri}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, help="config name")
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        default=False,
        action="store_true",
        help="force delete experiment",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        dest="exp",
        default="exp",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "-r", "--run", dest="run", default="run", type=str, help="run name"
    )
    args = parser.parse_args()

    cloud_profiler.init()

    if args.force:
        experiment_run = aiplatform.ExperimentRun(
            run_name=args.run,
            experiment=args.exp,
            project=PROJECT_ID,
            location=LOCATION,
        )
        experiment_run.delete(delete_backing_tensorboard_run=True)

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=BUCKET_URI,
        experiment=args.exp,
    )
    aiplatform.start_run(args.run)

    config_file = f"trainer/configs/{args.config}.yaml"
    hparams = prepare_hparams(config_file)
    run(args.config, hparams)
    aiplatform.end_run()
