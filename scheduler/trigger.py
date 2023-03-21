import argparse
from google.cloud import aiplatform
from trainer.common.gcp import (
    PROJECT_ID,
    IMAGE_URI,
    BUCKET_URI,
    MACHINE_TYPE,
    TENSORBOARD_INSTANCE_NAME,
    SERVICE_ACCOUNT,
    LOCATION,
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

    job_name = "movielens"
    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=BUCKET_URI,
    )
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name, container_uri=IMAGE_URI
    )

    base_output_dir = "{}/{}".format(BUCKET_URI, job_name)
    training_args = [
        f"--config={args.config}",
        f"--experiment={args.exp}",
        f"--run={args.run}",
    ]
    if args.force:
        training_args.append("-f")
    print(training_args)

    job.run(
        args=training_args,
        replica_count=1,
        machine_type=MACHINE_TYPE,
        base_output_dir=base_output_dir,
        tensorboard=TENSORBOARD_INSTANCE_NAME,
        service_account=SERVICE_ACCOUNT,
    )
