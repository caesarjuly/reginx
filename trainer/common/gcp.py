import glob
import os
from pathlib import Path
import shutil
from google.cloud import storage

PROJECT_ID = "fourth-blend-378118"
LOCATION = "us-central1"
BUCKET_URI = "gs://bucket-quickstart_fourth-blend-378118"
BUCKET_NAME = "bucket-quickstart_fourth-blend-378118"
IMAGE_URI = "us-docker.pkg.dev/fourth-blend-378118/tfrs/tfrs-custom-trainer"
MACHINE_TYPE = "e2-standard-4"
TENSORBOARD_INSTANCE_NAME = (
    "projects/977229647245/locations/us-central1/tensorboards/3837603444194017280"
)
SERVICE_ACCOUNT = "977229647245-compute@developer.gserviceaccount.com"

GCS_CLIENT = storage.Client()


def upload_blob(source_file_name, dest_bucket_name, dest_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    bucket = GCS_CLIENT.bucket(dest_bucket_name)
    blob = bucket.blob(dest_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(
        source_file_name, if_generation_match=generation_match_precondition
    )

    print(f"File {source_file_name} uploaded to {dest_blob_name}.")


def upload_from_directory(
    directory_path: str, dest_bucket_name: str, dest_blob_name: str
):
    rel_paths = glob.glob(directory_path + "/**", recursive=True)
    bucket = GCS_CLIENT.get_bucket(dest_bucket_name)
    for local_file in rel_paths:
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def download_from_directory(
    src_bucket_name: str, src_blob_name: str, dest_path: str, overwrite: bool = False
):
    # if this directory has data and not overwrite
    if os.path.isdir(dest_path):
        if os.listdir(dest_path) and not overwrite:
            return
    shutil.rmtree(dest_path)
    bucket = GCS_CLIENT.get_bucket(src_bucket_name)
    blobs = bucket.list_blobs(prefix=src_blob_name)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        name = dest_path.rstrip("/") + "/" + blob.name[len(src_blob_name) :].lstrip("/")
        file_split = name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(name)
