from google.cloud import storage
import os


def save_model_to_gcloud(model_path: str, bucket_name: str) -> None:
    """saves model to cloud"""
    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None


def download_model_from_gcloud(model_path: str, bucket_name: str) -> None:
    """downloads model from GCloud"""
    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_filename)
    blob.download_to_filename(model_path)

    print("✅ Model downloaded from GCS")

    return None


def download_nested_gcloud_folder(bucket_name, prefix, local_folder) -> None:
    """will be used for downlaoding training data from bucket"""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        destination_path = os.path.join(local_folder, blob.name.replace(prefix, ""))
        destination_folder = os.path.dirname(destination_path)

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        blob.download_to_filename(destination_path)

    print("✅ Training data downloaded from GCS")

    return None


def check_specific_model_in_bucket(model_name: str, bucket_name: str) -> bool:
    """Checks if there is specific file in GCloud bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)

    return blob.exists()
