#!/bin/bash
set -e

echo "Checking if the bucket '$BUCKET_NAME_TERRAFORM' already exists..."
REGION="europe-west1"
BUCKET_BACKEND="${TF_BACKEND_BUCKET}"
BUCKET_MODELS="${TF_VAR_models_bucket}"

for BUCKET in "$BUCKET_BACKEND" "$BUCKET_MODELS"
do
  echo "Checking if the bucket '$BUCKET' already exists..."
  if gsutil ls -b "gs://$BUCKET" >/dev/null 2>&1; then
    echo "The GCS bucket '$BUCKET' already exists."
  else
    echo "Creating GCS bucket '$BUCKET' in region '$REGION'..."
    gsutil mb -l "$REGION" "gs://$BUCKET"
    echo "Enabling versioning..."
    gsutil versioning set on "gs://$BUCKET"
    echo "Bucket '$BUCKET' successfully created with versioning enabled."
  fi
done