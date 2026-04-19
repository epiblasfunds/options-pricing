#!/bin/bash
set -e

REGION="europe-west1"
BUCKET_BACKEND="${TF_BACKEND_BUCKET}"

echo "Checking if the backend bucket '$BUCKET_BACKEND' already exists..."
if gsutil ls -b "gs://$BUCKET_BACKEND" >/dev/null 2>&1; then
  echo "The GCS backend bucket '$BUCKET_BACKEND' already exists."
else
  echo "Creating GCS backend bucket '$BUCKET_BACKEND' in region '$REGION'..."
  gsutil mb -l "$REGION" "gs://$BUCKET_BACKEND"
  echo "Enabling versioning..."
  gsutil versioning set on "gs://$BUCKET_BACKEND"
  echo "Backend bucket '$BUCKET_BACKEND' successfully created with versioning enabled."
fi