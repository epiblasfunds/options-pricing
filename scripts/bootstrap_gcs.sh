#!/bin/bash
set -e

REGION="europe-west1"
BUCKET_NAME_TERRAFORM="options-pricing-explainability-tfstate"

echo "Checking if the bucket '$BUCKET_NAME_TERRAFORM' already exists..."

if gsutil ls -b "gs://$BUCKET_NAME_TERRAFORM" >/dev/null 2>&1; then
  echo "The GCS bucket '$BUCKET_NAME_TERRAFORM' already exists."
else
  echo "Creating GCS bucket '$BUCKET_NAME_TERRAFORM' in region '$REGION'..."

  gsutil mb -l "$REGION" "gs://$BUCKET_NAME_TERRAFORM"

  echo "Enabling versioning..."
  gsutil versioning set on "gs://$BUCKET_NAME_TERRAFORM"

  echo "Bucket successfully created with versioning enabled."
fi