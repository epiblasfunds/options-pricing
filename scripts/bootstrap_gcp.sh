#!/bin/bash
set -euo pipefail

REGION="europe-west1"
BUCKET_BACKEND="${TF_BACKEND_BUCKET:-}"

if [[ -z "$BUCKET_BACKEND" ]]; then
  echo "TF_BACKEND_BUCKET is not set."
  exit 1
fi

echo "Checking if the backend bucket '$BUCKET_BACKEND' already exists..."
BUCKET_CHECK_OUTPUT=""
if BUCKET_CHECK_OUTPUT=$(gcloud storage buckets describe "gs://$BUCKET_BACKEND" 2>&1); then
  echo "The GCS backend bucket '$BUCKET_BACKEND' already exists."
else
  if echo "$BUCKET_CHECK_OUTPUT" | grep -qiE "NOT_FOUND|BucketNotFoundException|Not Found|404"; then
    echo "Creating GCS backend bucket '$BUCKET_BACKEND' in region '$REGION'..."
    gcloud storage buckets create "gs://$BUCKET_BACKEND" --location="$REGION"
    echo "Enabling versioning..."
    gcloud storage buckets update "gs://$BUCKET_BACKEND" --versioning
    echo "Backend bucket '$BUCKET_BACKEND' successfully created with versioning enabled."
  elif echo "$BUCKET_CHECK_OUTPUT" | grep -qiE "AccessDenied|PERMISSION_DENIED|storage\.buckets\.get|storage\.buckets\.create|403"; then
    echo "Insufficient permissions for backend bucket bootstrap."
    echo "Required at project level: storage.buckets.get and storage.buckets.create."
    echo "Grant roles/storage.admin to the GitHub Actions service account or pre-create the bucket."
    echo "$BUCKET_CHECK_OUTPUT"
    exit 1
  else
    echo "Unexpected error while checking bucket '$BUCKET_BACKEND':"
    echo "$BUCKET_CHECK_OUTPUT"
    exit 1
  fi
fi