#!/bin/bash
set -euo pipefail

REGION="europe-west1"
BUCKET_BACKEND="${TF_BACKEND_BUCKET:-}"
PROJECT_ID="${TF_VAR_gcp_project:-}"

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
    ACTIVE_ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
    echo "Insufficient permissions for backend bucket bootstrap."
    echo "Required at project level: storage.buckets.get and storage.buckets.create."
    echo "Authenticated account: ${ACTIVE_ACCOUNT:-unknown}"
    echo "If bucket already exists, grant bucket-level access:"
    echo "  gcloud storage buckets add-iam-policy-binding gs://$BUCKET_BACKEND --member=serviceAccount:$ACTIVE_ACCOUNT --role=roles/storage.admin"
    if [[ -n "$PROJECT_ID" ]]; then
      echo "If bucket may not exist, grant project-level create/get:"
      echo "  gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:$ACTIVE_ACCOUNT --role=roles/storage.admin"
    else
      echo "If bucket may not exist, grant project-level roles/storage.admin to the CI service account."
    fi
    echo "$BUCKET_CHECK_OUTPUT"
    exit 1
  else
    echo "Unexpected error while checking bucket '$BUCKET_BACKEND':"
    echo "$BUCKET_CHECK_OUTPUT"
    exit 1
  fi
fi