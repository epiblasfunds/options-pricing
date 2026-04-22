#!/bin/bash
set -euo pipefail

REGION="${TF_VAR_gcp_region:-${GCP_REGION:-europe-west1}}"
PROJECT_ID="${TF_VAR_gcp_project:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/resources/cloud_config.json"

BUCKET_BACKEND="${TF_BACKEND_BUCKET:-$(sed -n 's/.*"backend_bucket"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$CONFIG_FILE" | head -n1)}"

echo "Bootstrap backend bucket: $BUCKET_BACKEND"
if BUCKET_CHECK_OUTPUT=$(gcloud storage buckets describe "gs://$BUCKET_BACKEND" 2>&1); then
  echo "Bucket exists"
else
  case "$BUCKET_CHECK_OUTPUT" in
    *NOT_FOUND*|*BucketNotFoundException*|*"Not Found"*|*404*)
      echo "Creating bucket in $REGION"
      gcloud storage buckets create "gs://$BUCKET_BACKEND" --location="$REGION"
      gcloud storage buckets update "gs://$BUCKET_BACKEND" --versioning
      echo "Bucket created + versioning enabled"
      ;;
    *AccessDenied*|*PERMISSION_DENIED*|*storage.buckets.get*|*storage.buckets.create*|*403*)
      ACTIVE_ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
      echo "Permission denied (account: ${ACTIVE_ACCOUNT:-unknown})"
      echo "Need storage.buckets.get + storage.buckets.create (or roles/storage.admin)"
      if [[ -n "$PROJECT_ID" ]]; then
        echo "Grant example: gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:$ACTIVE_ACCOUNT --role=roles/storage.admin"
      else
        echo "Grant roles/storage.admin to CI service account"
      fi
      echo "$BUCKET_CHECK_OUTPUT"
      exit 1
      ;;
    *)
      echo "Unexpected error checking bucket"
      echo "$BUCKET_CHECK_OUTPUT"
      exit 1
      ;;
  esac
fi