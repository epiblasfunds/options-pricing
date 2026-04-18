#!/bin/bash

# This script creates the necessary S3 for the AWS project:
#    - terraform state
#    - temporary database
set -e

REGION="eu-west-1"

BUCKET_NAME_TERRAFORM="mini-ibex-options-terraform-state"
echo "Checking if the bucket '$BUCKET_NAME_TERRAFORM' already exists in $REGION..."
if aws s3api head-bucket --bucket "$BUCKET_NAME_TERRAFORM" 2>/dev/null; then
  echo "The S3 bucket '$BUCKET_NAME_TERRAFORM' already exists."
else
  echo "Creating S3 bucket '$BUCKET_NAME_TERRAFORM' in region '$REGION'..."

  aws s3api create-bucket \
    --bucket "$BUCKET_NAME_TERRAFORM" \
    --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"

  echo "Activating bucket versioning..."
  aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME_TERRAFORM" \
    --versioning-configuration Status=Enabled

  echo "Bucket successfully created with active versioning."
fi

BUCKET_NAME_TMP_DB="mini-ibex-options-tmp-db"
echo "Checking if the bucket '$BUCKET_NAME_TMP_DB' already exists in $REGION..."
if aws s3api head-bucket --bucket "$BUCKET_NAME_TMP_DB" 2>/dev/null; then
  echo "The S3 bucket '$BUCKET_NAME_TMP_DB' already exists."
else
  echo "Creating S3 bucket '$BUCKET_NAME_TMP_DB' in region '$REGION'..."

  aws s3api create-bucket \
    --bucket "$BUCKET_NAME_TMP_DB" \
    --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"

  echo "Activating bucket versioning..."
  aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME_TMP_DB" \
    --versioning-configuration Status=Enabled

  echo "Bucket successfully created with active versioning."
fi
