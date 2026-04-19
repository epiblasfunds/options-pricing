variable "tf_backend_bucket" {
  description = "Bucket name for Terraform backend state"
  type        = string
}
variable "gcp_region" {
  description = "GCP region to deploy resources"
  type        = string
  default     = "europe-west1"
}

variable "gcp_project" {
  description = "GCP project ID"
  type        = string
}

variable "kms_key" {
  description = "KMS key for bucket encryption"
  type        = string
  default     = ""
}
