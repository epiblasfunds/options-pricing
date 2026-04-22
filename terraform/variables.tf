variable "gcp_project" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region to deploy resources"
  type        = string
  default     = "europe-west1"
}

variable "artifact_repository_id" {
  description = "Artifact Registry repository ID used for API and dashboard images"
  type        = string
}

variable "volatility_models_bucket" {
  description = "Bucket name for storing volatility models"
  type        = string
}

variable "explainability_artifacts_bucket" {
  description = "Bucket name for storing explainability artifacts used by the dashboard"
  type        = string
}

variable "github_actions_service_account_email" {
  description = "Service account email used by GitHub Actions workflows"
  type        = string
}

variable "github_actions_project_roles" {
  description = "Project-level roles granted to the GitHub Actions service account"
  type        = set(string)
  default = [
    "roles/artifactregistry.admin",
    "roles/iam.serviceAccountUser",
    "roles/resourcemanager.projectIamAdmin",
    "roles/run.admin",
    "roles/storage.admin",
  ]
}

variable "kms_key" {
  description = "KMS key for bucket encryption (leave empty to use Google-managed keys)"
  type        = string
  default     = ""
}
