terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0.0"
    }
  }

  required_version = ">= 1.2.0"

  backend "gcs" {}
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

output "tf_backend_bucket" {
  value = var.tf_backend_bucket
}