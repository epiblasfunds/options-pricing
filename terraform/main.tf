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

output "volatility_models_bucket_name" {
  value = var.volatility_models_bucket
}

output "dashboard_models_bucket_name" {
  value = var.dashboard_models_bucket
}