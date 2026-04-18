terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0.0"
    }
  }

  required_version = ">= 1.2.0"

  backend "gcs" {
    bucket  = "options-pricing-explainability-tfstate"
    prefix  = "terraform/state"
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}