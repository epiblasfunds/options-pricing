terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"

  
  backend "s3" {
    bucket         = "ui-api-mini-ibex-options-terraform-state"
    key            = "terraform.tfstate"
    region         = "eu-west-1"
    encrypt        = true
  }
}

provider "google" {
  region      = var.gcp_region
}

provider "aws" {
  region = "eu-west-1"
}
