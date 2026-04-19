resource "google_storage_bucket" "volatility_models_bucket" {
  name          = var.volatility_models_bucket
  location      = var.gcp_region
  force_destroy = true

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = false
  }

  encryption {
    default_kms_key_name = var.kms_key != "" ? var.kms_key : null
  }

  storage_class = "STANDARD"
}

resource "google_storage_bucket" "dashboard_models_bucket" {
  name          = var.dashboard_models_bucket
  location      = var.gcp_region
  force_destroy = true

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = false
  }

  encryption {
    default_kms_key_name = var.kms_key != "" ? var.kms_key : null
  }

  storage_class = "STANDARD"
}