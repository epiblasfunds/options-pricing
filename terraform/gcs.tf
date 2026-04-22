resource "google_storage_bucket" "volatility_models_bucket" {
  name     = var.volatility_models_bucket
  location = var.gcp_region

  storage_class               = "STANDARD"
  force_destroy               = true
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  encryption {
    default_kms_key_name = var.kms_key != "" ? var.kms_key : null
  }
}

resource "google_storage_bucket" "explainability_artifacts_bucket" {
  name     = var.explainability_artifacts_bucket
  location = var.gcp_region

  storage_class               = "STANDARD"
  force_destroy               = true
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  encryption {
    default_kms_key_name = var.kms_key != "" ? var.kms_key : null
  }
}
