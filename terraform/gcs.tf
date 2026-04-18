resource "google_storage_bucket" "models_bucket" {
  name          = "${var.gcp_project}-models"
  location      = var.gcp_region
  force_destroy = true

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = false
  }

  encryption {
    default_kms_key_name = var.kms_key
  }

  storage_class = "STANDARD"
}