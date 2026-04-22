resource "google_artifact_registry_repository" "repo" {
  location      = var.gcp_region
  repository_id = var.artifact_repository_id
  format        = "DOCKER"
}