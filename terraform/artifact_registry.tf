resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "dashboard-repo-mini-ibex-options"
  format        = "DOCKER"
}