resource "google_cloud_run_service" "dashboard" {
  name     = "dashboard"
  location = var.gcp_region

  template {
    spec {
      containers {
        image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/dashboard-repo-mini-ibex-options/dashboard:latest"

        resources {
          limits = {
            memory = "512Mi"
            cpu    = "1"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.dashboard.name
  location = google_cloud_run_service.dashboard.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_project_iam_member" "cloud_run_artifact_registry_reader" {
  project = var.gcp_project
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:269293143637-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "github_actions_owner" {
  project = var.gcp_project
  role    = "roles/owner"
  member  = "serviceAccount:github-actions@options-pricing-explainability.iam.gserviceaccount.com"
}