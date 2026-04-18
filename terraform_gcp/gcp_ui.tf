resource "google_artifact_registry_repository" "ui_repo" {
  location      = var.gcp_region
  repository_id = var.ui_repo_id
  format        = "DOCKER"
  description   = "Repo for UI Streamlit container"
}

resource "google_cloud_run_service" "ui" {
  name     = var.ui_image_name
  location = var.gcp_region

  template {
    spec {
      containers {
        image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/${var.ui_repo_id}/${var.ui_image_name}:latest"

        env {
            name    = "API_URL"
            value   = google_cloud_run_service.api.status[0].url
        }

        ports {
          container_port = 8501
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  autogenerate_revision_name = true
}

resource "google_cloud_run_service_iam_member" "ui_allow_all" {
  service  = google_cloud_run_service.ui.name
  location = google_cloud_run_service.ui.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
