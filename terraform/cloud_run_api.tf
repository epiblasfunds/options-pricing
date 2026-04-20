resource "google_cloud_run_service" "api" {
  name     = "api"
  location = var.gcp_region

  template {
    spec {
      containers {
        image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/dashboard-repo-mini-ibex-options/api:latest"

        env {
          name  = "MODEL_STORAGE_BACKEND"
          value = "gcp"
        }

        resources {
          limits = {
            memory = "2Gi"
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

resource "google_cloud_run_service_iam_member" "api_public" {
  service  = google_cloud_run_service.api.name
  location = google_cloud_run_service.api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}