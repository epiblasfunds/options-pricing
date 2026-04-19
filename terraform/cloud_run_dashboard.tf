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