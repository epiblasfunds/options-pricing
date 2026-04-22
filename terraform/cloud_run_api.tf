resource "google_cloud_run_service" "api" {
  count    = var.manage_cloud_run_services ? 1 : 0
  name     = local.api_service_name
  location = var.gcp_region

  template {
    spec {
      service_account_name = var.github_actions_service_account_email

      containers {
        image = "${local.image_registry_base}/api:latest"

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

      timeout_seconds = local.api_cloud_run_timeout_seconds
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image,
    ]
  }
}

resource "google_cloud_run_service_iam_member" "api_invoker" {
  count    = var.manage_cloud_run_services ? 1 : 0
  service  = google_cloud_run_service.api[0].name
  location = google_cloud_run_service.api[0].location
  role     = "roles/run.invoker"
  member   = "allUsers"
}