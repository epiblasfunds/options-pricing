resource "google_cloud_run_service" "dashboard" {
  count    = var.manage_cloud_run_services ? 1 : 0
  name     = local.dashboard_service_name
  location = var.gcp_region

  template {
    metadata {
      annotations = {
        "run.googleapis.com/startup-cpu-boost" = "true"
      }
    }
    spec {
      service_account_name = var.github_actions_service_account_email

      containers {
        image = "${local.image_registry_base}/dashboard:latest"

        env {
          name  = "MODEL_STORAGE_BACKEND"
          value = "gcp"
        }

        env {
          name  = "API_BASE_URL"
          value = google_cloud_run_service.api[0].status[0].url
        }

        env {
          name  = "API_TIMEOUT_SECONDS"
          value = tostring(local.dashboard_api_client_timeout_seconds)
        }

        resources {
          limits = {
            memory = "2Gi"
            cpu    = "2"
          }
        }

      }

      timeout_seconds = local.dashboard_cloud_run_timeout_seconds
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

resource "google_cloud_run_service_iam_member" "dashboard_invoker" {
  count    = var.manage_cloud_run_services ? 1 : 0
  service  = google_cloud_run_service.dashboard[0].name
  location = google_cloud_run_service.dashboard[0].location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

