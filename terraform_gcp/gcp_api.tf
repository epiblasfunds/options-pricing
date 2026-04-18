resource "google_artifact_registry_repository" "api_repo" {
  location      = var.gcp_region
  repository_id = var.api_repo_id
  format        = "DOCKER"
  description   = "Repo for API container"
}

resource "google_cloud_run_service" "api" {
  name     = var.api_image_name
  location = var.gcp_region

  template {
    spec {
      containers {
        image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/${var.api_repo_id}/${var.api_image_name}:latest"

        env {
          name  = "AWS_DEFAULT_REGION"
          value = "eu-west-1"
        }
        
        env {
          name  = "AWS_ACCESS_KEY_ID"
          value = var.aws_access_key_id
        }

        env {
          name  = "AWS_SECRET_ACCESS_KEY"
          value = var.aws_secret_access_key
        }

        ports {
          container_port = 8000
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

resource "google_cloud_run_service_iam_member" "allow_all" {
  service  = google_cloud_run_service.api.name
  location = google_cloud_run_service.api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
