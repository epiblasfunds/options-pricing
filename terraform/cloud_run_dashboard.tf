resource "google_cloud_run_service" "dashboard" {
  name     = "dashboard"
  location = var.gcp_region

  template {
    spec {
      containers {
        image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/dashboard-repo-mini-ibex-options/dashboard:latest"

        env {
          name  = "MODEL_STORAGE_BACKEND"
          value = "gcp"
        }

        env {
          name  = "API_BASE_URL"
          value = google_cloud_run_service.api.status[0].url
        }

        resources {
          limits = {
            memory = "1Gi"
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

resource "google_project_iam_member" "cloud_run_storage_object_viewer" {
  project = var.gcp_project
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:269293143637-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "github_actions_run_admin" {
  project = var.gcp_project
  role    = "roles/run.admin"
  member  = "serviceAccount:github-actions@options-pricing-explainability.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "github_actions_sa_user" {
  project = var.gcp_project
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:github-actions@options-pricing-explainability.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "github_actions_artifact_registry" {
  project = var.gcp_project
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:github-actions@options-pricing-explainability.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "github_actions_storage_admin" {
  project = var.gcp_project
  role    = "roles/storage.admin"
  member  = "serviceAccount:github-actions@options-pricing-explainability.iam.gserviceaccount.com"
}