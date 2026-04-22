locals {
  image_registry_base                  = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/${var.artifact_repository_id}"
  compute_sa                           = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"
  api_service_name                     = "api"
  dashboard_service_name               = "dashboard"
  api_cloud_run_timeout_seconds        = 400
  dashboard_cloud_run_timeout_seconds  = 300
  dashboard_api_client_timeout_seconds = 300
}
