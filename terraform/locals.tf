locals {
  image_registry_base                  = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/${var.artifact_repository_id}"
  cloud_run_runtime_sa                 = "serviceAccount:${var.github_actions_service_account_email}"
  api_service_name                     = "api"
  dashboard_service_name               = "dashboard"
  api_cloud_run_timeout_seconds        = 400
  dashboard_cloud_run_timeout_seconds  = 300
  dashboard_api_client_timeout_seconds = 300
}
