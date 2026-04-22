resource "google_project_iam_member" "cloud_run_artifact_registry_reader" {
  project = var.gcp_project
  role    = "roles/artifactregistry.reader"
  member  = local.cloud_run_runtime_sa
}

resource "google_project_iam_member" "cloud_run_storage_object_viewer" {
  project = var.gcp_project
  role    = "roles/storage.objectViewer"
  member  = local.cloud_run_runtime_sa
}

resource "google_project_iam_member" "github_actions_project_roles" {
  for_each = var.github_actions_project_roles

  project = var.gcp_project
  role    = each.value
  member  = "serviceAccount:${var.github_actions_service_account_email}"
}
