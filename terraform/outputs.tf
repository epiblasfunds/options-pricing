output "api_url" {
  value = google_cloud_run_service.api.status[0].url
}

output "dashboard_url" {
  value = google_cloud_run_service.dashboard.status[0].url
}

output "volatility_models_bucket_name" {
  value = google_storage_bucket.volatility_models_bucket.name
}

output "explainability_artifacts_bucket_name" {
  value = google_storage_bucket.explainability_artifacts_bucket.name
}
