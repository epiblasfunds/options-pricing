output "api_url" {
  value = try(google_cloud_run_service.api[0].status[0].url, null)
}

output "dashboard_url" {
  value = try(google_cloud_run_service.dashboard[0].status[0].url, null)
}

output "volatility_models_bucket_name" {
  value = google_storage_bucket.volatility_models_bucket.name
}

output "explainability_artifacts_bucket_name" {
  value = google_storage_bucket.explainability_artifacts_bucket.name
}
