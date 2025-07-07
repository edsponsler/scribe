output "raw_data_bucket" {
  description = "The name of the Cloud Storage bucket for raw data."
  value       = module.storage.raw_bucket_name
}

output "processed_data_bucket" {
  description = "The name of the Cloud Storage bucket for processed data."
  value       = module.storage.processed_bucket_name
}

output "staging_bucket" {
  description = "The name of the Cloud Storage bucket for staging artifacts."
  value       = module.storage.staging_bucket_name
}

output "cloud_run_service_account_email" {
  description = "The email of the service account created for Cloud Run."
  value       = module.iam.service_account_email
}

output "artifact_registry_repository_url" {
  description = "The URL of the Artifact Registry Docker repository."
  value       = module.artifact_registry.repository_url
}
