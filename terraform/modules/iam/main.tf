variable "project_id" {
  type = string
}

variable "base_name" {
  type = string
}

# Create the service account for the Cloud Run service
resource "google_service_account" "cloud_run_sa" {
  account_id   = "${var.base_name}-cloud-run-sa"
  display_name = "Service Account for ${var.base_name} Cloud Run App"
}

# Grant the service account permission to use Vertex AI
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant the service account permission to view/search the Discovery Engine datastore
resource "google_project_iam_member" "discovery_engine_viewer" {
  project = var.project_id
  role    = "roles/discoveryengine.viewer"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Output the service account email for the Cloud Run module
output "service_account_email" {
  value = google_service_account.cloud_run_sa.email
}
