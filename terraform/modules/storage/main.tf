variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "base_name" {
  type = string
}

# Bucket for raw source documents
resource "google_storage_bucket" "raw_data" {
  name          = "${var.base_name}-raw-data-${var.project_id}"
  location      = var.region
  force_destroy = true # Set to false in production if you want to prevent accidental deletion
  uniform_bucket_level_access = true
}

# Bucket for processed, chunked documents (e.g., JSONL files)
resource "google_storage_bucket" "processed_data" {
  name          = "${var.base_name}-processed-data-${var.project_id}"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
}

# Staging bucket for Vertex AI and ADK artifacts
resource "google_storage_bucket" "staging" {
  name          = "${var.base_name}-staging-${var.project_id}"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
}

# Outputs for other modules to reference
output "raw_bucket_name" {
  value = google_storage_bucket.raw_data.name
}

output "processed_bucket_name" {
  value = google_storage_bucket.processed_data.name
}

output "staging_bucket_name" {
  value = google_storage_bucket.staging.name
}
