# This file enables all the necessary APIs for the project to function.

resource "google_project_service" "iam" {
  service = "iam.googleapis.com"
}

resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
}

resource "google_project_service" "artifact_registry" {
  service = "artifactregistry.googleapis.com"
}

resource "google_project_service" "vertex_ai" {
  service = "aiplatform.googleapis.com"
}

resource "google_project_service" "discovery_engine" {
  service = "discoveryengine.googleapis.com"
}

resource "google_project_service" "cloud_storage" {
  service = "storage.googleapis.com"
}

resource "google_project_service" "service_usage" {
  service = "serviceusage.googleapis.com"
}
