variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "base_name" {
  type = string
}

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "${var.base_name}-repo"
  description   = "Docker repository for the ${var.base_name} application"
  format        = "DOCKER"
}

output "repository_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}
