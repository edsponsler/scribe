# Configure the Google Cloud provider
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Module to enable all necessary APIs for the project
module "project_setup" {
  source = "./modules/project_setup"
}

# Module to create the Cloud Storage buckets
module "storage" {
  source      = "./modules/storage"
  project_id  = var.gcp_project_id
  region      = var.gcp_region
  base_name   = var.base_name
  # Ensure APIs are enabled before creating resources
  depends_on = [module.project_setup]
}

# Module to create the service account and grant IAM permissions
module "iam" {
  source              = "./modules/iam"
  project_id          = var.gcp_project_id
  base_name           = var.base_name
  depends_on          = [module.project_setup]
}

# Module to create the Artifact Registry repository
module "artifact_registry" {
  source      = "./modules/artifact_registry"
  project_id  = var.gcp_project_id
  region      = var.gcp_region
  base_name   = var.base_name
  depends_on  = [module.project_setup]
}
