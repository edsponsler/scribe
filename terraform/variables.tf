variable "gcp_project_id" {
  type        = string
  description = "The Google Cloud project ID to deploy the resources in."
}

variable "gcp_region" {
  type        = string
  description = "The Google Cloud region to deploy the resources in."
  default     = "us-central1"
}

variable "base_name" {
  type        = string
  description = "A base name to use as a prefix for all created resources."
  default     = "scribe"
}