variable "gcp_region" {
  description = "GCP region to deploy Cloud Run"
  type        = string
  default     = "europe-west1"
}

variable "gcp_project" {
  type    = string
  default = "practica-miax"
}

variable "api_repo_id" {
  type    = string
  default = "fastapi-repo-mini-ibex-options"
}

variable "ui_repo_id" {
  type    = string
  default = "streamlit-repo-mini-ibex-options"
}

variable "api_image_name" {
  type    = string
  default = "fastapi-querydynamodb"
}

variable "ui_image_name" {
  type    = string
  default = "streamlit"
}

variable "aws_access_key_id" {
  type = string
}

variable "aws_secret_access_key" {
  type = string
}
