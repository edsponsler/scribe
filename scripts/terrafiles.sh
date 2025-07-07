#!/bin/bash

# This script creates the directory and file structure for the
# modular Terraform configuration.

echo "Creating Terraform project structure..."

# Create the main terraform directory
mkdir -p terraform/modules

# Create the root .tf files
touch terraform/main.tf
touch terraform/variables.tf
touch terraform/outputs.tf

# Define the list of modules
modules=("project_setup" "storage" "iam" "artifact_registry")

# Loop through the modules list to create directories and files
for module in "${modules[@]}"
do
    echo "Creating module: $module"
    mkdir -p "terraform/modules/$module"
    touch "terraform/modules/$module/main.tf"
done

echo "✅ Terraform file structure created successfully."
echo "You can now populate the .tf files with the provided code."

# Make this script executable for future use (optional, but good practice)
chmod +x "$0"
