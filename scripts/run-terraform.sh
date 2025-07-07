#!/bin/bash

# This script provides instructions for provisioning the project's Google Cloud
# infrastructure using Terraform. It follows best practices by saving the execution
# plan to a file and then applying that exact plan.

echo "======================================================"
echo "  Terraform Infrastructure Provisioning Instructions  "
echo "======================================================"
echo

# --- Step 0: Prerequisites and Project ID ---
echo "--- Step 0: Prerequisites and Project ID ---"
echo "Before you begin, make sure you have:"
echo "1. Installed Terraform on your local machine."
echo "2. Authenticated with Google Cloud ('gcloud auth login' and 'gcloud auth application-default login')."
echo "3. Navigated your terminal into the 'terraform' directory of this project."
echo

# Prompt for the Project ID
read -p "Please enter your Google Cloud Project ID and press Enter: " gcp_project_id

# Validate that the project ID is not empty
if [ -z "$gcp_project_id" ]; then
    echo "Error: Project ID cannot be empty."
    exit 1
fi

# Define the plan file name based on the project ID
PLAN_FILE="${gcp_project_id}.tfplan"

echo
echo "Terraform will use the plan file name: $PLAN_FILE"
echo
echo "Press Enter to continue..."
read

# --- Step 1: Initialize Terraform ---
echo "--- Step 1: Initialize Terraform ---"
echo "The first command is 'terraform init'. This downloads the necessary provider"
echo "plugins. You only need to run this once per project."
echo
echo "From inside the 'terraform' directory, run the following command:"
echo
echo "terraform init"
echo
echo "Press Enter when you have completed this step..."
read

# --- Step 2: Plan the Deployment and Save to a File ---
echo "--- Step 2: Plan the Deployment and Save to a File ---"
echo "The next command is 'terraform plan -out'. This creates an execution plan,"
echo "shows you what will be created, and saves that exact plan to a file."
echo "This ensures that what you apply is exactly what you reviewed."
echo
echo "Copy and paste the following command:"
echo
echo "terraform plan -var=\"gcp_project_id=$gcp_project_id\" -out=\"$PLAN_FILE\""
echo
echo "Review the output carefully to ensure it matches your expectations."
echo
echo "Press Enter when you have completed this step..."
read

# --- Step 3: Apply the Saved Plan ---
echo "--- Step 3: Apply the Saved Plan ---"
echo "The final command is 'terraform apply'. By providing the plan file name,"
echo "Terraform will execute the exact plan you just saved, without asking for"
echo "any additional input or confirmation (as it was confirmed during the plan phase)."
echo
echo "Copy and paste the following command:"
echo
echo "terraform apply \"$PLAN_FILE\""
echo
echo "After the command completes, all your cloud infrastructure will be provisioned."
echo
echo "======================================================"
echo "            Instructions Complete"
echo "======================================================"
