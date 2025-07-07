#!/bin/bash

# This script automates the deployment of the ASRR/SOMMAS application to Google Cloud Run.
# It ensures all required environment variables are set, prompting to load them
# from a .env file if necessary.

# --- Configuration ---
# Uses the 'sommas' naming convention as requested.
SERVICE_NAME="sommas-service"
REPO_NAME="sommas-repo"
IMAGE_NAME="sommas-app"
SERVICE_ACCOUNT_NAME="sommas-cloud-run-sa"

# --- Script Logic ---

# Get the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to the project root directory (one level up from the script's directory).
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Function to check if all required variables are set.
check_vars() {
    if [ -z "$GCP_PROJECT_ID" ] || [ -z "$GCP_LOCATION" ] || [ -z "$STAGING_BUCKET" ] || [ -z "$FAST_MODEL" ] || [ -z "$DATA_STORE_ID" ]; then
        return 1 # Returns 1 (false) if any variable is missing.
    else
        return 0 # Returns 0 (true) if all variables are set.
    fi
}

# Check for variables and handle if they are missing.
if ! check_vars; then
    echo "One or more required environment variables are not set."
    if [ -f ".env" ]; then
        echo "A .env file was found."
        read -p "Do you want to load the variables from .env to continue? (y/n) " -n 1 -r
        echo # Move to a new line
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Loading variables from .env..."
            # Exports all variables from the .env file into the current shell.
            set -a
            source .env
            set +a
        else
            echo "Deployment cancelled by user."
            exit 1
        fi
    else
        echo "No .env file found. Please set the required environment variables manually."
        echo "Required variables: GCP_PROJECT_ID, GCP_LOCATION, STAGING_BUCKET, FAST_MODEL, DATA_STORE_ID"
        exit 1
    fi
fi

# Re-check variables after attempting to load them.
if ! check_vars; then
    echo "Error: Variables are still not set after loading .env. Please check your .env file."
    exit 1
fi

# Construct the full image URL for Artifact Registry.
IMAGE_URL="$GCP_LOCATION-docker.pkg.dev/$GCP_PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$GCP_PROJECT_ID.iam.gserviceaccount.com"

# Announce the deployment details.
echo "---------------------------------"
echo "Starting deployment to Cloud Run..."
echo "  Project: $GCP_PROJECT_ID"
echo "  Region: $GCP_LOCATION"
echo "  Service: $SERVICE_NAME"
echo "  Image: $IMAGE_URL"
echo "---------------------------------"

# Execute the gcloud run deploy command.
gcloud run deploy "$SERVICE_NAME" \
  --image="$IMAGE_URL" \
  --platform=managed \
  --region="$GCP_LOCATION" \
  --service-account="$SERVICE_ACCOUNT_EMAIL" \
  --allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=$GCP_PROJECT_ID,GCP_LOCATION=$GCP_LOCATION,STAGING_BUCKET=$STAGING_BUCKET,FAST_MODEL=$FAST_MODEL,DATA_STORE_ID=$DATA_STORE_ID"

# Check the exit code of the deployment command.
if [ $? -eq 0 ]; then
    echo "✅ Deployment completed successfully."
else
    echo "❌ Deployment failed."
fi
