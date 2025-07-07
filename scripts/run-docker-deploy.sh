#!/bin/bash

# This script provides instructions for the complete Docker build, push, and
# deploy loop. It does not execute the commands itself but guides the user
# on what to do and what to expect at each step.

echo "================================================================"
echo "  Docker Build, Push, and Deploy Instructions for SOMMAS App  "
echo "================================================================"
echo

# --- Step 0: Prerequisites ---
echo "--- Step 0: Prerequisites ---"
echo "Before you begin, please ensure:"
echo "1. Docker Desktop is running on your Windows machine."
echo "2. You are authenticated with Google Cloud. If you get an error later,"
echo "   run 'gcloud auth login --no-browser' and 'gcloud auth application-default login --no-browser'."
echo "3. You are running this script from the project's root directory."
echo
echo "Press Enter to continue..."
read

# --- Step 1: Build the Docker Image ---
echo "--- Step 1: Build the Docker Image ---"
echo "This command reads your 'Dockerfile' and builds a local container image"
echo "containing your application and all its dependencies. The '-t sommas-app'"
echo "part 'tags' or names the image for easy reference."
echo
echo "Copy and paste the following command to build your image:"
echo
echo "docker build -t sommas-app ."
echo
echo "Press Enter when the build is complete..."
read

# --- Step 2: Tag the Image for Artifact Registry ---
echo "--- Step 2: Tag the Image for Artifact Registry ---"
echo "Cloud Run needs to pull your image from Google's Artifact Registry. To get"
echo "it there, you must first 'tag' your local image with the full path to"
echo "your repository. This command does not create a new image, it just gives"
echo "your existing 'sommas-app' image an additional name."
echo
echo "Loading variables from .env to construct the command..."
set -a
source .env
set +a
echo
echo "Copy and paste the following command to tag your image:"
echo
echo "docker tag sommas-app \$GCP_LOCATION-docker.pkg.dev/\$GCP_PROJECT_ID/sommas-repo/sommas-app:latest"
echo
echo "Press Enter when you have completed this step..."
read

# --- Step 3: Push the Image to Artifact Registry ---
echo "--- Step 3: Push the Image to Artifact Registry ---"
echo "This command uploads your tagged image from your local machine to your"
echo "secure repository in Google Cloud, making it available for Cloud Run."
echo
echo "Copy and paste the following command to push your image:"
echo
echo "docker push \$GCP_LOCATION-docker.pkg.dev/\$GCP_PROJECT_ID/sommas-repo/sommas-app:latest"
echo
echo "Press Enter when the push is complete..."
read

# --- Step 4: Deploy to Cloud Run ---
echo "--- Step 4: Deploy to Cloud Run ---"
echo "The final step is to run the deployment script. This script will tell"
echo "Cloud Run to pull the latest image you just pushed and deploy it as a new"
echo "version of your service with all the correct environment variables."
echo
echo "Run the deployment script with the following command:"
echo
echo "./scripts/deploy.sh"
echo
echo "================================================================"
echo "                  Instructions Complete"
echo "================================================================"
