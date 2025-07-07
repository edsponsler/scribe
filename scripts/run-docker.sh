#!/bin/bash

# This script ensures that the docker-compose command is run from the project root.

# Get the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to the project root directory (one level up from the script's directory).
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Announce the current directory to confirm we are in the right place.
echo "Running docker-compose from: $(pwd)"

# Execute the docker-compose command.
docker-compose up