#!/bin/bash
# Source this file before running exercises:
#   source load_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    return 1 2>/dev/null || exit 1
fi

set -a
source "$ENV_FILE"
set +a

echo "Environment loaded from .env"
