#!/usr/bin/env bash
# ==============================================================================
# Harshu AI OS - Local Production Deployment Script (POSIX / Linux)
# ==============================================================================
# Deploys a commit-pinned GHCR Docker image tag to a local container simulation
# with preflight validation, health check polling, and automated rollback.
# ==============================================================================

set -euo pipefail

IMAGE_TAG="${1:-}"
HEALTH_TIMEOUT_SECONDS="${2:-30}"
HEALTH_INTERVAL_SECONDS="${3:-2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
DATA_DIR="${REPO_ROOT}/data"
CONTAINER_NAME="harshu-ai-os"
REGISTRY_BASE="ghcr.io/harshithnadig/harshu_ai_os"

echo "=================================================="
echo "  HARSHU AI OS - Local Deployment Simulation"
echo "=================================================="

# 1. Safety & Tag Validation
if [[ -z "${IMAGE_TAG}" ]]; then
    echo "[ERROR] Image tag cannot be empty."
    echo "Usage: $0 <image_tag> [health_timeout_seconds] [health_interval_seconds]"
    exit 1
fi

CLEAN_TAG="${IMAGE_TAG#${REGISTRY_BASE}:}"

if [[ "${CLEAN_TAG}" == "latest" || "${CLEAN_TAG}" == *":latest" ]]; then
    echo "[ERROR] Refusing to deploy mutable tag '${CLEAN_TAG}'."
    echo "Deployments must target commit-pinned tags (e.g., sha-xxxxxxx). Note that true content immutability requires @sha256 digests."
    exit 1
fi

FULL_IMAGE="${REGISTRY_BASE}:${CLEAN_TAG}"
echo "Deployment Target: ${FULL_IMAGE}"

# 2. Capture Previous State for Rollback
PREVIOUS_IMAGE=""
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    PREVIOUS_IMAGE="$(docker inspect "${CONTAINER_NAME}" --format '{{.Config.Image}}' 2>/dev/null || true)"
    echo "[STATE] Found existing container '${CONTAINER_NAME}' running ${PREVIOUS_IMAGE}."
    echo "Stopping and removing previous container..."
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# Function to start container
start_container() {
    local img="$1"
    docker run -d         --name "${CONTAINER_NAME}"         -p 8000:8000         --env-file "${ENV_FILE}"         -v "${DATA_DIR}:/app/data"         "${img}"
}

# Function to poll health
poll_health() {
    local timeout="$1"
    local interval="$2"
    local elapsed=0
    local health_url="http://127.0.0.1:8000/health"

    echo -n "Polling health endpoint (${health_url})..."
    while (( elapsed < timeout )); do
        local running
        running="$(docker inspect "${CONTAINER_NAME}" --format '{{.State.Running}}' 2>/dev/null || echo "false")"
        if [[ "${running}" != "true" ]]; then
            echo " [CONTAINER CRASHED]"
            return 1
        fi

        if curl -s -f -m 2 "${health_url}" | grep -q '"status":"healthy"' 2>/dev/null; then
            echo " [HEALTHY in ${elapsed}s]"
            return 0
        fi

        echo -n "."
        sleep "${interval}"
        elapsed=$(( elapsed + interval ))
    done

    echo " [TIMEOUT]"
    return 1
}

# 3. Deploy Candidate
echo "Launching container '${CONTAINER_NAME}' with ${FULL_IMAGE}..."
if ! start_container "${FULL_IMAGE}"; then
    echo "[ERROR] Failed to start container."
    DEPLOYMENT_SUCCESS=0
else
    if poll_health "${HEALTH_TIMEOUT_SECONDS}" "${HEALTH_INTERVAL_SECONDS}"; then
        DEPLOYMENT_SUCCESS=1
    else
        DEPLOYMENT_SUCCESS=0
    fi
fi

# 4. Rollback on Failure
if [[ "${DEPLOYMENT_SUCCESS}" -eq 0 ]]; then
    echo "=================================================="
    echo "  DEPLOYMENT FAILED - INITIATING ROLLBACK"
    echo "=================================================="
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true

    if [[ -n "${PREVIOUS_IMAGE}" ]]; then
        echo "Attempting rollback to previous image: ${PREVIOUS_IMAGE}"
        if start_container "${PREVIOUS_IMAGE}" && poll_health "${HEALTH_TIMEOUT_SECONDS}" "${HEALTH_INTERVAL_SECONDS}"; then
            echo "=================================================="
            echo "  ROLLBACK SUCCESSFUL"
            echo "=================================================="
            echo "Restored previous deployment: ${PREVIOUS_IMAGE}"
            exit 1
        else
            echo "[CRITICAL] Rollback failed to restore previous container."
            exit 1
        fi
    else
        echo "[ERROR] No previous image to rollback to."
        exit 1
    fi
fi

echo "=================================================="
echo "  DEPLOYMENT SUCCESSFUL"
echo "=================================================="
echo "Active Image: ${FULL_IMAGE}"
