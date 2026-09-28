#!/usr/bin/env bash
# Source this file before resolving project-relative paths.
ALIGNOPSD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ALIGNOPSD_ROOT"
export PYTHONPATH="$ALIGNOPSD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT="${DATA_ROOT:-$ALIGNOPSD_ROOT/data}"
export SEARCH_DATA_DIR="${SEARCH_DATA_DIR:-$DATA_ROOT/searchR1_processed_direct}"
export SEARCH_INDEX_DIR="${SEARCH_INDEX_DIR:-$DATA_ROOT/searchR1}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$DATA_ROOT/alfworld}"
export ALFWORLD_PREPARED_DATA_DIR="${ALFWORLD_PREPARED_DATA_DIR:-$DATA_ROOT/alfworld_prepared}"
export WEBSHOP_PREPARED_DATA_DIR="${WEBSHOP_PREPARED_DATA_DIR:-$DATA_ROOT/webshop_prepared}"
export WEBSHOP_ASSET_ROOT="${WEBSHOP_ASSET_ROOT:-$ALIGNOPSD_ROOT/agent_system/environments/env_package/webshop/webshop}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1"
ALIGNOPSD_CONFIG_ONLY=false
for alignopsd_arg in "$@"; do
    case "$alignopsd_arg" in
        --cfg|--cfg=*|--help|-h|--hydra-help|--info|--info=*) ALIGNOPSD_CONFIG_ONLY=true ;;
    esac
done
