#!/bin/bash

# Lightweight Scheduler Wrapper Template
# Copy this file to your workspace as scheduler.sh

# Get the directory containing this script (workspace directory)
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the centralized scheduler
PROJECT_ROOT="/cerea_raid/users/wenbo.yu/Documents/Github/wildfire-diffusion"
SCHEDULER_MAIN="$PROJECT_ROOT/src/scripts/scheduler/scheduler_main.sh"

# Default config file
DEFAULT_CONFIG="$WORKSPACE_DIR/scheduler_config.yml"
CONFIG_FILE="${1}"

# If first argument is not a yml file, treat it as a command
if [[ "$1" != *.yml && "$1" != *.yaml ]]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
else
    shift  # Remove config file from arguments
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file.yml] [command] [options]"
    echo ""
    echo "Please create scheduler_config.yml in this workspace directory."
    echo "Template available at: $PROJECT_ROOT/src/scripts/scheduler/scheduler_config_template.yml"
    exit 1
fi

# Check if centralized scheduler exists
if [[ ! -f "$SCHEDULER_MAIN" ]]; then
    echo "Error: Centralized scheduler not found: $SCHEDULER_MAIN"
    echo "Please ensure the scheduler is properly installed in src/scripts/scheduler/"
    exit 1
fi

# Show help if no commands given
if [[ $# -eq 0 ]]; then
    cat << EOF
Workspace Training Scheduler

Usage: ./scheduler.sh [config.yml] COMMAND [OPTIONS]

COMMANDS:
    start CONFIG_LIST GPU_LIST  Start scheduler with job queue
    status                      Show current status
    stop                        Stop scheduler gracefully
    kill-all                    Emergency stop all sessions
    config                      Show current configuration
    help                        Show scheduler help

EXAMPLES:
    # Start with specific configs and GPU assignment
    ./scheduler.sh start "config1.yml,config2.yml" "0,1"
    
    # Start with all configs in directory
    ./scheduler.sh start
    
    # Check status
    ./scheduler.sh status
    
    # Show current configuration
    ./scheduler.sh config
    
    # Stop everything
    ./scheduler.sh kill-all

WORKSPACE SETUP:
    1. Copy this script to your workspace as scheduler.sh
    2. Create scheduler_config.yml in the same directory
    3. Configure paths and settings in scheduler_config.yml
    4. Create your task config files in the specified config directory

All scheduling logic is centralized in src/scripts/scheduler/
This wrapper simply calls the centralized system with workspace context.
EOF
    exit 0
fi

# Forward all commands to the centralized scheduler
exec bash "$SCHEDULER_MAIN" "$CONFIG_FILE" "$WORKSPACE_DIR" "$@"