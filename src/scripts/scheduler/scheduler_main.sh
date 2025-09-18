#!/bin/bash

# Main Scheduler Entry Point
# This script orchestrates the scheduler functionality

# Get the directory containing this script (src/scripts/scheduler/)
SCHEDULER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the core functions
source "$SCHEDULER_DIR/scheduler_core.sh"

# Global variables
CONFIG_FILE=""
WORKSPACE_DIR=""

# Main function to run scheduler
run_scheduler() {
    local config_file="$1"
    local workspace_dir="$2"
    local command="$3"
    shift 3  # Remove first 3 arguments, rest are command arguments
    
    # Validate inputs
    if [[ ! -f "$config_file" ]]; then
        echo "Error: Configuration file not found: $config_file"
        exit 1
    fi
    
    if [[ ! -d "$workspace_dir" ]]; then
        echo "Error: Workspace directory not found: $workspace_dir"
        exit 1
    fi
    
    # Set global variables
    CONFIG_FILE="$config_file"
    WORKSPACE_DIR="$workspace_dir"
    
    # Parse configuration
    parse_config "$CONFIG_FILE" "$WORKSPACE_DIR"
    
    # Handle commands
    case "$command" in
        "start")
            # Start scheduler
            if tmux has-session -t "$SCHEDULER_SESSION" 2>/dev/null; then
                echo "Scheduler session already exists: $SCHEDULER_SESSION"
                echo "Use 'tmux attach -t $SCHEDULER_SESSION' to view it"
                echo "Or use 'stop' to stop it first"
                exit 1
            fi
            
            # Initialize job queue
            init_queue "$1" "$2"
            
            # Start scheduler in tmux
            tmux new-session -d -s "$SCHEDULER_SESSION"
            tmux send-keys -t "$SCHEDULER_SESSION" "cd $WORKSPACE_DIR" Enter
            tmux send-keys -t "$SCHEDULER_SESSION" "bash $SCHEDULER_DIR/scheduler_main.sh \"$CONFIG_FILE\" \"$WORKSPACE_DIR\" run" Enter
            
            echo "Scheduler started in session: $SCHEDULER_SESSION"
            echo "Use 'tmux attach -t $SCHEDULER_SESSION' to monitor"
            echo "Use 'status' to check progress"
            ;;
        "run")
            # Internal command - run the main loop
            main_loop
            ;;
        "status")
            show_status
            ;;
        "stop")
            if tmux has-session -t "$SCHEDULER_SESSION" 2>/dev/null; then
                tmux kill-session -t "$SCHEDULER_SESSION"
                echo "Scheduler stopped"
            else
                echo "Scheduler not running"
            fi
            ;;
        "kill-all")
            # Emergency stop - kill all training sessions
            echo "Killing all training sessions for task: $TASK_NAME"
            for session in $(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep "^${WORKER_PREFIX}_"); do
                echo "Killing: $session"
                tmux kill-session -t "$session"
            done
            if tmux has-session -t "$SCHEDULER_SESSION" 2>/dev/null; then
                tmux kill-session -t "$SCHEDULER_SESSION"
                echo "Scheduler stopped"
            fi
            echo "All sessions killed"
            ;;
        "config")
            # Show current configuration
            echo "=== Current Configuration ==="
            echo "Config file: $CONFIG_FILE"
            echo "Workspace: $WORKSPACE_DIR"
            echo "Task name: $TASK_NAME"
            echo "Work directory: $WORK_DIR"
            echo "Config directory: $CONFIG_DIR"
            echo "Script path: $SCRIPT_PATH"
            echo "Python environment: $PYTHON_ENV"
            echo "Output directory: $OUTPUT_DIR"
            echo "Max jobs per GPU: $MAX_JOBS_PER_GPU"
            echo "GPU count: $GPU_COUNT"
            echo "Check interval: $CHECK_INTERVAL seconds"
            echo "Logging: $ENABLE_LOGGING"
            if [[ "$ENABLE_LOGGING" == "true" ]]; then
                echo "Log file: $LOG_FILE"
            fi
            ;;
        "help"|*)
            cat << EOF
Configuration-Driven Training Scheduler (Centralized)

Usage: scheduler_main.sh CONFIG_FILE WORKSPACE_DIR COMMAND [OPTIONS]

COMMANDS:
    start CONFIG_LIST GPU_LIST  Start scheduler with job queue
    status                      Show current status
    stop                        Stop scheduler gracefully
    kill-all                    Emergency stop all sessions
    config                      Show current configuration
    help                        Show this help

EXAMPLES:
    # Start with specific configs and GPU assignment
    scheduler_main.sh config.yml /path/to/workspace start "task1.yml,task2.yml" "0,1"
    
    # Start with all configs in directory
    scheduler_main.sh config.yml /path/to/workspace start
    
    # Check status
    scheduler_main.sh config.yml /path/to/workspace status
    
    # Stop everything
    scheduler_main.sh config.yml /path/to/workspace kill-all

CONFIGURATION:
    Create scheduler_config.yml in your workspace with task settings
    Create lightweight scheduler.sh wrapper to call this script

FEATURES:
    - Centralized implementation in src/scripts/
    - Workspace-independent operation
    - Configuration-driven design
    - Reliable task scheduling with GPU limits
    - Persistent sessions survive terminal disconnection
    - Real-time monitoring and logging
EOF
            ;;
    esac
}

# Main execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being run directly
    run_scheduler "$@"
fi