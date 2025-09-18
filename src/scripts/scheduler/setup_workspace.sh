#!/bin/bash

# Workspace Scheduler Setup Script
# This script sets up a new workspace with the centralized scheduler system

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat << EOF
Scheduler Workspace Setup

Usage: $0 WORKSPACE_DIR [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -f, --force         Overwrite existing files

DESCRIPTION:
    Sets up a workspace directory with the centralized scheduler system.
    Creates scheduler.sh wrapper and scheduler_config.yml template.

EXAMPLES:
    # Setup new workspace
    $0 /path/to/my/workspace
    
    # Force overwrite existing setup
    $0 /path/to/my/workspace --force

WHAT IT CREATES:
    - scheduler.sh         Lightweight wrapper script
    - scheduler_config.yml Configuration file (from template)
    - out/                 Directory for completion files
    - configs/             Directory for task configurations (if needed)

After setup, customize scheduler_config.yml for your workspace needs.
EOF
}

setup_workspace() {
    local workspace_dir="$1"
    local force_overwrite="$2"
    
    # Validate workspace directory
    if [[ -z "$workspace_dir" ]]; then
        echo "Error: Workspace directory not specified"
        show_help
        exit 1
    fi
    
    # Create workspace directory if it doesn't exist
    if [[ ! -d "$workspace_dir" ]]; then
        echo "Creating workspace directory: $workspace_dir"
        mkdir -p "$workspace_dir"
    fi
    
    # Convert to absolute path
    workspace_dir="$(cd "$workspace_dir" && pwd)"
    
    echo "Setting up scheduler in workspace: $workspace_dir"
    
    # Check for existing files
    local scheduler_script="$workspace_dir/scheduler.sh"
    local config_file="$workspace_dir/scheduler_config.yml"
    
    if [[ -f "$scheduler_script" ]] && [[ "$force_overwrite" != "true" ]]; then
        echo "Error: scheduler.sh already exists. Use --force to overwrite."
        exit 1
    fi
    
    if [[ -f "$config_file" ]] && [[ "$force_overwrite" != "true" ]]; then
        echo "Error: scheduler_config.yml already exists. Use --force to overwrite."
        exit 1
    fi
    
    # Copy wrapper script
    echo "Creating scheduler.sh wrapper..."
    cp "$SCRIPT_DIR/scheduler_wrapper_template.sh" "$scheduler_script"
    chmod +x "$scheduler_script"
    
    # Copy config template
    echo "Creating scheduler_config.yml from template..."
    cp "$SCRIPT_DIR/scheduler_config_template.yml" "$config_file"
    
    # Create output directory
    mkdir -p "$workspace_dir/out"
    
    # Create configs directory if it doesn't exist
    if [[ ! -d "$workspace_dir/configs" ]]; then
        mkdir -p "$workspace_dir/configs"
        echo "Created configs/ directory for task configurations"
    fi
    
    echo ""
    echo "✅ Workspace setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Edit $config_file to configure your workspace"
    echo "2. Create task configuration files in the configs/ directory"
    echo "3. Run './scheduler.sh help' to see available commands"
    echo ""
    echo "Files created:"
    echo "  - scheduler.sh (executable wrapper)"
    echo "  - scheduler_config.yml (configuration template)"
    echo "  - out/ (completion files directory)"
    echo "  - configs/ (task configurations directory)"
}

# Parse command line arguments
WORKSPACE_DIR=""
FORCE_OVERWRITE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE_OVERWRITE="true"
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$WORKSPACE_DIR" ]]; then
                WORKSPACE_DIR="$1"
            else
                echo "Error: Multiple workspace directories specified"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Run setup
setup_workspace "$WORKSPACE_DIR" "$FORCE_OVERWRITE"