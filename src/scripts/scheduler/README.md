# Centralized Training Scheduler System

A centralized, configuration-driven system for managing multiple training tasks across GPUs. The scheduler logic is centralized in `src/scripts/scheduler/` and can be used from any workspace with lightweight wrappers.

## Architecture

```
Project Structure:
src/scripts/scheduler/
├── scheduler_core.sh           # Core scheduling logic
├── scheduler_main.sh           # Main entry point
├── setup_workspace.sh          # Workspace setup tool
├── scheduler_config_template.yml    # Configuration template
├── scheduler_wrapper_template.sh   # Workspace wrapper template
└── README.md                   # This documentation

Workspace Structure:
workspace_dir/
├── scheduler.sh                # Lightweight wrapper (calls centralized system)
├── scheduler_config.yml        # Workspace-specific configuration
├── configs/                    # Task configuration files
├── out/                        # Completion markers
└── [training scripts & data]
```

## Key Features

✅ **Centralized Logic**: All implementation in `src/scripts/scheduler/`  
✅ **Workspace Independence**: Each workspace has minimal setup  
✅ **Configuration-Driven**: YAML-based configuration  
✅ **GPU Resource Management**: Automatic allocation and concurrency control  
✅ **Queue-Based Scheduling**: Tasks wait for available GPU slots  
✅ **Completion Detection**: File-based and process-based detection  
✅ **Session Management**: Persistent tmux sessions  

## Quick Setup

### 1. Setup a New Workspace
```bash
# Use the setup script to create a new workspace
/path/to/project/src/scripts/scheduler/setup_workspace.sh /path/to/my/workspace

# Or with force overwrite
/path/to/project/src/scripts/scheduler/setup_workspace.sh /path/to/my/workspace --force
```

### 2. Configure Your Workspace
Edit the generated `scheduler_config.yml`:
```yaml
# Basic Configuration
task_name: "my_experiment"
work_dir: "/path/to/project/root"

# Paths (relative to workspace or absolute)
config_dir: "configs"
script_path: "train.py"
python_env: "/path/to/python"

# GPU Configuration  
max_jobs_per_gpu: 2
gpu_count: 2

# Environment
conda_env: "myenv"
output_dir: "out"
```

### 3. Create Task Configurations
Add your training task configs in the `configs/` directory:
```yaml
# configs/experiment1.yml
task_name: "experiment1"
model:
  type: "transformer"
  layers: 12
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
```

### 4. Run Training Tasks
```bash
cd /path/to/my/workspace

# Start with specific configs and GPU assignment
./scheduler.sh start "exp1.yml,exp2.yml,exp3.yml" "0,0,1"

# Start with all configs (auto GPU assignment)
./scheduler.sh start

# Check status
./scheduler.sh status

# Stop all tasks
./scheduler.sh kill-all
```

## Commands

### Workspace Commands
```bash
# From any workspace directory with scheduler.sh

./scheduler.sh start [CONFIG_LIST] [GPU_LIST]  # Start training
./scheduler.sh status                          # Show status
./scheduler.sh stop                           # Stop scheduler
./scheduler.sh kill-all                       # Emergency stop
./scheduler.sh config                         # Show configuration
./scheduler.sh help                           # Show help
```

### Direct Centralized Commands
```bash
# Calling centralized scheduler directly

/path/to/src/scripts/scheduler/scheduler_main.sh CONFIG_FILE WORKSPACE_DIR COMMAND [ARGS]

# Setup new workspace
/path/to/src/scripts/scheduler/setup_workspace.sh WORKSPACE_DIR [--force]
```

## Configuration Reference

### scheduler_config.yml
```yaml
# Basic Configuration
task_name: "experiment_name"        # Name for sessions and logs
work_dir: "/project/root"           # Project root directory

# Paths (relative to workspace or absolute)
config_dir: "configs"               # Task config directory  
script_path: "train.py"             # Training script
python_env: "/path/to/python"       # Python executable
output_dir: "out"                   # Completion files directory

# Session Configuration
scheduler_session_suffix: "_scheduler"  # Scheduler session suffix
worker_prefix: ""                       # Worker prefix (defaults to task_name)

# GPU Configuration
max_jobs_per_gpu: 2                 # Max concurrent tasks per GPU
gpu_count: 2                        # Number of GPUs

# Scheduling Configuration  
check_interval: 30                  # Status check interval (seconds)
startup_delay: 1                    # Delay between session steps
command_delay: 0.5                  # Delay between tmux commands

# Logging
log_dir: "/tmp"                     # Log directory
enable_logging: true                # Enable file logging

# Environment Setup
conda_env: "base"                   # Conda environment
cuda_visible_devices_var: "CUDA_VISIBLE_DEVICES"
config_pattern: "*.yml"             # Config file pattern

# Advanced Options
auto_restart_failed: false          # Auto-restart failed tasks
max_retries: 3                      # Max retry attempts
cleanup_on_complete: false          # Auto-cleanup completed sessions
```

## Usage Examples

### Example 1: Basic ML Experiment
```bash
# Setup workspace
./setup_workspace.sh /experiments/image_classification

cd /experiments/image_classification

# Edit scheduler_config.yml for your project
# Create configs/resnet.yml, configs/vgg.yml, etc.

# Run all experiments with balanced GPU allocation
./scheduler.sh start
```

### Example 2: GPU-Specific Assignment
```bash
# Run specific experiments with custom GPU assignment
./scheduler.sh start "heavy_model.yml,light_model.yml,test_model.yml" "1,0,0"
# heavy_model on GPU 1, light_model and test_model on GPU 0
```

### Example 3: High-Throughput Training
```bash
# Configure for maximum throughput
# In scheduler_config.yml:
# max_jobs_per_gpu: 4
# check_interval: 10

./scheduler.sh start "batch1.yml,batch2.yml,batch3.yml,batch4.yml,batch5.yml,batch6.yml,batch7.yml,batch8.yml" "0,0,0,0,1,1,1,1"
```

### Example 4: Multi-Workspace Setup
```bash
# Setup multiple workspaces for different projects
./setup_workspace.sh /experiments/nlp_project
./setup_workspace.sh /experiments/cv_project  
./setup_workspace.sh /experiments/rl_project

# Each workspace operates independently with its own:
# - Configuration
# - Task queue
# - Completion tracking
# - Session management
```

## Monitoring and Debugging

### Status Monitoring
```bash
./scheduler.sh status              # Quick status
tmux attach -t TASK_scheduler     # View scheduler session
tmux attach -t TASK_exp1_gpu0     # View specific training task
```

### Log Files
```bash
tail -f /tmp/TASK_scheduler.log   # Scheduler logs
tail -f /tmp/train_logs/exp1.log  # Training logs (if configured)
```

### Troubleshooting
```bash
# Check active sessions
tmux list-sessions | grep TASK

# Emergency cleanup
./scheduler.sh kill-all

# Verify configuration
./scheduler.sh config

# Check completion files
ls -la out/
```

## Advanced Features

### Custom Training Script Integration
Your training script should:
1. Accept `--config_path` parameter
2. Create completion marker file when finished
3. Handle GPU assignment via `CUDA_VISIBLE_DEVICES`

```python
# Example integration
import argparse
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    args = parser.parse_args()
    
    # Your training logic here
    train_model(args.config_path)
    
    # Create completion marker
    task_name = os.path.splitext(os.path.basename(args.config_path))[0]
    workspace = os.path.dirname(os.path.abspath(__file__))
    completion_file = f"{workspace}/out/{task_name}_completed.txt"
    
    with open(completion_file, 'w') as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Completed at: {datetime.now()}\n")
        # Add other relevant info
```

### Custom Completion Detection
The scheduler detects completion via:
1. **Primary**: Completion marker files in `output_dir/`
2. **Secondary**: Process monitoring (no Python processes)
3. **Fallback**: Session existence checking

### Extending the System
- **Custom Config Parsers**: Modify `parse_config()` in `scheduler_core.sh`
- **Alternative Training Scripts**: Update `script_path` in config
- **Different Environments**: Modify environment setup in `start_job()`
- **Custom Session Management**: Extend tmux session handling

## Migration from Workspace-Specific Schedulers

### From ddim_demo/scheduler.sh
```bash
# Old approach (workspace-specific)
cd workspace/ddim_demo
./scheduler.sh start

# New approach (centralized)
cd workspace/ddim_demo
# Copy centralized wrapper and config
cp /src/scripts/scheduler/scheduler_wrapper_template.sh ./scheduler.sh
cp /src/scripts/scheduler/scheduler_config_template.yml ./scheduler_config.yml
# Customize config and run
./scheduler.sh start
```

### Automatic Migration
Use the setup script to convert existing workspaces:
```bash
./setup_workspace.sh /existing/workspace --force
```

## Performance and Scalability

### Recommended Settings
- **Light workloads**: `max_jobs_per_gpu: 4`, `check_interval: 10`
- **Heavy workloads**: `max_jobs_per_gpu: 1`, `check_interval: 60`
- **Mixed workloads**: `max_jobs_per_gpu: 2`, `check_interval: 30`

### Resource Management
- GPU memory is managed via concurrency limits
- CPU usage scales with number of concurrent tasks
- Disk I/O optimized through intelligent task scheduling
- Network overhead minimal (local tmux sessions)

## Security and Isolation

- Each workspace operates independently
- Sessions are namespaced by task name
- No cross-workspace interference  
- Secure temporary file handling
- Clean session teardown

This centralized system provides maximum flexibility while maintaining simplicity for users. Each workspace only needs two files (`scheduler.sh` + `scheduler_config.yml`) to access the full power of the centralized scheduling system.