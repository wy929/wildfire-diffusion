#!/bin/bash

# Core Training Scheduler - Configuration-driven design
# This is the centralized implementation that workspaces call

# Function to extract root_dir from a task config file
get_task_root_dir() {
    local config_file="$1"
    if [[ -f "$config_file" ]]; then
        grep "^root_dir:" "$config_file" | sed 's/root_dir: *//' | tr -d '"' || echo ""
    else
        echo ""
    fi
}

# Function to parse YAML config (simple parser for our needs)
parse_config() {
    local config_file="$1"
    local workspace_dir="$2"
    
    # Read configuration values
    TASK_NAME=$(grep "^task_name:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    WORK_DIR=$(grep "^work_dir:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    CONFIG_DIR=$(grep "^config_dir:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    SCRIPT_PATH=$(grep "^script_path:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    PYTHON_ENV=$(grep "^python_env:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    
    SCHEDULER_SESSION_SUFFIX=$(grep "^scheduler_session_suffix:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    WORKER_PREFIX=$(grep "^worker_prefix:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    
    MAX_JOBS_PER_GPU=$(grep "^max_jobs_per_gpu:" "$config_file" | awk '{print $2}')
    GPU_COUNT=$(grep "^gpu_count:" "$config_file" | awk '{print $2}')
    
    CHECK_INTERVAL=$(grep "^check_interval:" "$config_file" | awk '{print $2}')
    STARTUP_DELAY=$(grep "^startup_delay:" "$config_file" | awk '{print $2}')
    COMMAND_DELAY=$(grep "^command_delay:" "$config_file" | awk '{print $2}')
    
    LOG_DIR=$(grep "^log_dir:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    ENABLE_LOGGING=$(grep "^enable_logging:" "$config_file" | awk '{print $2}')
    
    CONFIG_PATTERN=$(grep "^config_pattern:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    CONDA_ENV=$(grep "^conda_env:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    CUDA_VAR=$(grep "^cuda_visible_devices_var:" "$config_file" | cut -d'"' -f2 | tr -d ' ')
    
    AUTO_RESTART=$(grep "^auto_restart_failed:" "$config_file" | awk '{print $2}')
    MAX_RETRIES=$(grep "^max_retries:" "$config_file" | awk '{print $2}')
    CLEANUP_ON_COMPLETE=$(grep "^cleanup_on_complete:" "$config_file" | awk '{print $2}')
    
    # Set defaults if empty
    TASK_NAME="${TASK_NAME:-default_task}"
    WORKER_PREFIX="${WORKER_PREFIX:-$TASK_NAME}"
    SCHEDULER_SESSION_SUFFIX="${SCHEDULER_SESSION_SUFFIX:-_scheduler}"
    MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
    GPU_COUNT="${GPU_COUNT:-2}"
    CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
    STARTUP_DELAY="${STARTUP_DELAY:-1}"
    COMMAND_DELAY="${COMMAND_DELAY:-0.5}"
    LOG_DIR="${LOG_DIR:-/tmp}"
    ENABLE_LOGGING="${ENABLE_LOGGING:-true}"
    CONFIG_PATTERN="${CONFIG_PATTERN:-*.yml}"
    CONDA_ENV="${CONDA_ENV:-acse}"
    CUDA_VAR="${CUDA_VAR:-CUDA_VISIBLE_DEVICES}"
    AUTO_RESTART="${AUTO_RESTART:-false}"
    MAX_RETRIES="${MAX_RETRIES:-3}"
    CLEANUP_ON_COMPLETE="${CLEANUP_ON_COMPLETE:-false}"
    
    # Perform variable substitution for paths containing ${task_name}
    CONFIG_DIR=$(echo "$CONFIG_DIR" | sed "s/\${task_name}/$TASK_NAME/g")
    SCRIPT_PATH=$(echo "$SCRIPT_PATH" | sed "s/\${task_name}/$TASK_NAME/g")
    
    # Ensure paths are absolute
    if [[ "$CONFIG_DIR" != /* ]]; then
        CONFIG_DIR="$workspace_dir/$CONFIG_DIR"
    fi
    if [[ "$SCRIPT_PATH" != /* ]]; then
        SCRIPT_PATH="$workspace_dir/$SCRIPT_PATH"
    fi
    
    # Derived values
    SCHEDULER_SESSION="${TASK_NAME}${SCHEDULER_SESSION_SUFFIX}"
    JOB_QUEUE="$LOG_DIR/${TASK_NAME}_job_queue.txt"
    RUNNING_JOBS="$LOG_DIR/${TASK_NAME}_running_jobs.txt"
    COMPLETED_JOBS="$LOG_DIR/${TASK_NAME}_completed_jobs.txt"
    FAILED_JOBS="$LOG_DIR/${TASK_NAME}_failed_jobs.txt"
    LOG_FILE="$LOG_DIR/${TASK_NAME}_scheduler.log"
}

# Function to log messages
log() {
    local message="$1"
    local timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
    
    if [[ "$ENABLE_LOGGING" == "true" ]]; then
        echo "$timestamp $message" | tee -a "$LOG_FILE"
    else
        echo "$timestamp $message"
    fi
}

# Function to initialize job queue
init_queue() {
    local config_list="$1"
    local gpu_assignment="$2"
    
    # Clear existing queue files
    > "$JOB_QUEUE"
    > "$RUNNING_JOBS" 
    > "$COMPLETED_JOBS"
    > "$FAILED_JOBS"
    
    if [[ -n "$config_list" ]]; then
        # Parse config list and GPU assignments
        IFS=',' read -ra configs <<< "$config_list"
        IFS=',' read -ra gpus <<< "$gpu_assignment"
        
        for i in "${!configs[@]}"; do
            local config="${configs[$i]}"
            local gpu_id="${gpus[$i]:-$((i % GPU_COUNT))}"
            local config_path="$CONFIG_DIR/$config"
            
            if [[ -f "$config_path" ]]; then
                echo "$config_path:$gpu_id" >> "$JOB_QUEUE"
                log "Queued: $(basename "$config" .yml) on GPU $gpu_id"
            else
                log "WARNING: Config file not found: $config_path"
            fi
        done
    else
        # Use all configs in directory
        local count=0
        for config_file in "$CONFIG_DIR"/$CONFIG_PATTERN; do
            if [[ -f "$config_file" ]]; then
                local gpu_id=$((count % GPU_COUNT))
                echo "$config_file:$gpu_id" >> "$JOB_QUEUE"
                log "Queued: $(basename "$config_file" .yml) on GPU $gpu_id"
                count=$((count + 1))
            fi
        done
    fi
    
    log "Job queue initialized with $(wc -l < "$JOB_QUEUE") jobs"
}

# Function to count running jobs on a specific GPU
count_gpu_jobs() {
    local gpu_id="$1"
    grep ":$gpu_id$" "$RUNNING_JOBS" 2>/dev/null | wc -l
}

# Function to start a training job
start_job() {
    local job_line="$1"
    local config_path="${job_line%:*}"
    local gpu_id="${job_line#*:}"
    
    local task_name=$(basename "$config_path" .yml)
    local session_name="${WORKER_PREFIX}_${task_name}_gpu${gpu_id}"
    
    log "Starting job: $task_name on GPU $gpu_id"
    
    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log "WARNING: Session $session_name already exists, killing old session"
        tmux kill-session -t "$session_name" 2>/dev/null
        sleep "$STARTUP_DELAY"
    fi
    
    # Create worker session
    tmux new-session -d -s "$session_name"
    sleep "$STARTUP_DELAY"
    
    # Set up environment and start training
    tmux send-keys -t "$session_name" "cd $WORK_DIR" Enter
    sleep "$COMMAND_DELAY"
    tmux send-keys -t "$session_name" "conda activate $CONDA_ENV" Enter
    sleep "$COMMAND_DELAY"
    tmux send-keys -t "$session_name" "export $CUDA_VAR=$gpu_id" Enter
    sleep "$COMMAND_DELAY"
    # Get task-specific output directory from config
    local task_root_dir=$(get_task_root_dir "$config_path")
    local completion_file
    if [[ -n "$task_root_dir" ]]; then
        completion_file="$task_root_dir/${task_name}_completed.txt"
    else
        log "ERROR: No root_dir found in config file: $config_path"
        return 1
    fi
    
    # Build command to run training and create completion file
    local train_cmd="$PYTHON_ENV $SCRIPT_PATH --config_path $config_path"
    local completion_cmd="echo 'Training completed at \$(date)' > '$completion_file'"
    tmux send-keys -t "$session_name" "$train_cmd && $completion_cmd; exit" Enter
    
    # Add to running jobs
    echo "$job_line" >> "$RUNNING_JOBS"
    log "Job started: $session_name"
    
    return 0
}

# Function to check job status and handle completion
check_jobs() {
    if [[ ! -f "$RUNNING_JOBS" ]]; then
        return 0
    fi
    
    local temp_running="$LOG_DIR/${TASK_NAME}_running_temp.txt"
    > "$temp_running"
    
    while IFS= read -r job_line; do
        if [[ -z "$job_line" ]]; then continue; fi
        
        local config_path="${job_line%:*}"
        local gpu_id="${job_line#*:}"
        local task_name=$(basename "$config_path" .yml)
        local session_name="${WORKER_PREFIX}_${task_name}_gpu${gpu_id}"
        
        # Get task-specific output directory from config
        local task_root_dir=$(get_task_root_dir "$config_path")
        local completion_file
        if [[ -n "$task_root_dir" ]]; then
            completion_file="$task_root_dir/${task_name}_completed.txt"
        else
            # Skip task without root_dir - mark as failed
            log "Job failed (no root_dir in config): $task_name"
            echo "$job_line" >> "$FAILED_JOBS"
            continue
        fi
        
        if [[ -f "$completion_file" ]]; then
            # Task completed - cleanup session and mark as completed
            log "Job completed: $task_name (completion file found)"
            echo "$job_line" >> "$COMPLETED_JOBS"
            tmux kill-session -t "$session_name" 2>/dev/null
        elif tmux has-session -t "$session_name" 2>/dev/null; then
            # Check if training process is still running
            local python_pids=$(tmux list-panes -t "$session_name" -F "#{pane_pid}" 2>/dev/null | xargs -I {} pgrep -P {} python 2>/dev/null || true)
            
            if [[ -n "$python_pids" ]]; then
                # Python process still running - task is active
                echo "$job_line" >> "$temp_running"
            else
                # No Python process but session exists - check if just finished
                sleep 2  # Give a moment for completion file to be written
                if [[ -f "$completion_file" ]]; then
                    log "Job completed: $task_name (just finished)"
                    echo "$job_line" >> "$COMPLETED_JOBS"
                    tmux kill-session -t "$session_name" 2>/dev/null
                else
                    # Session exists with no python and no completion - might have failed
                    echo "$job_line" >> "$temp_running"
                fi
            fi
        else
            # Session doesn't exist - job completed or failed
            if [[ -f "$completion_file" ]]; then
                log "Job completed: $task_name"
                echo "$job_line" >> "$COMPLETED_JOBS"
            else
                log "Job failed (no completion file): $task_name"
                echo "$job_line" >> "$FAILED_JOBS"
            fi
        fi
    done < "$RUNNING_JOBS"
    
    # Update running jobs file
    mv "$temp_running" "$RUNNING_JOBS"
}

# Function to schedule next jobs
schedule_jobs() {
    if [[ ! -f "$JOB_QUEUE" ]] || [[ ! -s "$JOB_QUEUE" ]]; then
        return 0
    fi
    
    local temp_queue="$LOG_DIR/${TASK_NAME}_queue_temp.txt"
    > "$temp_queue"
    local scheduled=0
    
    while IFS= read -r job_line; do
        if [[ -z "$job_line" ]]; then continue; fi
        
        local gpu_id="${job_line#*:}"
        
        # Check if this GPU has available slots
        if [[ $(count_gpu_jobs "$gpu_id") -lt $MAX_JOBS_PER_GPU ]]; then
            # Start the job
            if start_job "$job_line"; then
                scheduled=$((scheduled + 1))
                log "Scheduled job from queue on GPU $gpu_id"
            else
                # Failed to start, put back in queue
                echo "$job_line" >> "$temp_queue"
            fi
        else
            # GPU full, keep in queue
            echo "$job_line" >> "$temp_queue"
        fi
    done < "$JOB_QUEUE"
    
    # Update queue
    mv "$temp_queue" "$JOB_QUEUE"
    
    if [[ $scheduled -gt 0 ]]; then
        log "Scheduled $scheduled new jobs"
    fi
}

# Function to show status
show_status() {
    echo "=== $TASK_NAME Training Scheduler Status ==="
    echo "Config: $CONFIG_FILE"
    echo "Queue: $(wc -l < "$JOB_QUEUE" 2>/dev/null || echo 0) jobs waiting"
    echo "Running: $(wc -l < "$RUNNING_JOBS" 2>/dev/null || echo 0) jobs active"
    echo "Completed: $(wc -l < "$COMPLETED_JOBS" 2>/dev/null || echo 0) jobs finished"
    echo "Failed: $(wc -l < "$FAILED_JOBS" 2>/dev/null || echo 0) jobs failed"
    echo ""
    
    for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
        echo "GPU $gpu jobs: $(count_gpu_jobs "$gpu")/$MAX_JOBS_PER_GPU"
    done
    
    echo ""
    echo "Active sessions:"
    tmux list-sessions 2>/dev/null | grep "^${WORKER_PREFIX}_" || echo "  No active training sessions"
    
    if [[ "$ENABLE_LOGGING" == "true" ]]; then
        echo ""
        echo "Log file: $LOG_FILE"
    fi
}

# Main scheduler loop
main_loop() {
    log "Scheduler started with config: $CONFIG_FILE"
    
    while true; do
        # Check status of running jobs
        check_jobs
        
        # Schedule new jobs if possible
        schedule_jobs
        
        # Check if we're done
        local queue_size=$(wc -l < "$JOB_QUEUE" 2>/dev/null || echo 0)
        local running_size=$(wc -l < "$RUNNING_JOBS" 2>/dev/null || echo 0)
        
        if [[ $queue_size -eq 0 ]] && [[ $running_size -eq 0 ]]; then
            log "All jobs completed!"
            show_status
            break
        fi
        
        # Wait before next check
        sleep "$CHECK_INTERVAL"
    done
}

# Export functions for external use
export -f parse_config log init_queue count_gpu_jobs start_job check_jobs schedule_jobs show_status main_loop