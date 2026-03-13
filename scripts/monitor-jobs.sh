#!/usr/bin/env bash
# monitor-jobs.sh -- Monitor all running Colab training jobs
#
# Shows a compact status table with progress, FPS, and resource usage
# from each job's progress.json.
#
# Usage:
#   bash scripts/monitor-jobs.sh <URL> <TOKEN>
#   bash scripts/monitor-jobs.sh <URL> <TOKEN> --watch   # refresh every 30s
#
# Requires: curl, jq

set -euo pipefail

URL="${1:?Usage: monitor-jobs.sh <URL> <TOKEN> [--watch]}"
TOKEN="${2:?Usage: monitor-jobs.sh <URL> <TOKEN> [--watch]}"
WATCH="${3:-}"

print_status() {
    # Get all jobs
    local jobs_json
    jobs_json=$(curl -s "$URL/jobs" -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    if [[ -z "$jobs_json" || "$jobs_json" == "null" ]]; then
        echo "ERROR: Could not reach runner API"
        return 1
    fi

    local job_ids
    job_ids=$(echo "$jobs_json" | jq -r 'keys[]' 2>/dev/null)
    if [[ -z "$job_ids" ]]; then
        echo "No jobs found."
        return 0
    fi

    echo "$(date '+%H:%M:%S')  Jobs on $(echo "$URL" | sed 's|https://||;s|\.trycloudflare\.com||')"
    echo ""
    printf "%-8s %-6s %-35s %6s %6s %8s %6s %8s\n" \
        "JOB" "STATUS" "RUN" "PCT" "FPS" "VRAM_MB" "GPU%" "ETA"
    printf "%s\n" "-------------------------------------------------------------------------------------------------------------"

    for jid in $job_ids; do
        local job_info status cmd run_name
        job_info=$(echo "$jobs_json" | jq -r --arg id "$jid" '.[$id]')
        status=$(echo "$job_info" | jq -r '.status // "unknown"')
        cmd=$(echo "$job_info" | jq -r '.cmd // ""')

        # Extract run name from command (config path -> experiment name)
        run_name=$(echo "$cmd" | grep -oP 'configs/\K[^ ]+\.yaml' | sed 's/\.yaml//' || echo "?")

        if [[ "$status" == "running" ]]; then
            # Try to find and read progress.json
            # Derive the run directory from the config name and seed
            local seed
            seed=$(echo "$cmd" | grep -oP '\-\-seed \K[0-9]+' || echo "42")

            # Find the run dir by listing recent dirs matching the config
            local progress=""
            local dirs_json
            dirs_json=$(curl -s "$URL/exec" \
                -H "Authorization: Bearer $TOKEN" \
                -H "Content-Type: application/json" \
                -d "{\"cmd\": \"ls -td experiments/dqn_atari/runs/${run_name}_${seed}_* 2>/dev/null | head -1\"}" 2>/dev/null)
            local run_dir
            run_dir=$(echo "$dirs_json" | jq -r '.stdout // ""' | tr -d '[:space:]')

            if [[ -n "$run_dir" && "$run_dir" != "null" ]]; then
                progress=$(curl -s "$URL/files/thesis/$run_dir/progress.json" \
                    -H "Authorization: Bearer $TOKEN" 2>/dev/null)
            fi

            if [[ -n "$progress" && "$progress" != "null" && "$progress" != *"error"* ]]; then
                local pct fps vram gpu eta_s eta_str
                pct=$(echo "$progress" | jq -r '.percent // "-"')
                fps=$(echo "$progress" | jq -r '.fps // "-"')
                vram=$(echo "$progress" | jq -r '.resources.gpu_memory_used_mb // "-"')
                gpu=$(echo "$progress" | jq -r '.resources.gpu_utilization_pct // "-"')
                eta_s=$(echo "$progress" | jq -r '.eta_seconds // 0')

                # Format ETA as Xh Ym
                if [[ "$eta_s" != "0" && "$eta_s" != "null" ]]; then
                    local eta_m=$((${eta_s%.*} / 60))
                    local eta_h=$((eta_m / 60))
                    eta_m=$((eta_m % 60))
                    eta_str="${eta_h}h${eta_m}m"
                else
                    eta_str="-"
                fi

                printf "%-8s %-6s %-35s %5s%% %6s %8s %5s%% %8s\n" \
                    "$jid" "RUN" "$run_name" "$pct" "$fps" "$vram" "$gpu" "$eta_str"
            else
                printf "%-8s %-6s %-35s %6s %6s %8s %6s %8s\n" \
                    "$jid" "RUN" "$run_name" "..." "..." "..." "..." "..."
            fi
        else
            printf "%-8s %-6s %-35s\n" "$jid" "$status" "$run_name"
        fi
    done

    echo ""

    # Summary line: total VRAM across all running jobs
    local total_vram
    total_vram=$(echo "$jobs_json" | jq -r 'keys[]' | while read jid; do
        local s=$(echo "$jobs_json" | jq -r --arg id "$jid" '.[$id].status // ""')
        [[ "$s" == "running" ]] && echo 1 || true
    done | wc -l | tr -d ' ')
    echo "Running: $total_vram job(s)"
}

if [[ "$WATCH" == "--watch" ]]; then
    while true; do
        clear
        print_status || true
        echo ""
        echo "(refreshing every 30s, Ctrl-C to stop)"
        sleep 30
    done
else
    print_status
fi
