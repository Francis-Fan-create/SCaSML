#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_all_experiments.sh [options]

Run every experiment_run.py script in this repository sequentially. Logs for each
run are stored under logs/experiment_runs/ with timestamped filenames.

Options:
  -n, --dry-run           Print the commands that would be executed without running them.
  -s, --stop-on-failure   Abort immediately if any experiment exits with a non-zero code.
      --python <cmd>      Python interpreter to use (default: value of PYTHON_CMD env var or "python").
  -h, --help              Show this help message and exit.

Environment:
  PYTHON_CMD   Overrides the default python interpreter when --python is not supplied.

Examples:
  ./run_all_experiments.sh
  ./run_all_experiments.sh --python python3.11 --stop-on-failure
  PYTHON_CMD=$(which python) ./run_all_experiments.sh -n
EOF
}

DRY_RUN=0
STOP_ON_FAILURE=0
PYTHON_CMD=${PYTHON_CMD:-python}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -s|--stop-on-failure)
            STOP_ON_FAILURE=1
            shift
            ;;
        --python)
            if [[ $# -lt 2 ]]; then
                echo "Error: --python requires an argument." >&2
                exit 64
            fi
            PYTHON_CMD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 64
            ;;
    esac
done

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo "Error: python interpreter '$PYTHON_CMD' not found." >&2
    exit 127
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$SCRIPT_DIR/logs/experiment_runs"
mkdir -p "$LOG_DIR"

# Ordered list of all experiment_run.py scripts
EXPERIMENTS=(
    "results/Grad_Dependent_Nonlinear/20d/experiment_run.py"
    "results/Grad_Dependent_Nonlinear/40d/experiment_run.py"
    "results/Grad_Dependent_Nonlinear/60d/experiment_run.py"
    "results/Grad_Dependent_Nonlinear/80d/experiment_run.py"
    "results_full_history/Grad_Dependent_Nonlinear/20d/experiment_run.py"
    "results_full_history/Grad_Dependent_Nonlinear/40d/experiment_run.py"
    "results_full_history/Grad_Dependent_Nonlinear/60d/experiment_run.py"
    "results_full_history/Grad_Dependent_Nonlinear/80d/experiment_run.py"
    "results_full_history/Linear_Convection_Diffusion/10d/experiment_run.py"
    "results_full_history/Linear_Convection_Diffusion/20d/experiment_run.py"
    "results_full_history/Linear_Convection_Diffusion/30d/experiment_run.py"
    "results_full_history/Linear_Convection_Diffusion/60d/experiment_run.py"
    "results_full_history/LQG/100d/experiment_run.py"
    "results_full_history/LQG/120d/experiment_run.py"
    "results_full_history/LQG/140d/experiment_run.py"
    "results_full_history/LQG/160d/experiment_run.py"
    "results_full_history/Oscillating_Solution/100d/experiment_run.py"
    "results_full_history/Oscillating_Solution/120d/experiment_run.py"
    "results_full_history/Oscillating_Solution/140d/experiment_run.py"
    "results_full_history/Oscillating_Solution/160d/experiment_run.py"
)

SUCCESS_LIST=()
FAIL_LIST=()
TOTAL=0

for REL_PATH in "${EXPERIMENTS[@]}"; do
    TOTAL=$((TOTAL + 1))
    ABS_PATH="$SCRIPT_DIR/$REL_PATH"

    if [[ ! -f "$ABS_PATH" ]]; then
        echo "[SKIP] Missing file: $REL_PATH" >&2
        FAIL_LIST+=("$REL_PATH (missing)")
        [[ $STOP_ON_FAILURE -eq 1 ]] && break
        continue
    fi

    LOG_FILE="$LOG_DIR/$(echo "$REL_PATH" | tr '/ ' '__')_$(date +%Y%m%d_%H%M%S).log"
    CMD=("$PYTHON_CMD" "$ABS_PATH")

    echo "[RUN ] $REL_PATH"
    echo "       Log: $LOG_FILE"

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '       Command:'
        for arg in "${CMD[@]}"; do
            printf ' %q' "$arg"
        done
        printf '\n'
        SUCCESS_LIST+=("$REL_PATH (dry-run)")
        continue
    fi

    START_TS=$(date --iso-8601=seconds)
    echo "$START_TS :: START :: $REL_PATH" >"$LOG_FILE"

    if "${CMD[@]}" >>"$LOG_FILE" 2>&1; then
        STATUS=0
    else
        STATUS=$?
    fi

    END_TS=$(date --iso-8601=seconds)
    echo "$END_TS :: END :: $REL_PATH (status $STATUS)" >>"$LOG_FILE"

    if [[ $STATUS -eq 0 ]]; then
        echo "[OK  ] $REL_PATH"
        SUCCESS_LIST+=("$REL_PATH")
    else
        echo "[FAIL] $REL_PATH (exit code $STATUS)" >&2
        FAIL_LIST+=("$REL_PATH (exit $STATUS)")
        if [[ $STOP_ON_FAILURE -eq 1 ]]; then
            echo "Stopping early due to failure." >&2
            break
        fi
    fi
    echo
    sleep 1

done

printf '\n==== Summary ====\n'
echo "Total experiments: $TOTAL"
echo "Succeeded: ${#SUCCESS_LIST[@]}"
echo "Failed: ${#FAIL_LIST[@]}"

if [[ ${#SUCCESS_LIST[@]} -gt 0 ]]; then
    printf '\nSuccessful runs:\n'
    for item in "${SUCCESS_LIST[@]}"; do
        echo "  - $item"
    done
fi

if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
    printf '\nFailed runs:\n'
    for item in "${FAIL_LIST[@]}"; do
        echo "  - $item"
    done
fi

if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
    exit 1
fi

exit 0
