#!/usr/bin/env bash
set -euo pipefail

script_path="${BASH_SOURCE[0]}"
[[ "$script_path" == */* ]] || script_path="./$script_path"
project_root="$(cd "${script_path%/*}" && pwd)"
preferred_python="/d/miniconda/envs/PJT_1/python.exe"

if [[ -x "$preferred_python" ]]; then
    python_exe="$preferred_python"
else
    python_exe="python"
fi

export YOLO_CONFIG_DIR="$project_root/.runtime/ultralytics"
export YOLO_AUTOINSTALL="False"
export YOLO_VERBOSE="False"

cd "$project_root"
exec "$python_exe" "$project_root/mot_app.py" "$@"
