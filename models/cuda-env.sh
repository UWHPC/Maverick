# Sourced by the training and capture launchers.
maverick_cuda_setup() {
    local scratch=$1 venv
    if [[ -n ${MAVERICK_PYTHON:-} ]]; then
        return
    fi

    venv=${MAVERICK_VENV:-"$scratch/venv"}
    mkdir -p "$venv"
    (
        flock 9
        if [[ ! -f "$venv/.maverick-cu128" ]]; then
            export PIP_CACHE_DIR="$scratch/pip-cache"
            "${MAVERICK_BOOTSTRAP_PYTHON:-python3}" -m venv "$venv"
            "$venv/bin/python" -m pip install torch==2.8.0 \
                --index-url https://download.pytorch.org/whl/cu128
            "$venv/bin/python" -m pip install torch-geometric==2.7.0 ogb==1.3.6 \
                safetensors==0.6.2 packaging
            "$venv/bin/python" -m pip install --no-index --only-binary=:all: \
                --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html \
                pyg_lib==0.4.0+pt28cu128
            touch "$venv/.maverick-cu128"
        fi
    ) 9>"$venv/.install.lock"
    export MAVERICK_PYTHON="$venv/bin/python"
}
