#!/bin/bash

if [ ! -d .git ]; then
    echo "This script must be run from the root of the repository!"
    exit 1
fi

#################### EXPERIMENT VARIABLES ####################
MODEL_ID=$1
TASK_NAME=$2

OUTPUT_ROOT_P=${OUTPUT_ROOT_P:-"${PWD}/results"}
SEED=${SEED:-1312}
WANDB_LOGGING=${WANDB_LOGGING:-"True"}
WANDB_PROJECT=${WANDB_PROJECT:-"gimmick"}
#################### AUXILIARY VARIABLES #####################
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PWD}/src"
export TOKENIZERS_PARALLELISM="true"
export HF_TOKEN=${HF_TOKEN:-"PUT YOUR KEY HERE"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-"PUT YOUR KEY HERE"}
export GOOGLE_SERVICE_ACCOUNT_FILE=${GOOGLE_SERVICE_ACCOUNT_FILE:-"PUT YOUR CRED FILE HERE"}
export GOOGLE_PROJECT_ID=${GOOGLE_PROJECT_ID:-"PUT YOUR G PROJ ID HERE"}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-"PUT YOUR KEY HERE"}

MAMBA_ENV_NAME=${MAMBA_ENV_NAME:-"gimmick"}
MAMBA_PREFIX="${HOME}/miniforge3"

VERBOSE=${VERBOSE:-0}
YES=${YES:-"False"}
##############################################################

init_mamba() {
    if [ ! -f ${MAMBA_PREFIX}/bin/conda ]; then
        echo "Conda not found at ${MAMBA_PREFIX}/bin/conda! Exiting."
        exit 1
    fi

    # >>> conda initialize >>>
    CONDA_SETUP="${MAMBA_PREFIX}/bin/conda 'shell.bash' 'hook'"
    __conda_setup=$("${CONDA_SETUP}" 2>/dev/null)
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "${MAMBA_PREFIX}/etc/profile.d/conda.sh" ]; then
            . "${MAMBA_PREFIX}/etc/profile.d/conda.sh"
        else
            export PATH="${MAMBA_PREFIX}/bin:${PATH}"
        fi
    fi
    unset __conda_setup

    if [ -f "${MAMBA_PREFIX}/etc/profile.d/mamba.sh" ]; then
        . "${MAMBA_PREFIX}/etc/profile.d/mamba.sh"
    fi
    # <<< conda initialize <<<
}

activate_mamba_env() {
    mamba activate ${MAMBA_ENV_NAME} || {
        echo "Failed to activate Conda environment ${MAMBA_ENV_NAME}! Exiting."
        exit 1
    }
}

print_python_env_info() {
    echo ""
    echo ""
    echo ""
    echo "##################### PYTHON ENV '${MAMBA_ENV_NAME}' INFO START #####################"
    echo "Python version: $(python -c 'import sys; print(sys.version)')"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "Flash Attention 2 Support: $(python -c 'import importlib.util; import torch; fattn="flash_attn_2_cuda";print(importlib.util.find_spec(fattn) is not None)')"
    echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
    echo "##################### PYTHON ENV '${MAMBA_ENV_NAME}' INFO END #####################"
    echo ""
    echo ""
    echo ""
    sleep 1
}

run_gimmick_eval() {
    RUN_COMMAND="
        python src/gimmick/main.py \
        eval \
        --model_id="${MODEL_ID}" \
        --task_name=${TASK_NAME} \
        --output_root=${OUTPUT_ROOT_P} \
        --yes=${YES} \
        --seed=${SEED} \
        --wandb_logging=${WANDB_LOGGING} \
        --wandb_project=${WANDB_PROJECT} \
        $@
    "

    echo "********************************* RUN COMMAND ***********************************"
    echo "$(echo ${RUN_COMMAND} | sed 's/ \+/ \\\n/g')"
    echo "*********************************************************************************"
    echo ""
    sleep 2
    clear

    eval ${RUN_COMMAND}

    if [ $? -ne 0 ]; then
        echo "Error: GIMMICK evaluation failed!"
        exit 1
    fi

    echo "GIMMICK evaluation completed successfully!"
    exit 0
}

# check if MODEL_ID and TASK_NAME are set
if [ -z "${MODEL_ID}" ]; then
    echo "MODEL_ID is not set! Exiting."
    exit 1
fi

if [ -z "${TASK_NAME}" ]; then
    echo "TASK_NAME is not set! Exiting."
    exit 1
fi

init_mamba
activate_mamba_env

if [ ${VERBOSE} -eq 1 ]; then
    print_python_env_info
fi

shift 2

run_gimmick_eval $@
