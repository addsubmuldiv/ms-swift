NPU_TORCH_VERSION=${NPU_TORCH_VERSION:-2.7.1}
NPU_TORCH_NPU_VERSION=${NPU_TORCH_NPU_VERSION:-2.7.1.post2}
NPU_PIP_INDEX=${NPU_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}

print_npu_warning() {
    echo "======================================================================"
    echo "WARNING: NPU runtime is unavailable, tests will continue on CPU path"
    echo "======================================================================"
}

install_npu_runtime() {
    echo "Installing NPU runtime: torch==$NPU_TORCH_VERSION torch_npu==$NPU_TORCH_NPU_VERSION"
    if ! python -m pip install "torch==$NPU_TORCH_VERSION" "torch_npu==$NPU_TORCH_NPU_VERSION" -i "$NPU_PIP_INDEX"; then
        echo "WARNING: Failed to install torch/torch_npu NPU runtime packages."
        print_npu_warning
    fi
}

report_npu_runtime() {
    echo "==================== NPU runtime report ===================="
    echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    if command -v npu-smi >/dev/null 2>&1; then
        npu-smi info || echo "WARNING: npu-smi info failed."
    else
        echo "WARNING: npu-smi not found."
    fi
    python - <<'PY'
import importlib.util
import os

warning = 'WARNING: NPU runtime is unavailable, tests will continue on CPU path'
print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

try:
    import torch
    print(f"torch={torch.__version__}")
except Exception as e:
    print(f"WARNING: failed to import torch: {e!r}")
    print('=' * 70)
    print(warning)
    print('=' * 70)
    raise SystemExit(0)

if importlib.util.find_spec('torch_npu') is None:
    print('WARNING: torch_npu is not installed.')
    print('=' * 70)
    print(warning)
    print('=' * 70)
    raise SystemExit(0)

try:
    import torch_npu
    print(f"torch_npu={getattr(torch_npu, '__version__', 'unknown')}")
except Exception as e:
    print(f"WARNING: failed to import torch_npu: {e!r}")
    print('=' * 70)
    print(warning)
    print('=' * 70)
    raise SystemExit(0)

try:
    npu = getattr(torch, 'npu', None)
    available = bool(npu is not None and npu.is_available())
    count = npu.device_count() if npu is not None else 0
    print(f"torch.npu.is_available={available}")
    print(f"torch.npu.device_count={count}")
    if not available:
        print('=' * 70)
        print(warning)
        print('=' * 70)
except Exception as e:
    print(f"WARNING: failed to query torch.npu status: {e!r}")
    print('=' * 70)
    print(warning)
    print('=' * 70)
PY
    echo "============================================================"
}

if [ "$MODELSCOPE_SDK_DEBUG" == "True" ]; then
    # pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -r requirements/tests.txt -i https://mirrors.aliyun.com/pypi/simple/
    git config --global --add safe.directory /ms-swift
    git config --global user.email tmp
    git config --global user.name tmp.com

    # linter test
    # use internal project for pre-commit due to the network problem
    if [ `git remote -v | grep alibaba  | wc -l` -gt 1 ]; then
        pre-commit run -c .pre-commit-config_local.yaml --all-files
        if [ $? -ne 0 ]; then
            echo "linter test failed, please run 'pre-commit run --all-files' to check"
            echo "From the repository folder"
            echo "Run 'pip install -r requirements/tests.txt' install test dependencies."
            echo "Run 'pre-commit install' install pre-commit hooks."
            echo "Finally run linter with command: 'pre-commit run --all-files' to check."
            echo "Ensure there is no failure!!!!!!!!"
            exit -1
        fi
    fi

    pip install -r requirements/framework.txt -U -i https://mirrors.aliyun.com/pypi/simple/
    if [ "$SWIFT_CI_USE_NPU" == "True" ]; then
        install_npu_runtime
        report_npu_runtime
    fi
    pip install decord einops -U -i https://mirrors.aliyun.com/pypi/simple/
    pip uninstall autoawq -y
    pip install optimum
    pip install diffusers
    pip install "transformers<5.0" "peft<0.19"
    # pip install autoawq -U --no-deps

    # test with install
    pip install .
    pip install auto_gptq bitsandbytes deepspeed -U -i https://mirrors.aliyun.com/pypi/simple/
else
    echo "Running case in release image, run case directly!"
fi
# remove torch_extensions folder to avoid ci hang.
rm -rf ~/.cache/torch_extensions
if [ $# -eq 0 ]; then
    ci_command="python tests/run.py --subprocess"
else
    ci_command="$@"
fi
echo "Running case with command: $ci_command"
$ci_command
