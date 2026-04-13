# syntax=docker/dockerfile:1
FROM intel/oneapi:2025.3.1-0-devel-ubuntu24.04

ENV http_proxy=http://proxy-ir.intel.com:911 \
    https_proxy=http://proxy-ir.intel.com:912 \
    no_proxy=localhost,127.0.0.1 \
    HTTP_PROXY=http://proxy-ir.intel.com:911 \
    HTTPS_PROXY=http://proxy-ir.intel.com:912 \
    NO_PROXY=localhost,127.0.0.1

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu

RUN apt update && \
    apt install -y --no-install-recommends python3-pip ninja-build ccache && \
    ln -s /usr/bin/python3 /usr/bin/python

ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache

RUN pip3 install torch==2.9.0+xpu

RUN pip3 uninstall -y intel-sycl-rt

# llvm-foreach is in bin/compiler/ but setvars.sh only adds bin/ to PATH
ENV PATH="/opt/intel/oneapi/compiler/2025.3/bin/compiler:${PATH}"

RUN apt install -y intel-ocloc python3-dev

RUN pip3 install pyyaml

WORKDIR /workspace

SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/git-src \
    source /opt/intel/oneapi/setvars.sh --force && \
    source /opt/intel/oneapi/ccl/latest/env/vars.sh --force && \
    if [ ! -d /root/.cache/git-src/intel-extension-for-pytorch/.git ]; then \
        git clone https://github.com/abenmao/intel-extension-for-pytorch /root/.cache/git-src/intel-extension-for-pytorch; \
    fi && \
    cd /root/.cache/git-src/intel-extension-for-pytorch && \
    git fetch --all && \
    git checkout quant_matmul && \
    git submodule sync && \
    git submodule update --init --recursive && \
    cd /workspace && \
    cp -a /root/.cache/git-src/intel-extension-for-pytorch . && \
    cd intel-extension-for-pytorch && \
    MAX_JOBS=64 \
    TORCH_XPU_ARCH_LIST="bmg" \
    pip install -v . --no-build-isolation

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/git-src \
    source /opt/intel/oneapi/setvars.sh --force && \
    source /opt/intel/oneapi/ccl/latest/env/vars.sh --force && \
    if [ ! -d /root/.cache/git-src/torch-ccl/.git ]; then \
        git clone https://github.com/intel/torch-ccl.git /root/.cache/git-src/torch-ccl; \
    fi && \
    cd /root/.cache/git-src/torch-ccl && \
    git fetch --all && \
    git submodule sync && \
    git submodule update --init --recursive && \
    cd /workspace && \
    cp -a /root/.cache/git-src/torch-ccl . && \
    cd torch-ccl && \
    export INTELONEAPIROOT=/opt/intel/oneapi && \
    USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python setup.py install

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/git-src \
    source /opt/intel/oneapi/setvars.sh --force && \
    source /opt/intel/oneapi/ccl/latest/env/vars.sh --force && \
    if [ ! -d /root/.cache/git-src/vllm-xpu-kernels/.git ]; then \
        git clone https://github.com/vllm-project/vllm-xpu-kernels.git /root/.cache/git-src/vllm-xpu-kernels; \
    fi && \
    cp -a /root/.cache/git-src/vllm-xpu-kernels . && \
    cd vllm-xpu-kernels && \
    git checkout release/v0.1.4 && \
    sed -i '/^torch==/d' requirements.txt && \
    pip install -r requirements.txt && \
    MAX_JOBS=64 \
    VLLM_XPU_AOT_DEVICES="bmg" \
    VLLM_XPU_XE2_AOT_DEVICES="bmg" \
    pip install --no-build-isolation -v -e .  

RUN if dpkg -s python3-blinker >/dev/null 2>&1; then apt remove -y python3-blinker; fi

RUN pip3 install prettytable jsonlines psutil flask requests