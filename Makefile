# define the standard colors
ifneq ($(findstring xterm,${TERM}),)
	# check if the terminal supports color
	override COLOR_BLACK := $(shell tput -Txterm setaf 0)
	override COLOR_RED := $(shell tput -Txterm setaf 1)
	override COLOR_GREEN := $(shell tput -Txterm setaf 2)
	override COLOR_BLUE := $(shell tput -Txterm setaf 4)
	override COLOR_CYAN := $(shell tput -Txterm setaf 6)
	override COLOR_WHITE := $(shell tput -Txterm setaf 7)
	override COLOR_RESET := $(shell tput -Txterm sgr0)
else
	override COLOR_BLACK := ""
	override COLOR_RED := ""
	override COLOR_GREEN := ""
	override COLOR_BLUE := ""
	override COLOR_CYAN := ""
	override COLOR_WHITE := ""
	override COLOR_RESET := ""
endif

IMAGE_NAME ?= dockertrainsam3
TAG ?= latest
DOCKERFILE ?= docker/Dockerfile.homedesk
CONTEXT ?= .

.PHONY: builddocker
builddocker:
	@set -e; \
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME):$(TAG) $(CONTEXT)
	echo "${COLOR_CYAN}finished building docker image ${IMAGE_NAME}:${TAG} ${COLOR_RESET}"

CONTAINER_NAME ?= trainsam3
DATA_PATH ?= /mydata/

.PHONY: rundocker
rundocker:
	@set -e; \
	# allow GUI in container to reach host.
	xhost +local:docker; \
	docker run -it \
	    --env="DISPLAY=${DISPLAY}" \
	    --env="QT_X11_NO_MITSHM=1" \
		-p 8888:8888 \
	    --gpus all \
	    --ipc=host \
	    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	    --volume="/usr/local:/usr/local" \
	    --volume="$(shell pwd)/..:/root/code:rw" \
	    --volume="${DATA_PATH}:/root/data:rw" \
	    --volume="/dev/shm:/dev/shm:rw" \
	    --name=${CONTAINER_NAME} \
	    ${IMAGE_NAME}; \
	echo "${COLOR_CYAN}finished building docker container ${CONTAINER_NAME}${COLOR_RESET}"

.PHONY: execdocker
execdocker:
	@set -e; \
	if [ -n "$$(docker ps -q -f name=^/${CONTAINER_NAME}$$)" ]; then \
	    echo "${COLOR_CYAN}container ${CONTAINER_NAME} is already running. Executing into it..${COLOR_RESET}"; \
	else \
	    if [ -n "$$(docker ps -aq -f name=^/${CONTAINER_NAME}$$)" ]; then \
	        echo "${COLOR_CYAN}starting existing container ${CONTAINER_NAME}...${COLOR_RESET}"; \
		docker start -ai ${CONTAINER_NAME}; \
	    else \
	        echo "${COLOR_RED}container ${CONTAINER_NAME} does not exist. Stop.${COLOR_RESET}"; \
	    fi; \
	    exit 0; \
	fi; \
	docker exec -it ${CONTAINER_NAME} /bin/bash

TRT_VERSION ?= 10.9.0.34-1+cuda12.8

.PHONY: installdeps[infer]
installdeps[infer]:
	@set -e; \
	if [ ! -f /usr/local/cuda/include/cuda.h ]; then \
	    echo "${COLOR_RED}cuda.h not found at /usr/local/cuda/include/cuda.h. Check CUDA installation.${COLOR_RESET}"; \
	    exit 1; \
	fi; \
	printf "Proceed with TensorRT installation? [y/N] "; \
	read confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "${COLOR_CYAN}Adding NVIDIA CUDA apt repository...${COLOR_RESET}"; \
		wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb; \
		sudo dpkg -i cuda-keyring_1.1-1_all.deb; \
		sudo apt-get update; \
		echo "${COLOR_CYAN}Installing TensorRT apt packages (version $(TRT_VERSION))...${COLOR_RESET}"; \
		echo "apt-get install -y cuda-cudart-dev-12-8 libnvinfer10=$(TRT_VERSION) libnvinfer-lean10=$(TRT_VERSION) libnvinfer-dispatch10=$(TRT_VERSION) libnvinfer-plugin10=$(TRT_VERSION) libnvinfer-vc-plugin10=$(TRT_VERSION) libnvonnxparsers10=$(TRT_VERSION) libnvinfer-bin=$(TRT_VERSION)"; \
	    sudo apt-get install -y \
	        cuda-cudart-dev-12-8 \
	        libnvinfer10=$(TRT_VERSION) \
	        libnvinfer-lean10=$(TRT_VERSION) \
	        libnvinfer-dispatch10=$(TRT_VERSION) \
	        libnvinfer-plugin10=$(TRT_VERSION) \
	        libnvinfer-vc-plugin10=$(TRT_VERSION) \
	        libnvonnxparsers10=$(TRT_VERSION) \
	        libnvinfer-bin=$(TRT_VERSION); \
	else \
	    echo "${COLOR_CYAN}Skipping TensorRT installation.${COLOR_RESET}"; \
	fi; \
	echo "${COLOR_CYAN}Installing Python deps via uv...${COLOR_RESET}"; \
	if ! command -v uv > /dev/null 2>&1; then \
	    echo "${COLOR_CYAN}uv not found, installing...${COLOR_RESET}"; \
	    curl -LsSf https://astral.sh/uv/install.sh | sh; \
	    export PATH="$$HOME/.local/bin:$$PATH"; \
	fi; \
	export CUDA_HOME=/usr/local/cuda; \
	export CPATH=/usr/local/cuda/include:$$CPATH; \
	export LIBRARY_PATH=/usr/local/cuda/lib64:$$LIBRARY_PATH; \
	uv venv --system-site-packages; \
	uv sync; \
	TRT_PY_VERSION=$$(echo "$(TRT_VERSION)" | cut -d'-' -f1); \
	uv pip install tensorrt==$$TRT_PY_VERSION; \
	echo "${COLOR_GREEN}All infer deps installed.${COLOR_RESET}"

# use ruff to format, lint python files
override PYFILES ?= $(shell find ./python -type f -name '*.py')

.PHONY: env[infer]
shell:
	uv run bash

.PHONY: formatpy
formatpy:
	if hash ruff >/dev/null 2>&1; then \
		echo "${COLOR_CYAN}Running ruff${COLOR_RESET}"; \
		ruff check --fix ${PYFILES}; \
		ruff format ${PYFILES}; \
	else \
		echo "${COLOR_RED}Install ruff to enable Python formatter${COLOR_RESET}"; \
	fi

.PHONY: lintpy
lintpy:
	@set -e; \
	if hash ruff >/dev/null 2>&1; then \
		echo "${COLOR_CYAN}Running ruff${COLOR_RESET}"; \
		ruff check ${PYFILES}; \
	else \
		echo "${COLOR_RED}Install ruff to enable Python linters${COLOR_RESET}"; \
	fi

.PHONY: mypy
mypy:
	@set -e; \
	if hash mypy >/dev/null 2>&1; then \
		echo "${COLOR_CYAN}Running mypy${COLOR_RESET}"; \
		mypy ${PYFILES}; \
	else \
		echo "${COLOR_RED}Install mypy to enable Python type check${COLOR_RESET}"; \
	fi

.PHONY: test
test:
	@set -e; \
	echo "Running a dummy trainging pipeline test..."; \
	pytest -s -v tests/test_knowledge_distillation.py; \
	echo "${COLOR_CYAN}Finished test training pipeline.${COLOR_RESET}"
