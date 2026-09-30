SHELL := /bin/bash
ROOT := $(CURDIR)
CONFIG := $(ROOT)/config/chpc.env

.PHONY: chpc-config chpc-setup check gpu-check kernel cpu-session gpu-session jupyter-cpu jupyter-gpu

chpc-config:
	@$(ROOT)/bin/configure-chpc

chpc-setup:
	@$(ROOT)/bin/setup-chpc

check:
	@$(ROOT)/bin/run ml python -c 'import ml, torch; print("modeling imports: OK"); print("torch", torch.__version__)'
	@$(ROOT)/bin/run ml python -m unittest ml.tests.test_cnn.PreprocessingTests.test_parity_geometry

gpu-check:
	@$(ROOT)/bin/run ml python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; run this inside make gpu-session (no CPU fallback accepted)"; x=torch.arange(4,device="cuda"); print("CUDA:",torch.cuda.get_device_name(0),"tensor:",(x*x).tolist())'

kernel:
	@$(ROOT)/bin/run ml python -m ipykernel install --user --name radioastro-ml --display-name 'Python (radioastro-ml CHPC)'

cpu-session:
	@[[ -z "$${SLURM_JOB_ID:-}" ]] || { echo 'Exit the current Slurm allocation before requesting another session.' >&2; exit 2; }; \
		$(ROOT)/bin/configure-chpc --check >/dev/null; source $(CONFIG); echo 'On the compute-node shell: cd $(ROOT) && make check'; \
		exec salloc --clusters="$${CHPC_CPU_CLUSTER}" --partition="$${CHPC_CPU_PARTITION}" --qos="$${CHPC_CPU_QOS}" \
		--account="$${CHPC_CPU_ACCOUNT}" --nodes=1 --ntasks=1 \
		--cpus-per-task="$${SESSION_CPUS:-4}" --mem="$${SESSION_MEM:-32G}" \
		--time="$${SESSION_TIME:-04:00:00}" srun --pty bash -l

gpu-session:
	@[[ -z "$${SLURM_JOB_ID:-}" ]] || { echo 'Exit the current Slurm allocation before requesting another session.' >&2; exit 2; }; \
		$(ROOT)/bin/configure-chpc --check >/dev/null; source $(CONFIG); echo 'On the compute-node shell: cd $(ROOT) && make gpu-check'; \
		options=(--clusters="$${CHPC_GPU_CLUSTER}" --partition="$${CHPC_GPU_PARTITION}" --account="$${CHPC_GPU_ACCOUNT}" \
		--nodes=1 --ntasks=1 --cpus-per-task="$${SESSION_CPUS:-4}" \
		--mem="$${SESSION_MEM:-32G}" --time="$${SESSION_TIME:-04:00:00}" \
		--gres="$${CHPC_GPU_GRES:-gpu:1}"); \
		[[ -z "$${CHPC_GPU_QOS:-}" ]] || options+=(--qos="$${CHPC_GPU_QOS}"); \
		exec salloc "$${options[@]}" srun --pty bash -l

jupyter-cpu:
	@[[ -z "$${SLURM_JOB_ID:-}" ]] || { echo 'Exit the current Slurm allocation before requesting another session.' >&2; exit 2; }; \
		$(ROOT)/bin/configure-chpc --check >/dev/null; source $(CONFIG); export RADIOASTRO_REPO='$(ROOT)' RADIOASTRO_LOGIN="$${CHPC_LOGIN_HOST}" RADIOASTRO_PORT="$${JUPYTER_PORT:-8888}"; \
		exec salloc --clusters="$${CHPC_CPU_CLUSTER}" --partition="$${CHPC_CPU_PARTITION}" --qos="$${CHPC_CPU_QOS}" \
		--account="$${CHPC_CPU_ACCOUNT}" --nodes=1 --ntasks=1 \
		--cpus-per-task="$${SESSION_CPUS:-4}" --mem="$${SESSION_MEM:-32G}" \
		--time="$${SESSION_TIME:-04:00:00}" \
		srun --pty bash -lc 'node=$$(hostname -s); echo "Mac tunnel: ssh -N -L $${RADIOASTRO_PORT}:$${node}:$${RADIOASTRO_PORT} $${USER}@$${RADIOASTRO_LOGIN}"; exec "$${RADIOASTRO_REPO}/bin/run" ml python -m jupyter lab --no-browser --ip=0.0.0.0 --port="$${RADIOASTRO_PORT}"'

jupyter-gpu:
	@[[ -z "$${SLURM_JOB_ID:-}" ]] || { echo 'Exit the current Slurm allocation before requesting another session.' >&2; exit 2; }; \
		$(ROOT)/bin/configure-chpc --check >/dev/null; source $(CONFIG); export RADIOASTRO_REPO='$(ROOT)' RADIOASTRO_LOGIN="$${CHPC_LOGIN_HOST}" RADIOASTRO_PORT="$${JUPYTER_PORT:-8888}"; \
		options=(--clusters="$${CHPC_GPU_CLUSTER}" --partition="$${CHPC_GPU_PARTITION}" --account="$${CHPC_GPU_ACCOUNT}" \
		--nodes=1 --ntasks=1 --cpus-per-task="$${SESSION_CPUS:-4}" \
		--mem="$${SESSION_MEM:-32G}" --time="$${SESSION_TIME:-04:00:00}" \
		--gres="$${CHPC_GPU_GRES:-gpu:1}"); \
		[[ -z "$${CHPC_GPU_QOS:-}" ]] || options+=(--qos="$${CHPC_GPU_QOS}"); \
		exec salloc "$${options[@]}" srun --pty bash -lc 'node=$$(hostname -s); echo "Mac tunnel: ssh -N -L $${RADIOASTRO_PORT}:$${node}:$${RADIOASTRO_PORT} $${USER}@$${RADIOASTRO_LOGIN}"; exec "$${RADIOASTRO_REPO}/bin/run" ml python -m jupyter lab --no-browser --ip=0.0.0.0 --port="$${RADIOASTRO_PORT}"'
