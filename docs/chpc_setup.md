# CHPC operator guide

## Bootstrap

Run on a CHPC login node:

```bash
git clone git@github.com:simonpedrogonzalez/radioastro-ml.git "$HOME/radioastro-ml"
cd "$HOME/radioastro-ml"
bin/configure-chpc
bin/configure-chpc --check
make chpc-setup
```

`bin/configure-chpc` queries the accounts, partitions, and QoS values visible
to the current Slurm user. It fills unambiguous values automatically and asks
when more than one valid choice exists. The resulting `config/chpc.env` has
literal values and is gitignored. You can instead securely copy an existing
private `config/chpc.env` into a fresh clone; never commit it. The public
`config/chpc.env.example` intentionally retains placeholders.
CPU and GPU cluster names are stored separately and passed with Slurm's
`--clusters` option, so a Notchpeak allocation can be requested from Granite.

The setup command creates the locked Python 3.12 environment and caches below
`$RADIOASTRO_SCRATCH`, then links the repository's large data/output paths to
scratch. It refuses to replace any existing real path. It also prints the CASA
module versions and probes whether `casa-vla` needs `--pipeline`; copy the
reported `CASA_PIPELINE_FLAG` value into `config/chpc.env`.
The standard CASA module gets its own version-specific scratch user base with
Astropy, Matplotlib, `fbm`, and `stochastic`; it does not reuse the ML virtual
environment or install packages under `$HOME`.

Compare the reported versions with the exercised references: local standard
CASA `6.7.6.14`, and the integration fixture's CASA `6.6.6.18` with Pipeline
`2025.1.0.36`. Exact equality is unnecessary; the smoke tests below decide
compatibility.

The resulting layout is:

```text
$RADIOASTRO_SCRATCH/
├── .cache/
├── .uv-cache/
├── casa/data/
├── data/
├── envs/
│   ├── ml/
│   └── python/
├── outputs/
└── runs/
```

The launchers set CASA's `measurespath` to `casa/data/` through
`config/casasiteconfig.py`. On first startup CASA downloads its runtime and
Measures data there, avoiding the home quota. Setup also creates the guarded
compatibility symlink `~/.casa/data` for older CASA modules. Later CASA
invocations reuse the same scratch data.

Confirm that no large environment was created in the clone:

```bash
source config/chpc.env
test -x "$UV_PROJECT_ENVIRONMENT/bin/python"
base_python=$("$UV_PROJECT_ENVIRONMENT/bin/python" -c \
  'import sys; print(sys._base_executable)')
[[ "$base_python" == "${UV_PYTHON_INSTALL_DIR:-$RADIOASTRO_SCRATCH/envs/python}"/* ]]
test ! -d ml/.venv
readlink data collect/extracted collect/downloads collect/experiments ml/runs runs
```

## Modeling checks and sessions

Request an interactive allocation, then run the command printed by the target:

```bash
make cpu-session
# In the allocation:
cd "$HOME/radioastro-ml" && make check
exit
```

GPU verification must report CUDA as available, the GPU name, and the tensor
result. It intentionally fails instead of silently using CPU:

```bash
make gpu-session
# In the allocation:
cd "$HOME/radioastro-ml" && make gpu-check
```

If this fails because the locked PyTorch CUDA runtime is incompatible with the
allocated node's driver, select a supported upstream PyTorch wheel source and
regenerate `ml/uv.lock`; do not continue with a CPU fallback.

Install the scratch-environment kernel once:

```bash
make kernel
```

Start JupyterLab with either allocation profile:

```bash
make jupyter-cpu
make jupyter-gpu
```

Run the SSH tunnel printed on screen from the Mac and open
`http://localhost:8888`. In a GPU notebook, verify:

```python
import ml
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
```

## Runtime wrappers

The modeling wrapper uses the already synchronized environment and never
imports CASA:

```bash
bin/run ml python -m unittest ml.tests.test_cnn
bin/run ml python -m ml.run_nn_experiments --help
```

CASA uses CHPC modules, not the `uv` environment:

```bash
bin/run casa scripts/create_dataset_v1.py --help
bin/run casa-pipeline scripts/vla_imaging_pipeline_test.py
```

`make chpc-setup` already checks `casatasks`, `casatools`, and `pipeline`
imports and prints their versions. If `fbm` or `stochastic` is absent, setup
installs the pinned packages into a CASA-module-specific user directory below
scratch, leaving the shared module unchanged.

After copying the reference MS to `collect/extracted/0012-399/0012-399/`, run
the real smoke tests:

```bash
RUN_CASA_SIMULATION_INTEGRATION=1 \
  bin/run casa tests/test_simulation_0012_399_integration.py

RUN_CASA_IMAGING_INTEGRATION=1 RUN_VLA_PIPELINE_INTEGRATION=1 \
  bin/run casa-pipeline tests/test_imaging_0012_399_integration.py
```

## Batch submission

Arguments are passed directly as an array to one generic Slurm entry point:

```bash
bin/submit ml-cpu python -m unittest ml.tests.test_cnn
bin/submit ml-gpu python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
bin/submit casa scripts/create_dataset_v1.py --help
bin/submit casa-pipeline scripts/vla_imaging_pipeline_test.py
```

Resource selections and defaults are in the private `config/chpc.env`. The
configuration helper shows the allocations visible to the current user;
select a non-preemptible allocation appropriate for the work.

Slurm stdout/stderr remains in the small clone-local `slurm/logs/` directory.
Every direct or batch invocation creates a unique scratch run directory with a
manifest containing the command, Git commit, host, loaded modules, runtime
version, Slurm job ID, timestamps, and exit status.

Scratch is temporary and not backed up. Copy final reports and irreplaceable
products to persistent storage.
