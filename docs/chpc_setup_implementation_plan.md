# CHPC Setup Implementation Plan

## Goal

Add a small, repository-owned interface for running `radioastro-ml` in two places:

- the existing local macOS setup, without changing its CASA apps, data, shell helpers, or current `uv` workflow;
- a fresh clone on CHPC, using Slurm, scratch storage, CHPC software modules, and a newly created Linux environment.

The implementation must keep two separate runtimes:

- modeling, tests, and notebooks: `uv` using `ml/pyproject.toml` and `ml/uv.lock`;
- CASA scripts: the existing CASA applications on the Mac and CHPC's `casa`/`casa-vla` modules on CHPC.

Do not introduce Conda, copy the Mac `.venv` to CHPC, copy a Mac CASA installation to CHPC, or attempt to install CASA into the modeling environment.

## Files to add or change

Add only the setup interface, rather than reorganizing the repository:

- `Makefile`: short setup, session, test, and submission commands.
- `bin/configure-chpc`: discover visible Slurm allocations and atomically write
  a literal, gitignored `config/chpc.env`.
- `bin/run`: run one command in the selected runtime (`ml`, `casa`, or `casa-pipeline`).
- `bin/submit`: submit the same runtime/command to Slurm.
- `bin/setup-chpc`: create CHPC scratch directories and the CHPC `uv` environment.
- `config/chpc.env.example`: editable CHPC resource and path defaults.
- `slurm/run_profile.sbatch`: one generic batch entry point used by `bin/submit`.
- `docs/chpc_setup.md`: brief operator instructions generated from the commands below.

Make only these targeted existing-file edits:

- `ml/pyproject.toml` and `ml/uv.lock`: add Jupyter/kernel dependencies and, only if required by testing, correct the PyTorch GPU source.
- `.gitignore`: ignore `config/chpc.env`, Slurm logs, and scratch symlinks as appropriate.
- Active script entry points only: replace an absolute Mac path if it prevents the same script from accepting a CLI argument or environment variable on CHPC.

Do not perform a repository-wide path or package-layout refactor.

## CHPC defaults

Create `config/chpc.env` from the example after cloning on CHPC. Use these defaults:

```bash
# Primary non-preemptible CPU allocation
CHPC_CPU_CLUSTER=__CHPC_CPU_CLUSTER__
CHPC_CPU_PARTITION=__CHPC_CPU_PARTITION__
CHPC_CPU_QOS=__CHPC_CPU_QOS__
CHPC_CPU_ACCOUNT=__CHPC_CPU_ACCOUNT__

# Primary non-preemptible general GPU allocation
CHPC_GPU_CLUSTER=__CHPC_GPU_CLUSTER__
CHPC_GPU_PARTITION=__CHPC_GPU_PARTITION__
CHPC_GPU_ACCOUNT=__CHPC_GPU_ACCOUNT__
CHPC_GPU_QOS=
CHPC_GPU_GRES=gpu:1

RADIOASTRO_SCRATCH=/scratch/general/vast/__CHPC_USERNAME__/radioastroml
UV_PROJECT_ENVIRONMENT=/scratch/general/vast/__CHPC_USERNAME__/radioastroml/envs/ml
UV_CACHE_DIR=/scratch/general/vast/__CHPC_USERNAME__/radioastroml/.uv-cache
UV_PYTHON_INSTALL_DIR=/scratch/general/vast/__CHPC_USERNAME__/radioastroml/envs/python
UV_PYTHON_BIN_DIR=/scratch/general/vast/__CHPC_USERNAME__/radioastroml/bin

CASA_STANDARD_MODULE=casa
CASA_PIPELINE_MODULE=casa-vla
```

Keep the GPU partition configurable. Discover general or owner-resource
partitions from the current user's Slurm associations instead of publishing
specific allocation names:

```bash
# CHPC_GPU_PARTITION=__OWNER_GPU_PARTITION__
# CHPC_GPU_QOS=__OWNER_GPU_QOS__
# CHPC_GPU_ACCOUNT=__OWNER_GPU_ACCOUNT__
```

Do not include freecycle or guest queues in the default path.

## Storage behavior

Keep the clone in `$HOME`, for example `$HOME/radioastro-ml`, but do not hardcode that location. Resolve the repository root from the launcher path.

Because `$HOME` is nearly full and the locked CUDA environment may be large, create all large or reproducible content below:

```text
/scratch/general/vast/$USER/radioastroml/
├── .uv-cache/
├── casa/data/
├── data/
├── envs/ml/
├── outputs/
└── runs/
```

`bin/setup-chpc` must:

1. create those directories;
2. create repository symlinks for large-data locations actually used by the active scripts, including `data` and output/run directories;
3. refuse to replace a real file or directory when creating a symlink;
4. leave all local Mac paths and data untouched.

Scratch is temporary and not backed up. Source, configuration, and small logs stay in the clone; large Measurement Sets, CASA products, simulations, ML datasets, and training outputs go to scratch.

## Modeling environment

The CHPC bootstrap sequence should be equivalent to:

```bash
git clone <repo-url> "$HOME/radioastro-ml"
cd "$HOME/radioastro-ml"
bin/configure-chpc
bin/setup-chpc
```

Inside `bin/setup-chpc`:

```bash
module load uv
module load quarto
uv python install 3.12
uv sync --project ml --locked --group notebook
```

Add `ipykernel` and JupyterLab as a development/notebook dependency group in `ml/pyproject.toml`; keep them out of the core runtime dependencies. Batch jobs must use the already synchronized environment:

```bash
uv run --project ml --no-sync <command...>
```

The current lock contains Linux CUDA PyTorch dependencies. Validate it on an allocated GPU node with a tiny `torch.cuda` check. If it is incompatible with the driver exposed there, configure a supported PyTorch wheel source in the `uv` project and regenerate `ml/uv.lock`; do not silently fall back to CPU.

## CASA runtime

CASA is not part of the `uv` environment. On CHPC, discover and validate the provided modules during setup:

```bash
module spider casa
module spider casa-vla
module show casa
module show casa-vla
```

Then load the appropriate module per invocation:

```bash
module load casa       # standard CASA scripts
module load casa-vla   # VLA pipeline scripts
```

`bin/run casa ...` and `bin/run casa-pipeline ...` must select the runtime by platform:

- macOS: use the existing local CASA applications or `CASA_STANDARD_BIN` / `CASA_PIPELINE_BIN` overrides;
- CHPC: load `CASA_STANDARD_MODULE` or `CASA_PIPELINE_MODULE`, then invoke the executable exposed by that module.

Confirm whether the `casa-vla` module's executable still requires `--pipeline`; derive this from `module show` and a smoke test rather than assuming it. Record the exact CHPC CASA and pipeline versions in the setup check output and compare them with the versions currently exercised by the repository:

- local standard CASA: 6.7.6.14;
- integration fixture: CASA 6.6.6.18 and Pipeline 2025.1.0.36.

Version equality is not required, but the repository's standard-CASA and VLA-pipeline smoke tests must pass. Consider a private CASA install only if the CHPC modules fail those tests and no compatible module version exists.

If a CASA module lacks the simulation-only Python packages, install them into a version-specific user package directory without changing the shared module:

```bash
python -m pip install --user 'fbm==0.3.0'
python -m pip install --user 'stochastic==0.6.0' --no-deps
```

Run that step only when an import check shows it is needed.

## Command interface

Implement these repository-root commands:

```bash
make chpc-config
make chpc-setup
make check
make gpu-check
make kernel
make cpu-session
make gpu-session
make jupyter-cpu
make jupyter-gpu

bin/run ml <command...>
bin/run casa <script.py> [args...]
bin/run casa-pipeline <script.py> [args...]

bin/submit ml-cpu <command...>
bin/submit ml-gpu <command...>
bin/submit casa <script.py> [args...]
bin/submit casa-pipeline <script.py> [args...]
```

The local Mac must continue to support its existing direct commands; these wrappers are additive.

Use these initial job defaults, all overridable from `config/chpc.env` or command options:

| Profile | Slurm allocation | Initial resources |
|---|---|---|
| `ml-cpu` | configured CPU allocation | 8 CPUs, 64 GB, 12 hours |
| `casa` | configured CPU allocation | 8 CPUs, 64 GB, 24 hours |
| `casa-pipeline` | configured CPU allocation | 8 CPUs, 64 GB, 24 hours |
| `ml-gpu` | configured general GPU partition/account | 1 GPU, 4 CPUs, 48 GB, 12 hours |
| CPU session | configured CPU allocation | 4 CPUs, 32 GB, 4 hours |
| GPU session | configured general GPU partition/account | 1 GPU, 4 CPUs, 32 GB, 4 hours |

The session targets should call `salloc` and print the next command to run. Jupyter targets should start JupyterLab on the allocated compute node with no browser and print an SSH-tunnel command for the Mac.

## Launcher requirements

The launchers must:

- pass commands as argument arrays; do not use `eval`;
- resolve the repository root dynamically;
- load modules only on CHPC;
- create a unique run directory under scratch and a small Slurm log directory;
- set `MPLCONFIGDIR` and other writable caches inside the run or scratch directory;
- write the command, Git commit, host, loaded modules, Python/CASA version, and Slurm job ID to a manifest;
- preserve the command's exit status;
- avoid importing CASA modules from the `uv` interpreter.

## Verification

Before editing, record a small Mac baseline. After implementation, repeat it to prove the local workflow is unchanged.

On a fresh CHPC clone, verify:

1. `make chpc-setup` creates scratch directories and a fresh Linux `uv` environment without writing a large environment into `$HOME`.
2. `make check` imports the core modeling package and runs a small CPU test.
3. `make gpu-check` runs inside a GPU allocation and reports `torch.cuda.is_available() == True`, device name, and a small tensor operation.
4. CPU and GPU interactive sessions start with the requested allocations.
5. CPU and GPU Jupyter kernels import the repository package; the GPU kernel sees the GPU.
6. Standard CASA imports `casatasks` and runs the smallest relevant simulation/imaging smoke test.
7. `casa-vla` imports the pipeline and runs the smallest relevant pipeline smoke test.
8. Each batch profile writes a manifest, log, and expected output below scratch.
9. Existing Mac `uv`, CASA, and pipeline commands still pass the baseline checks and use the same local data.

Implementation is complete only when all applicable checks pass and `docs/chpc_setup.md` contains the exact clone/setup/session/submission commands a user can copy and run.
