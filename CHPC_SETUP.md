# CHPC setup and verification

1. Clone on a CHPC login node. Choose a writable storage root ending in
   `radioastroml`; it may be scratch or persistent shared storage. Generate the
   private configuration from the Slurm allocations visible to your user:

   ```bash
   git clone git@github.com:simonpedrogonzalez/radioastro-ml.git "$HOME/radioastro-ml"
   cd "$HOME/radioastro-ml"
   export RADIOASTRO_SCRATCH=/path/to/your/storage/radioastroml
   bin/configure-chpc
   bin/configure-chpc --check
   ```

   Configuration records the Slurm cluster as well as the partition, account,
   and QoS so jobs can be launched from a different cluster's login node.

   Alternatively, securely copy an existing private `config/chpc.env` into the
   clone. The public `.example` is documentation and intentionally contains
   placeholders; do not use it as the runtime configuration.

2. Build the storage-backed environment and inspect the reported CASA modules.
   Set `CASA_PIPELINE_FLAG=--pipeline` in `config/chpc.env` only if the probe
   requests it. Setup installs the simulation and plotting dependencies into a
   version-specific CASA user environment under the configured storage root.

   ```bash
   make chpc-setup
   source config/chpc.env
   test -x "$UV_PROJECT_ENVIRONMENT/bin/python"
   readlink data collect/extracted collect/downloads collect/experiments ml/runs runs
   ```

3. Verify CPU execution in an allocation:

   ```bash
   make cpu-session
   cd "$HOME/radioastro-ml"
   make check
   exit
   ```

4. Verify that the locked environment uses the allocated GPU, with no CPU
   fallback:

   ```bash
   make gpu-session
   cd "$HOME/radioastro-ml"
   make gpu-check
   ```

5. Install the kernel once and start Jupyter on an allocated node. Run the
   printed SSH tunnel command on the Mac, then open `http://localhost:8888`.

   ```bash
   make kernel
   make jupyter-cpu    # or: make jupyter-gpu
   ```

6. Verify CASA after placing the `0012-399` reference data under
   `collect/extracted/`:

   ```bash
   RUN_CASA_SIMULATION_INTEGRATION=1 \
     bin/run casa tests/test_simulation_0012_399_integration.py

   RUN_CASA_IMAGING_INTEGRATION=1 RUN_VLA_PIPELINE_INTEGRATION=1 \
     bin/run casa-pipeline tests/test_imaging_0012_399_integration.py
   ```

7. Submit batch work:

   ```bash
   bin/submit ml-cpu python -m unittest ml.tests.test_cnn
   bin/submit ml-gpu python -c \
     'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
   bin/submit casa scripts/create_dataset_v1.py --help
   ```

## Changing the storage root

For the specific move from general scratch to the astronomy work directory,
including rerunning rsync and verifying the result, follow
[CHPC_STORAGE_MIGRATION.md](CHPC_STORAGE_MIGRATION.md).

1. Confirm the new location is mounted and writable on login, CPU, and GPU
   nodes. Use a dedicated child directory ending in `radioastroml`.

2. Copy valuable data and outputs. Recreate `envs/`, `.uv-cache/`, `.cache/`,
   and `bin/`; virtual environments may contain absolute paths and should not
   be copied.

   ```bash
   OLD_ROOT=/old/storage/radioastroml
   NEW_ROOT=/new/storage/radioastroml
   mkdir -p "$NEW_ROOT"
   test -r "$NEW_ROOT" && test -w "$NEW_ROOT" && test -x "$NEW_ROOT"
   rsync -aH --info=progress2 "$OLD_ROOT/data/" "$NEW_ROOT/data/"
   rsync -aH --info=progress2 "$OLD_ROOT/runs/" "$NEW_ROOT/runs/"
   rsync -aH --info=progress2 "$OLD_ROOT/outputs/" "$NEW_ROOT/outputs/"
   rsync -aH --info=progress2 "$OLD_ROOT/casa/" "$NEW_ROOT/casa/"
   ```

3. Change `RADIOASTRO_SCRATCH` and every derived path in the ignored private
   `config/chpc.env`. Alternatively, regenerate it while preserving or
   reselecting the prompted Slurm allocations:

   ```bash
   export RADIOASTRO_SCRATCH="$NEW_ROOT"
   bin/configure-chpc --force
   ```

4. Inspect the existing links. After confirming that each is a symlink to the
   old root, unlink it; never remove a real directory here.

   ```bash
   readlink data collect/extracted collect/downloads collect/experiments ml/runs runs
   readlink "$HOME/.casa/data"
   unlink data
   unlink collect/extracted
   unlink collect/downloads
   unlink collect/experiments
   unlink ml/runs
   unlink runs
   unlink "$HOME/.casa/data"
   ```

5. Recreate the links and environments, then verify them:

   ```bash
   bin/configure-chpc --check
   make chpc-setup
   source config/chpc.env
   test -x "$UV_PROJECT_ENVIRONMENT/bin/python"
   readlink data collect/extracted collect/downloads collect/experiments ml/runs runs
   ```

Slurm stdout/stderr remains in `slurm/logs/` in the code clone. Every launcher
invocation writes a manifest under `$RADIOASTRO_SCRATCH/runs/`. uv's managed
Python, package cache, project environment, and CASA runtime data are kept below
the configured root. Retention and backup guarantees depend on the selected
CHPC filesystem; verify them with the storage owner or CHPC.
