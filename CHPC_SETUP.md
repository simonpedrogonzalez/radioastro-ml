# CHPC setup and verification

1. Clone on a CHPC login node and generate the private configuration from the
   Slurm allocations visible to your user:

   ```bash
   git clone git@github.com:simonpedrogonzalez/radioastro-ml.git "$HOME/radioastroml"
   cd "$HOME/radioastroml"
   bin/configure-chpc
   bin/configure-chpc --check
   ```

   Alternatively, securely copy an existing private `config/chpc.env` into the
   clone. The public `.example` is documentation and intentionally contains
   placeholders; do not use it as the runtime configuration.

2. Build the scratch-backed environment and inspect the reported CASA modules.
   Set `CASA_PIPELINE_FLAG=--pipeline` in `config/chpc.env` only if the probe
   requests it.

   ```bash
   make chpc-setup
   source config/chpc.env
   test -x "$UV_PROJECT_ENVIRONMENT/bin/python"
   readlink data collect/extracted collect/downloads collect/experiments ml/runs runs
   ```

3. Verify CPU execution in an allocation:

   ```bash
   make cpu-session
   cd "$HOME/radioastroml"
   make check
   ```

4. Verify that the locked environment uses the allocated GPU, with no CPU
   fallback:

   ```bash
   make gpu-session
   cd "$HOME/radioastroml"
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

Slurm stdout/stderr is in `slurm/logs/`. Every launcher invocation writes a
manifest under `$RADIOASTRO_SCRATCH/runs/`. Scratch is temporary and not backed
up; keep source and configuration in the clone and copy important results out.
uv's managed Python, package cache, and project environment are kept below
`$RADIOASTRO_SCRATCH`, avoiding the home quota.
CASA's downloaded runtime/measures data is kept under
`$RADIOASTRO_SCRATCH/casa/data`, not `$HOME/.casa/data`.
