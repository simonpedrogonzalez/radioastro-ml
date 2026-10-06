# CHPC storage migration

Move data, results, environments, caches, and CASA data from:

```text
/scratch/general/vast/u1528314/radioastroml
```

to:

```text
/uufs/astro.utah.edu/common/home/u6039460/work/sgonzalez/radioastroml
```

The code clone stays at `$HOME/radioastro-ml`. `RADIOASTRO_SCRATCH` retains its
existing name but points to the new astronomy work directory.

## Confirmed progress

- Code changes have been pulled on CHPC and the updated private configuration copied.
- `bin/configure-chpc --check` passed on `granite1`.
- The new root is readable, writable, and searchable on that login node.
- The seven repository/CASA links were inspected and still pointed to old scratch.
- Copy completion, environment setup, and access from CPU/GPU compute nodes remain unconfirmed.

Run the following steps on CHPC. Stop jobs that write to the old directories
before copying. Keep the original scratch contents until verification is complete.

## 1. Check the configured destination

`config/chpc.env` is gitignored: pulling code does not update it. It must point
all six storage settings (`RADIOASTRO_SCRATCH`, `UV_PROJECT_ENVIRONMENT`,
`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_PYTHON_BIN_DIR`, and `CASA_DATA_DIR`)
to the new root or its appropriate subdirectory.

```bash
cd "$HOME/radioastro-ml"
(
    set -e
    source config/chpc.env
    bin/configure-chpc --check
    test "$RADIOASTRO_SCRATCH" = /uufs/astro.utah.edu/common/home/u6039460/work/sgonzalez/radioastroml
    mkdir -p "$RADIOASTRO_SCRATCH"
    test -r "$RADIOASTRO_SCRATCH"
    test -w "$RADIOASTRO_SCRATCH"
    test -x "$RADIOASTRO_SCRATCH"
    echo "Destination and storage access OK"
)
```

Proceed only if this prints `Destination and storage access OK`.

## 2. Copy or finish copying data and results

This block stops if any copy fails. Copy the data and results; recreate the
Python/CASA user environments, package caches, and managed Python installation
through setup in step 3.

```bash
(
    set -e
    source config/chpc.env
    OLD_ROOT=/scratch/general/vast/u1528314/radioastroml
    test "$RADIOASTRO_SCRATCH" = /uufs/astro.utah.edu/common/home/u6039460/work/sgonzalez/radioastroml

    for subdir in data runs outputs casa/data; do
        mkdir -p "$RADIOASTRO_SCRATCH/$subdir"
        rsync -aH --info=progress2 \
            "$OLD_ROOT/$subdir/" \
            "$RADIOASTRO_SCRATCH/$subdir/"
    done
    echo "All copies completed successfully"
)
```

If rsync already ran, rerun this block to finish anything missed. `rsync -aH`
compares file sizes and modification times, skips matching files, copies missing
files, and updates differing files. An incomplete file is transferred again as
needed. The commands scan the directories but do not recopy every matching file.
They do not delete the original scratch contents or extra destination files.

For stronger verification during copying, change `-aH` to `-aHc`. The `c` flag
compares file contents using checksums and reads both copies completely, so it
can take substantially longer. It only transfers missing or differing files.

To check file contents without changing the destination, use this optional
checksum dry run:

```bash
(
    set -e
    source config/chpc.env
    OLD_ROOT=/scratch/general/vast/u1528314/radioastroml

    for subdir in data runs outputs casa/data; do
        rsync -aHnc --itemize-changes \
            "$OLD_ROOT/$subdir/" \
            "$RADIOASTRO_SCRATCH/$subdir/"
    done
)
```

No itemized changes and no errors mean rsync found nothing to update. Itemized
entries can also indicate metadata differences; they do not all imply different
file contents. If updates are needed, rerun the copying block with `-aHc`.

Proceed only after copying finishes successfully. If an old source directory is
missing or inaccessible, investigate that error before continuing.

## 3. Recreate links and environments

The block below removes links pointing to old scratch, leaves links already
pointing under the new root in place, and stops on an unexpected link or a real
file/directory. Setup checks each remaining link against its exact expected
target. This allows the step to be rerun after an interrupted setup.

```bash
(
    set -e
    source config/chpc.env
    bin/configure-chpc --check
    OLD_ROOT=/scratch/general/vast/u1528314/radioastroml
    test "$RADIOASTRO_SCRATCH" = /uufs/astro.utah.edu/common/home/u6039460/work/sgonzalez/radioastroml

    for link in data collect/extracted collect/downloads \
        collect/experiments ml/runs runs "$HOME/.casa/data"; do
        if [[ -L "$link" ]]; then
            target=$(readlink "$link")
            case "$target" in
                "$OLD_ROOT"/*) unlink "$link" ;;
                "$RADIOASTRO_SCRATCH"/*) ;;
                *) echo "Unexpected link: $link -> $target" >&2; exit 1 ;;
            esac
        elif [[ -e "$link" ]]; then
            echo "Refusing to replace real path: $link" >&2
            exit 1
        fi
    done

    make chpc-setup
    test -x "$UV_PROJECT_ENVIRONMENT/bin/python"
    ls -ld data collect/extracted collect/downloads \
        collect/experiments ml/runs runs "$HOME/.casa/data"
)
```

All seven links should now point under the astronomy work directory. Setup logs
are saved in `slurm/logs/chpc-setup-*.log`. If setup fails, retain the final error
output and log path before retrying.

## 4. Check CPU and GPU compute nodes

Request a CPU allocation from the login node:

```bash
make cpu-session
```

In the allocated compute-node shell:

```bash
cd "$HOME/radioastro-ml"
source config/chpc.env
test -r "$RADIOASTRO_SCRATCH" && test -w "$RADIOASTRO_SCRATCH" && test -x "$RADIOASTRO_SCRATCH" && echo "CPU storage access OK"
make check
exit
```

Back on the login node, request a GPU allocation:

```bash
make gpu-session
```

In the allocated GPU-node shell:

```bash
cd "$HOME/radioastro-ml"
source config/chpc.env
test -r "$RADIOASTRO_SCRATCH" && test -w "$RADIOASTRO_SCRATCH" && test -x "$RADIOASTRO_SCRATCH" && echo "GPU storage access OK"
make gpu-check
exit
```

Both storage checks and the corresponding execution checks must pass. For CASA
integration tests, Jupyter kernel installation, and batch submission examples,
continue with [CHPC_SETUP.md](CHPC_SETUP.md).

## If `make check` reports missing `torch`

The Python environment used by the launcher lacks PyTorch. `make check` uses
`uv run --no-sync`, so it does not install missing dependencies. Synchronize the
configured environment with the committed lockfile.

Exit the compute-node shell with `exit`, then run this on the login node:

```bash
cd "$HOME/radioastro-ml"
(
    set -e
    source config/chpc.env
    bin/configure-chpc --check
    module load "${UV_MODULE:-uv}"
    export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR
    export UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR
    export UV_MANAGED_PYTHON=true
    uv sync --project "$PWD/ml" --locked --group notebook
    "$UV_PROJECT_ENVIRONMENT/bin/python" -c \
        'import sys, torch; print(sys.executable); print("torch:", torch.__version__)'
)
```

If synchronization fails, retain the error output before proceeding. Once the
import succeeds, request a fresh CPU allocation and repeat step 4. This repairs
the ML dependency installation; completion of the full CASA setup still needs
to be verified separately if `make chpc-setup` did not finish successfully.
