
# Single "run environment" setup

1. Edit:
`nano ~/.zshrc`

2. Paste:
```
export PATH="$HOME/homebrew/bin:$PATH"
. "$HOME/.local/bin/env"

unalias casa 2>/dev/null
unset -f casa 2>/dev/null

casa() {
  "/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa" --workingdir "/Users/u1528314/Documents/radioastro-ml" "$@"
}

unalias casa_run 2>/dev/null
unset -f casa_run 2>/dev/null

casa_run() {
  local root="/Users/u1528314/Documents/radioastro-ml/runs"
  local data_root="/Users/u1528314/Documents/radioastro-ml/data"
  local scripts_root="/Users/u1528314/Documents/radioastro-ml/scripts"
  local name
  local -a casa_args

  # optional: first arg = run name
  if [[ $# -gt 0 && "$1" != -* ]]; then
    name="$1"
    shift
  else
    name="$(date +%Y-%m-%d_%H%M%S)"
  fi

  local dir="$root/$name"

  mkdir -p "$dir"/{casa-logs,caltables,images} || return 1

  # Always expose canonical data as ./data inside the run
  cd "$dir" || return 1

  ln -sfh "$data_root" data || return 1
  ln -sfh "$scripts_root" scripts || return 1

  "/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa" \
    --workingdir "$PWD" \
    --logfile "$PWD/casa-logs/casa.log" \
    "$@"
}
```

3. Update:

```
source ~/.zshrc
```

# CASA common problems / solutions

1. Not displaying plotms plots

`export DISPLAY=:0`

2. Make it single threaded

```
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

3. Install the Python packages used by the simulation scripts

Use CASA's bundled Python, not the system Python:

```zsh
CASA_PYTHON="/Users/u1528314/Applications/CASA.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3"

"$CASA_PYTHON" -m pip install --user "fbm==0.3.0"
"$CASA_PYTHON" -m pip install --user --no-deps "stochastic==0.6.0"
```

The `--no-deps` option for `stochastic` is intentional. Without it, pip installs
older user-site copies of NumPy and SciPy that shadow CASA's bundled versions and
conflict with CASA's Astropy installation.

Verify the installation:

```zsh
"$CASA_PYTHON" -c 'import fbm, stochastic; print(fbm.__version__, stochastic.__version__)'
```

Expected output:

```text
0.3.0 0.6.0
```
