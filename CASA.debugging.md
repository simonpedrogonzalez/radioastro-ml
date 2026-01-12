
nano ~/.zshrc

export PATH="$HOME/homebrew/bin:$PATH"
. "$HOME/.local/bin/env"

unalias casa 2>/dev/null
unset -f casa 2>/dev/null

casa() {
  "/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa" --workingdir "/Users/u1528314/repos/radioastro-ml" "$@"
}

unalias casa_run 2>/dev/null
unset -f casa_run 2>/dev/null

casa_run() {
  local root="/Users/u1528314/repos/radioastro-ml/runs"
  local data_root="/Users/u1528314/repos/radioastro-ml/data"
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
  ln -sfn "$data_root" "$dir/data" || return 1
  ln -s "$repo/scripts" "$dir/scripts"
  cd "$dir" || return 1

  "/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa" \
    --workingdir "$PWD" \
    --logfile "$PWD/casa-logs/casa.log" \
    "$@"
}

source ~/.zshrc

export DISPLAY=:0


unalias casa 2>/dev/null

