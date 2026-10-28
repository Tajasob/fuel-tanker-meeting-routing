#!/bin/bash
# Solves the exported integer programs with the CPLEX Interactive Optimizer (macOS) and writes the results to results/.
# A model arrives in inbox/ as <name>.mps.bz2.part* with <name>.ready (first line: md5 of the joined file).
# Results: <name>.out (log), <name>.sol (integer solution), <name>.relax.sol (linear relaxation).
# Start in Terminal:   caffeinate -is bash cplex/run.sh
# The script waits for new models and ends by itself when the file STOP appears in this folder.
HERE="$(cd "$(dirname "$0")" && pwd)"
INBOX="$HERE/inbox"
OUT="$HERE/results"
WORK="/tmp/p1_cplex"
mkdir -p "$INBOX" "$OUT" "$WORK"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
CPLEX="$(ls "$HOME"/IBM/ILOG/CPLEX_Studio*/cplex/bin/*/cplex /Applications/CPLEX_Studio*/cplex/bin/*/cplex 2>/dev/null | head -n 1)"
if [ -z "$CPLEX" ] || [ ! -x "$CPLEX" ]; then
  echo "$(stamp) CPLEX not found" >> "$HERE/status.txt"
  echo "CPLEX not found"
  exit 1
fi
{
  echo "cplex: $CPLEX"
  echo "model: $(sysctl -n hw.model)"
  echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
  echo "cores: $(sysctl -n hw.ncpu)"
  echo "performance_cores: $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)"
  echo "efficiency_cores: $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null)"
  echo "memory_bytes: $(sysctl -n hw.memsize)"
  echo "macos: $(sw_vers -productVersion)"
  echo "arch: $(uname -m)"
} > "$HERE/machine.txt"
echo "$(stamp) started" >> "$HERE/status.txt"
echo "CPLEX is running. Keep this window open; the script ends by itself when all models are solved."
while true; do
  found=0
  for ready in "$INBOX"/*.ready; do
    [ -e "$ready" ] || continue
    name="$(basename "$ready" .ready)"
    [ -e "$OUT/$name.done" ] && continue
    found=1
    echo "$(stamp) solving $name" | tee -a "$HERE/status.txt"
    cat "$INBOX/$name".mps.bz2.part* > "$WORK/$name.mps.bz2"
    if [ "$(md5 -q "$WORK/$name.mps.bz2")" != "$(head -n 1 "$ready")" ]; then
      echo "$(stamp) checksum mismatch $name" | tee -a "$HERE/status.txt"
      rm -f "$WORK/$name.mps.bz2"
      mv "$ready" "$ready.bad"
      continue
    fi
    rm -f "$WORK/$name.mps" "$WORK/$name.sol" "$WORK/$name.relax.sol" "$WORK/$name.out"
    bunzip2 "$WORK/$name.mps.bz2"
    start=$(date +%s)
    "$CPLEX" -c "set parallel 1" "set timelimit 3600" "set mip tolerances mipgap 1e-09" "set output writelevel 4" "set workdir $WORK" "read $WORK/$name.mps" "optimize" "write $WORK/$name.sol" "change problem lp" "optimize" "write $WORK/$name.relax.sol" "quit" > "$WORK/$name.out" 2>&1
    echo "wall_seconds $(( $(date +%s) - start ))" >> "$WORK/$name.out"
    for f in "$name.out" "$name.sol" "$name.relax.sol"; do
      if [ -e "$WORK/$f" ]; then mv "$WORK/$f" "$OUT/$f"; fi
    done
    rm -f "$WORK/$name.mps"
    stamp > "$OUT/$name.done"
    echo "$(stamp) finished $name" | tee -a "$HERE/status.txt"
  done
  if [ -e "$HERE/STOP" ]; then break; fi
  if [ "$found" -eq 0 ]; then sleep 20; fi
done
echo "$(stamp) stopped" >> "$HERE/status.txt"
echo "All models are solved. You can close this window."
