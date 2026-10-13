#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/setup_vist3a.sh [target_dir]

Clone VIST3A at the pinned commit and apply the UniVEdit patch. VIST3A (and the
AnySplat copy it vendors) is only needed for the 3D decoder: 3d23d Gaussian
export at inference (--enable_3d_decoder) and geometry post-training.

Arguments:
  target_dir   Where to clone VIST3A (default: third_party/VIST3A)

Afterwards:
  export VIST3A_ROOT=<target_dir>
  pip install --no-deps -r requirements-3d.txt
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${1:-${project_root}/third_party/VIST3A}"
commit="32253c3"
patch_file="${project_root}/third_party/vist3a_32253c3.patch"

if [[ ! -d "${target_dir}/.git" ]]; then
  git clone https://github.com/gohyojun15/VIST3A.git "${target_dir}"
fi
git -C "${target_dir}" checkout --quiet "${commit}"

if git -C "${target_dir}" apply --reverse --check "${patch_file}" 2>/dev/null; then
  echo "Patch already applied."
else
  git -C "${target_dir}" apply "${patch_file}"
  echo "Applied $(basename "${patch_file}")."
fi

echo "Done. export VIST3A_ROOT=${target_dir}"
