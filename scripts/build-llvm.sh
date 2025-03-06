#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script builds the LLVM library of the tool  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASE=${SCRIPT_DIR}/..

# --------------------------------------------------------------------
# Build MLIR and LLVM
# --------------------------------------------------------------------
echo ""
echo ">>> Build LLVM and MLIR "
echo ""

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

# Go to the llvm directory and carry out installation.
LLVM_DIR="${MASE}/llvm"
bash ${MASE}/mlir-air/utils/build-llvm-local.sh ${LLVM_DIR}
