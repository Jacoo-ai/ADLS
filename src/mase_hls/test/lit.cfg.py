# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "mase"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".mlir", ".ll"]

# excludes: A list of directories or files to exclude from the testsuite even
# if they match the suffixes pattern.
config.excludes = []

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mase_obj_root, "test")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("MASE_LIBS_DIR", config.mase_libs_dir)

# Propagate some variables from the host environment.
llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])


llvm_config.use_default_substitutions()

# For each occurrence of a tool name, replace it with the full path to
# the build directory holding that tool.  We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.

tool_dirs = [config.mase_tools_dir, config.llvm_tools_dir]
tools = [
    "opt",
    "mlir-opt",
    "mlir-translate",
    "mase-opt",
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
