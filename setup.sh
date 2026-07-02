#!/usr/bin/env bash
set -euo pipefail

echo "=== Dynamo C++20 - Third-party dependency setup ==="
echo ""

# Initialize submodules (shallow clone for speed)
echo "Initializing third-party submodules..."
git submodule update --init --depth 1 --recursive

# For header-only libraries, shallow clone is fine
echo ""
echo "All dependencies initialized successfully."
echo ""
echo "To build:"
echo "  mkdir build && cd build"
echo "  cmake .."
echo "  cmake --build ."
echo ""
echo "To run tests:"
echo "  cd build && ctest --output-on-failure"
