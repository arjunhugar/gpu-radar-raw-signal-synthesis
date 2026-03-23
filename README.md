# GPU Radar Raw Signal Synthesis

This repository contains:
- CUDA source files for radar raw-signal synthesis
- CMake build setup for generating a DLL
- Python scripts for plotting Range–Doppler outputs

## Files
- `src/radar.cu`
- `src/cluster.cu`
- `CMakeLists.txt`

## Build
This project requires:
- CUDA Toolkit
- local vendor SDK headers (not included in this repository)

Example:
```bash
cmake -S . -B build -DVENDOR_SDK_ROOT="C:/path/to/your/sdk"
cmake --build build --config Release
