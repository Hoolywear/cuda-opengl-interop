# Basic Interop Skeleton

Minimal starting point for **OpenGL + GLFW + GLAD + CUDA** interoperability experiments on Windows (MSVC).
Currently: creates an OpenGL 4.1 core context, clears the screen, and (if CUDA toolkit installed) launches a dummy CUDA kernel.

---

## Project Structure

```
basic_interop_skel/
├── CMakeLists.txt             # Build script (auto-fetches GLFW if not vendored)
├── include/
│   ├── callbacks.h            # GLFW callback declarations
│   ├── gl_setup.h             # GLAD init helper
│   └── cuda_kernels.cuh       # CUDA kernel declarations
├── src/
│   ├── main.cpp               # Entry point (simple C-style)
│   ├── callbacks.cpp          # Callback implementations
│   └── cuda_kernels.cu        # Dummy CUDA kernel (future expansion)
├── external/
│   └── glad/                  # GLAD (headers + loader)
├── build/                     # Generated build directory (not tracked)
└── .gitignore
```

---

## Dependencies

- **CMake** >= 3.16
- **Visual Studio 2022** (MSVC, Desktop development with C++)
- **CUDA Toolkit** (for nvcc and device runtime) — required (CUDA is now mandatory)
- **GLFW** fetched automatically via `FetchContent` (unless a local `external/glfw` exists)
- **GLAD** vendored (core OpenGL loader)

---

## Build Instructions (PowerShell)

```powershell
# Configure (will fetch GLFW 3.4 if not vendored)
cmake -S . -B build

# Build (multi-config generator assumed: VS)
cmake --build build --config Release

# Run
build/Release/basic_interop_skel.exe
```

Debug build:
```powershell
cmake --build build --config Debug
```

If you prefer a single-config generator (Ninja):
```powershell
cmake -S . -B build -G Ninja
cmake --build build --config Release
```

---

## Runtime

On launch you should see:
- GLFW version string in stdout.
- Creation of an 800x600 OpenGL 4.1 core window.
- Message: `CUDA dummy kernel executed successfully.` (if CUDA runtime succeeded).
- Press `Esc` to close.

---

## CUDA Interop Roadmap (Suggested Next Steps)

1. Create shared OpenGL buffers/textures (PBOs, SSBOs, textures) for compute.
2. Register them with CUDA using `cudaGraphicsGLRegisterBuffer` / `cudaGraphicsGLRegisterImage`.
3. Map, launch kernels writing into registered memory, unmap, render.
4. Add sync primitives (glFinish, CUDA events) only where necessary.
5. Consider a small utility header for error-check macros (e.g. `CUDA_CHECK`).

The current `cuda_kernels.cu` provides a dummy kernel to validate the toolchain.

To extend:
```cpp
#ifdef HAS_CUDA
CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&res, glBufferId, cudaGraphicsRegisterFlagsWriteDiscard));
#endif
```

## Troubleshooting

- Ensure the CUDA toolkit bin directory (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`) is in PATH.
- If CMake cannot enable CUDA: verify installed toolkit and matching MSVC.
- Re-run configure if you add/remove `external/glfw`.

## License

This scaffold includes vendored GLAD (per its license) and fetches GLFW (zlib/libpng license). Review upstream licenses before redistribution.

---

## Author Note

Initial README adapted after refactors; CUDA now mandatory for future interop experimentation.
