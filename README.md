# CUDA Game of Life
A project to familiarize myself with the CUDA toolkit. Play Conway's Game of Life using both processing on the CPU and on the GPU with CUDA.
## Requirements
- CUDA-capable GPU
- [CUDA Toolkit 13+](https://developer.nvidia.com/cuda-downloads)
- SFML 3+
- CMake 3.18+
## Build
On Windows:
1. Ensure all dependencies above are installed
2. Navigate to the directory *CUDAGameOfLife*. Create a directory, *build*, as in:  
`mkdir build && cd build`
3. When building for the first time, run the commands:  
`cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/Users/cielr/vcpkg/scripts/buildsystems/vcpkg.cmake`  
`cmake --build . --config Release`
5. For subsequent builds, only run:  
`cmake --build . --config Release`

## Usage
### Execution
In the directory *build*, run:  
`Release\game_of_life.exe`  
Press escape to close the program.
### Switching between CPU and GPU
In *main.cpp*, replace the value of the enum MODE to either CPU or GPU (at the top of the file).
### Switching between simple and shared memory kernels
This program allows you to switch between a simple CUDA implementation and a more optimized implementation using shared memory.  

With the mode set to GPU, press spacebar to switch implementations during runtime.  

On my system, I have observed ~90 FPS using the CPU, ~100 FPS using the simple implementation and ~120 FPS using shared memory, demonstrating the processing advantages of CUDA.

## What is Conway's Game of Life?
Conway's Game of Life is a "zero" player game on a 2D grid. Each cell in the grid is either dead or alive, depending on its immediate neighbours: 
* A live cell surrounded by less than two live cells dies.
* A live cell surrounded by two or three live cells continues to live.
* A live cell surrounded by more than three live cells dies.
* A dead cell surrounded by three live cells comes alive.

From these rules, the behaviours of various configurations can be explored. In this case, the Game of Life is being used as a benchmark to illustrate the processing capabilities of CUDA.
