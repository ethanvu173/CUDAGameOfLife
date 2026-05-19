#include "simulation.cuh"
#include <iostream>

#define SECTION_W 32
#define SECTION_H 32

__global__ void update_kernel(const uint8_t* curr, uint8_t* next, 
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int neighbours = 0;
    for (int check_y = -1; check_y <= 1; check_y++) {
        for (int check_x = -1; check_x <= 1; check_x++) {
            if (check_x == 0 && check_y == 0) continue;

            int new_x = (x+check_x+width) % width;
            int new_y = (y+check_y+height) % height;
            neighbours += curr[new_y * width + new_x];
        }
    }
    // bool alive = curr[y * width + x];
    // if (alive) {
    //     if (neighbours < 2 || neighbours > 3) next[y * width + x] = 0;
    //     else next[y * width + x] = 1;
    // }
    // else if (neighbours == 3) next[y * width + x] = 0;
    bool is_alive = curr[y * width + x];
    next[y * width + x] = (neighbours == 3) || (is_alive && neighbours == 2);
}

__global__ void update_shared(uint8_t* curr, uint8_t* next, int width, int height) {
    // A shared section that holds the section of cells the block must compute
    __shared__ uint8_t section[SECTION_H+2][SECTION_W+2];
    
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int x = blockIdx.x * SECTION_W + thread_x, y = blockIdx.y * SECTION_H + thread_y;

    // Fetch the cell at (x, y) 
    auto fetch = [&](int x, int y) {
        x = (x + width) % width;
        y = (y + height) % height;
        return curr[y * width + x];
    };

    // Fetch the thread's own cell
    section[thread_y+1][thread_x+1] = fetch(x, y);

    // Threads bordering the edge of the section must load its cells to prevent
    // repeatedly checking from global memory
    if (thread_x == 0) 
        section[thread_y+1][0] = fetch(x-1, y);
    if (thread_x == SECTION_W-1) 
        section[thread_y+1][SECTION_W-1] = fetch(x+1, y);
    if (thread_y == 0) 
        section[0][thread_x+1] = fetch(x, y-1);
    if (thread_y == SECTION_H-1) 
        section[SECTION_H+1][thread_x+1] = fetch(x, y+1);

    // Threads in the corner must also load its cells
    if (thread_x == 0 && thread_y == 0) 
        section[0][0] = fetch(x-1, y-1);
    if (thread_x == SECTION_W-1 && thread_y == 0) 
        section[0][SECTION_W+1] = fetch(x+1, y-1);
    if (thread_x == 0 && thread_y == SECTION_H-1) 
        section[SECTION_H+1][0] = fetch(x-1, y+1);
    if (thread_x == SECTION_W-1 && thread_y == SECTION_H-1) 
        section[SECTION_H+1][SECTION_W+1] = fetch(x+1, y+1);

    __syncthreads;

    if (x >= width || y >= height) return;

    // Compute alive neighbours using shared memory now
    int neighbours = 0;
    for (int check_y = 0; check_y <= 2; check_y++) {
        for (int check_x = 0; check_x <= 2; check_x++) {
            if (check_x != 1 || check_y != 1) 
                neighbours += section[thread_y+check_y][thread_x+check_x];
        }
    }
    bool is_alive = section[thread_y+1][thread_x+1];
    next[y * width + x] = (neighbours == 3) || (is_alive &&  neighbours == 2);
}

void call_update(uint8_t* d_curr, uint8_t* d_next, int width, int height, int mode) {
    if (mode % 2 == 0) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
        update_kernel<<<grid, block>>>(d_curr, d_next, width, height);
    }
    else {
        dim3 block(32, 32); // different block sizes for global and shared processing
        dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
        update_shared<<<grid, block>>>(d_curr, d_next, width, height);
    }
}   