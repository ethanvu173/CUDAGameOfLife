#include <vector>
#include <random>
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include "simulation.cuh"
#include <iostream>

#define WIDTH 1024 
#define HEIGHT 1024
typedef enum {CPU, GPU} mode;
mode MODE = GPU;  // Change this from CPU to GPU to change the mode


// Use two grids to avoid reading and writing to the same grid.
// In the grid, 1 = alive and 0 = dead. Use uint8_t instead of bool for 
// optimization (we want peak performance).
std::vector<uint8_t> grid1(WIDTH * HEIGHT, 0), grid2(WIDTH * HEIGHT, 0);
uint8_t* curr = grid1.data(), *next = grid2.data();

// Updates the game of life using the CPU (ie. no CUDA shenanigans)
void cpu_update(uint8_t* curr, uint8_t* next, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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
            bool alive = curr[y * width + x];
            next[y * width + x] = (neighbours == 3) || (alive && neighbours == 2);
        }
    }
}

int main() {
    // Even number indicates processing using global memory (slow)
    // Odd number indicates processing using shared memory (faster)
    int gpu_mode = 0;
    
    // Randomly assign ~33% of cells to be alive
    std::mt19937 rng(42);
    // Adjust distribution range to change portion of alive cells
    std::uniform_int_distribution<int> dist(0, 2); 
    for (auto& cell : grid1)
        cell = (dist(rng) == 0) ? 1 : 0; 
    
    sf::Vector2u dims(WIDTH, HEIGHT);
    sf::RenderWindow window(sf::VideoMode(dims), "Game of Life");
    window.setKeyRepeatEnabled(false);
    sf::Image image; 
    sf::Texture texture;
    sf::Clock clock;
    int frames = 0;

    image.resize(dims, sf::Color::Black);
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);
    
    // CUDA variables and memory management
    uint8_t *d_curr, *d_next;
    size_t grid_bytes = WIDTH * HEIGHT * sizeof(uint8_t);
    if (MODE == GPU) {
        cudaMalloc(&d_curr, grid_bytes);
        cudaMalloc(&d_next, grid_bytes);
        cudaMemcpy(d_curr, grid1.data(), grid_bytes, cudaMemcpyHostToDevice);
    }

    // Window event handling 
    const auto on_close = [&window](const sf::Event::Closed&) {
        window.close();
    };
    const auto on_key_pressed = [&window](const sf::Event::KeyPressed& keyPressed) {
        if (keyPressed.scancode == sf::Keyboard::Scancode::Escape)
            window.close();
    };

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) 
                window.close();
            if (auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->code == sf::Keyboard::Key::Escape)
                    window.close();
                else if (keyPressed->code == sf::Keyboard::Key::Space) 
                    gpu_mode++;
                    // When switching GPU modes, clear the previous buffer to
                    // prevent old cells appearing when generating new cells.
                    cudaDeviceSynchronize();
                    cudaMemset(d_next, 0, WIDTH * HEIGHT * sizeof(uint8_t));
                }
            }

            // Update the game depending on the mode.
            if (MODE == CPU) {
                cpu_update(curr, next, WIDTH, HEIGHT);
                std::swap(curr, next);
            }
            else {
                call_update(d_curr, d_next, WIDTH, HEIGHT, gpu_mode);
                cudaDeviceSynchronize();
                std::swap(d_curr, d_next);
                cudaMemcpy(grid1.data(), d_curr, grid_bytes, cudaMemcpyDeviceToHost);
            } 

            // Colour all cells based on based on whether they are alive.
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    sf::Vector2u coords(x, y);
                    auto c = curr[y * WIDTH + x] ? sf::Color::White : sf::Color::Black;
                    image.setPixel(coords, c);
                }
            }

            // FPS Counter in the window title.
            frames++;
            if (clock.getElapsedTime().asSeconds() >= 1.0) {
                if (MODE == CPU) {window.setTitle("Game of Life on CPU. FPS: " + std::to_string(frames));}
                else {
                    if (gpu_mode % 2 == 0)
                        window.setTitle("Game of Life on GPU (global). FPS: " + std::to_string(frames));
                    else
                        window.setTitle("Game of Life on GPU (shared). FPS: " + std::to_string(frames));
                }
                frames = 0;
                clock.restart();
            }

            texture.update(image);
            window.clear();
            window.draw(sprite);
            window.display();
    }
    return 0;
}