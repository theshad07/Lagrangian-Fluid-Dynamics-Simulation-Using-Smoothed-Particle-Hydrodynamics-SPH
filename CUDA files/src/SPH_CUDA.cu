%%writefile /content/SPH_CUDA_clean.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

using namespace std;

// Physical Constants
__constant__ double MU_D = 0.001;
__constant__ double G_D = -50.0;
__constant__ double PI_D = 3.14159265359;
__constant__ double GAMMA_TAIT_D = 7.0;
__constant__ double C0_D = 20.0;
__constant__ double RHO0_D = 1000.0;
__constant__ double WALL_RESTITUTION_D = 0.5;

#define GRID_W 64
#define GRID_H 64
#define MAX_PER_CELL 128

struct SPH_particle_cuda {
    double x[2]; double v[2]; double rho; double p; double a[2]; double drho_dt; int is_boundary;
    double prev_x[2]; double prev_v[2];
};

__device__ double calculate_W(double r, double h) {
    double q = r / h;
    double alpha = 10.0 / (7.0 * PI_D * h * h);
    if (q >= 0 && q <= 1) return alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    else if (q > 1 && q <= 2) return alpha * 0.25 * pow((2.0 - q), 3);
    return 0.0;
}

__device__ double calculate_dW(double r, double h) {
    double q = r / h;
    double alpha = 10.0 / (7.0 * PI_D * h * h * h);
    if (q >= 0 && q <= 1) return alpha * (-3.0 * q + 2.25 * q * q);
    else if (q > 1 && q <= 2) return alpha * -0.75 * pow((2.0 - q), 2);
    return 0.0;
}

__global__ void clear_grid(int* grid_counters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < GRID_W * GRID_H) grid_counters[i] = 0;
}

__global__ void populate_grid(SPH_particle_cuda* particles, int num_particles, int* grid_counters, int* grid_cells, double h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    int gx = (int)((particles[i].x[0] + 5.0) / (2.0 * h));
    int gy = (int)((particles[i].x[1] + 5.0) / (2.0 * h));

    if (gx < 0) gx = 0; if (gx >= GRID_W) gx = GRID_W - 1;
    if (gy < 0) gy = 0; if (gy >= GRID_H) gy = GRID_H - 1;

    int grid_idx = gx + gy * GRID_W;
    int cell_pos = atomicAdd(&grid_counters[grid_idx], 1);
    if (cell_pos < MAX_PER_CELL) grid_cells[grid_idx * MAX_PER_CELL + cell_pos] = i;
}

__global__ void kernel_density_pressure(SPH_particle_cuda* particles, int num_particles, double h, double dx, int* grid_counters, int* grid_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double density = 0.0; double m_j = dx * dx * RHO0_D;
    int gx = (int)((particles[i].x[0] + 5.0) / (2.0 * h));
    int gy = (int)((particles[i].x[1] + 5.0) / (2.0 * h));

    for (int nx = gx - 1; nx <= gx + 1; nx++) {
        for (int ny = gy - 1; ny <= gy + 1; ny++) {
            if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
                int grid_idx = nx + ny * GRID_W;
                int count = grid_counters[grid_idx];
                if (count > MAX_PER_CELL) count = MAX_PER_CELL;
                for (int c = 0; c < count; c++) {
                    int j = grid_cells[grid_idx * MAX_PER_CELL + c];
                    double dx_pos = particles[i].x[0] - particles[j].x[0];
                    double dy_pos = particles[i].x[1] - particles[j].x[1];
                    double dist = sqrt(dx_pos * dx_pos + dy_pos * dy_pos);
                    if (dist < 2.0 * h) density += m_j * calculate_W(dist, h);
                }
            }
        }
    }
    particles[i].rho = max(density, RHO0_D);
    double B = (RHO0_D * C0_D * C0_D) / GAMMA_TAIT_D;
    particles[i].p = B * (pow((particles[i].rho / RHO0_D), GAMMA_TAIT_D) - 1.0);
}

__global__ void kernel_forces(SPH_particle_cuda* particles, int num_particles, double h, double dx, int* grid_counters, int* grid_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    if (particles[i].is_boundary == 1) { particles[i].a[0] = 0.0; particles[i].a[1] = 0.0; return; }

    double m_j = dx * dx * RHO0_D;
    double force_pressure_x = 0.0, force_pressure_y = 0.0;
    double force_visc_x = 0.0, force_visc_y = 0.0;
    int gx = (int)((particles[i].x[0] + 5.0) / (2.0 * h));
    int gy = (int)((particles[i].x[1] + 5.0) / (2.0 * h));

    for (int nx = gx - 1; nx <= gx + 1; nx++) {
        for (int ny = gy - 1; ny <= gy + 1; ny++) {
            if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
                int grid_idx = nx + ny * GRID_W;
                int count = grid_counters[grid_idx];
                if (count > MAX_PER_CELL) count = MAX_PER_CELL;
                for (int c = 0; c < count; c++) {
                    int j = grid_cells[grid_idx * MAX_PER_CELL + c];
                    if (i == j) continue;
                    double dx_pos = particles[i].x[0] - particles[j].x[0];
                    double dy_pos = particles[i].x[1] - particles[j].x[1];
                    double dist = sqrt(dx_pos * dx_pos + dy_pos * dy_pos);
                    if (dist > 0.0 && dist < 2.0 * h) {
                        double dW = calculate_dW(dist, h);
                        double ex = dx_pos / dist, ey = dy_pos / dist;
                        double pressure_term = (particles[i].p / (particles[i].rho * particles[i].rho)) + (particles[j].p / (particles[j].rho * particles[j].rho));
                        force_pressure_x -= m_j * pressure_term * dW * ex;
                        force_pressure_y -= m_j * pressure_term * dW * ey;
                        double v_dot_e = ((particles[i].v[0] - particles[j].v[0]) * ex) + ((particles[i].v[1] - particles[j].v[1]) * ey);
                        double visc_term = MU_D * (1.0 / (particles[i].rho * particles[i].rho) + 1.0 / (particles[j].rho * particles[j].rho));
                        force_visc_x += m_j * visc_term * dW * v_dot_e * ex;
                        force_visc_y += m_j * visc_term * dW * v_dot_e * ey;
                    }
                }
            }
        }
    }
    particles[i].a[0] = force_pressure_x + force_visc_x;
    particles[i].a[1] = force_pressure_y + force_visc_y + G_D;
}

__global__ void kernel_integrate_half(SPH_particle_cuda* particles, int num_particles, double delta_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles || particles[i].is_boundary == 1) return;
    particles[i].prev_x[0] = particles[i].x[0]; particles[i].prev_x[1] = particles[i].x[1];
    particles[i].prev_v[0] = particles[i].v[0]; particles[i].prev_v[1] = particles[i].v[1];
    particles[i].x[0] += 0.5 * delta_t * particles[i].v[0]; particles[i].x[1] += 0.5 * delta_t * particles[i].v[1];
    particles[i].v[0] += 0.5 * delta_t * particles[i].a[0]; particles[i].v[1] += 0.5 * delta_t * particles[i].a[1];
    if (particles[i].x[0] < 0.0) { particles[i].x[0] = 0.0; particles[i].v[0] = -WALL_RESTITUTION_D * particles[i].prev_v[0]; }
    if (particles[i].x[0] > 20.0) { particles[i].x[0] = 20.0; particles[i].v[0] = -WALL_RESTITUTION_D * particles[i].prev_v[0]; }
    if (particles[i].x[1] < 0.0) { particles[i].x[1] = 0.0; particles[i].v[1] = -WALL_RESTITUTION_D * particles[i].prev_v[1]; }
    if (particles[i].x[1] > 10.0) { particles[i].x[1] = 10.0; particles[i].v[1] = -WALL_RESTITUTION_D * particles[i].prev_v[1]; }
}

__global__ void kernel_integrate_full(SPH_particle_cuda* particles, int num_particles, double delta_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles || particles[i].is_boundary == 1) return;
    particles[i].x[0] = 2.0 * (particles[i].prev_x[0] + 0.5 * delta_t * particles[i].v[0]) - particles[i].prev_x[0];
    particles[i].x[1] = 2.0 * (particles[i].prev_x[1] + 0.5 * delta_t * particles[i].v[1]) - particles[i].prev_x[1];
    particles[i].v[0] = 2.0 * (particles[i].prev_v[0] + 0.5 * delta_t * particles[i].a[0]) - particles[i].prev_v[0];
    particles[i].v[1] = 2.0 * (particles[i].prev_v[1] + 0.5 * delta_t * particles[i].a[1]) - particles[i].prev_v[1];
    if (particles[i].x[0] < 0.0) { particles[i].x[0] = 0.0; particles[i].v[0] = -WALL_RESTITUTION_D * particles[i].prev_v[0]; }
    if (particles[i].x[0] > 20.0) { particles[i].x[0] = 20.0; particles[i].v[0] = -WALL_RESTITUTION_D * particles[i].prev_v[0]; }
    if (particles[i].x[1] < 0.0) { particles[i].x[1] = 0.0; particles[i].v[1] = -WALL_RESTITUTION_D * particles[i].prev_v[1]; }
    if (particles[i].x[1] > 10.0) { particles[i].x[1] = 10.0; particles[i].v[1] = -WALL_RESTITUTION_D * particles[i].prev_v[1]; }
}

int main() {
    double DX = 0.2; double T_MAX = 15.0;
    double h_factor = 1.3; double h = DX * h_factor;
    double delta_t = 0.01 * h / 20.0;

    cout << "=========================================" << endl;
    cout << "  SPH GPU Simulator (Pure Profiling Mode)" << endl;
    cout << "=========================================" << endl;

    vector<SPH_particle_cuda> temp_particles;
    for (double x = -0.6; x <= 20.6; x += DX) {
        for (double y = -0.6; y <= 10.6; y += DX) {
            bool is_boundary = (x < 0.0 || x > 20.0 || y < 0.0 || y > 10.0);
            bool is_fluid = (x > 0.0 && x <= 4.0 && y > 0.0 && y <= 8.0);
            if (is_boundary || is_fluid) {
                SPH_particle_cuda p = {};
                p.x[0] = x; p.x[1] = y; p.rho = 1000.0; p.is_boundary = is_boundary ? 1 : 0;
                temp_particles.push_back(p);
            }
        }
    }
    int num_particles = temp_particles.size();

    SPH_particle_cuda* d_particles;
    int *d_grid_counters, *d_grid_cells;
    cudaMallocManaged(&d_particles, num_particles * sizeof(SPH_particle_cuda));
    cudaMallocManaged(&d_grid_counters, GRID_W * GRID_H * sizeof(int));
    cudaMallocManaged(&d_grid_cells, GRID_W * GRID_H * MAX_PER_CELL * sizeof(int));
    for(int i = 0; i < num_particles; i++) d_particles[i] = temp_particles[i];

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;
    int gridBlocks = (GRID_W * GRID_H + threadsPerBlock - 1) / threadsPerBlock;

    double current_time = 0; int step = 0;
    clock_t start = clock();

    while (current_time < T_MAX) {
        clear_grid<<<gridBlocks, threadsPerBlock>>>(d_grid_counters);
        populate_grid<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, d_grid_counters, d_grid_cells, h);
        kernel_density_pressure<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, h, DX, d_grid_counters, d_grid_cells);
        kernel_forces<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, h, DX, d_grid_counters, d_grid_cells);
        cudaDeviceSynchronize();

        kernel_integrate_half<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, delta_t);
        cudaDeviceSynchronize();

        clear_grid<<<gridBlocks, threadsPerBlock>>>(d_grid_counters);
        populate_grid<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, d_grid_counters, d_grid_cells, h);
        kernel_density_pressure<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, h, DX, d_grid_counters, d_grid_cells);
        kernel_forces<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, h, DX, d_grid_counters, d_grid_cells);
        cudaDeviceSynchronize();

        kernel_integrate_full<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, delta_t);
        cudaDeviceSynchronize();

        if (step % 100 == 0) cout << "Step: " << step << " | Sim Time: " << current_time << "s" << endl;
        step++; current_time += delta_t;
    }

    clock_t end = clock();
    cudaFree(d_particles); cudaFree(d_grid_counters); cudaFree(d_grid_cells);
    cout << "Execution time: " << (end - start) / (double)CLOCKS_PER_SEC << " seconds." << endl;
    return 0;
}