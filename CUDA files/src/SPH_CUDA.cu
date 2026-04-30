#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include "file_writer.h"

using namespace std;

// Physical Constants
__constant__ double MU_D = 0.001;          
__constant__ double G_D = -50.0;           
__constant__ double PI_D = 3.14159265359;
__constant__ double GAMMA_TAIT_D = 7.0;    
__constant__ double C0_D = 20.0;           
__constant__ double RHO0_D = 1000.0;       
__constant__ double WALL_RESTITUTION_D = 0.5; 

// -----------------------------------------------------------------------------
// Flattened Particle Structure for GPU Memory
// -----------------------------------------------------------------------------
struct SPH_particle_cuda {
    double x[2];    
    double v[2];    
    double rho;  
    double p;           
    double a[2];        
    double drho_dt;     
    int is_boundary; // Changed to int for GPU memory alignment
};

// -----------------------------------------------------------------------------
// GPU Device Functions (Cubic Spline Kernels)
// -----------------------------------------------------------------------------
__device__ double calculate_W(double r, double h) {
    double q = r / h;
    double alpha = 10.0 / (7.0 * PI_D * h * h);
    if (q >= 0 && q <= 1)
        return alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    else if (q > 1 && q <= 2)
        return alpha * 0.25 * pow((2.0 - q), 3);
    return 0.0;
}

__device__ double calculate_dW(double r, double h) {
    double q = r / h;
    double alpha = 10.0 / (7.0 * PI_D * h * h * h); 
    if (q >= 0 && q <= 1)
        return alpha * (-3.0 * q + 2.25 * q * q);
    else if (q > 1 && q <= 2)
        return alpha * -0.75 * pow((2.0 - q), 2);
    return 0.0;
}

// -----------------------------------------------------------------------------
// CUDA Kernels
// -----------------------------------------------------------------------------
__global__ void kernel_density_pressure(SPH_particle_cuda* particles, int num_particles, double h, double dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double density = 0.0;
    double m_j = dx * dx * RHO0_D;

    for (int j = 0; j < num_particles; j++) {
        double dx_pos = particles[i].x[0] - particles[j].x[0];
        double dy_pos = particles[i].x[1] - particles[j].x[1];
        double dist = sqrt(dx_pos * dx_pos + dy_pos * dy_pos);

        if (dist < 2.0 * h) {
            density += m_j * calculate_W(dist, h);
        }
    }
    
    // Smooth density & calculate pressure (Tait Equation)
    particles[i].rho = max(density, RHO0_D);
    double B = (RHO0_D * C0_D * C0_D) / GAMMA_TAIT_D;
    particles[i].p = B * (pow((particles[i].rho / RHO0_D), GAMMA_TAIT_D) - 1.0);
}

__global__ void kernel_forces_integrate(SPH_particle_cuda* particles, int num_particles, double h, double dx, double delta_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    if (particles[i].is_boundary == 1) {
        particles[i].v[0] = 0.0;
        particles[i].v[1] = 0.0;
        return;
    }

    double m_j = dx * dx * RHO0_D;
    double force_pressure_x = 0.0, force_pressure_y = 0.0;
    double force_visc_x = 0.0, force_visc_y = 0.0;

    for (int j = 0; j < num_particles; j++) {
        if (i == j) continue;

        double dx_pos = particles[i].x[0] - particles[j].x[0];
        double dy_pos = particles[i].x[1] - particles[j].x[1];
        double dist = sqrt(dx_pos * dx_pos + dy_pos * dy_pos);

        if (dist > 0.0 && dist < 2.0 * h) {
            double dW = calculate_dW(dist, h);
            double ex = dx_pos / dist;
            double ey = dy_pos / dist;

            double pressure_term = (particles[i].p / (particles[i].rho * particles[i].rho)) + 
                                   (particles[j].p / (particles[j].rho * particles[j].rho));
            
            force_pressure_x -= m_j * pressure_term * dW * ex;
            force_pressure_y -= m_j * pressure_term * dW * ey;

            double dvx = particles[i].v[0] - particles[j].v[0];
            double dvy = particles[i].v[1] - particles[j].v[1];
            double v_dot_e = (dvx * ex) + (dvy * ey);
            double visc_term = MU_D * (1.0 / (particles[i].rho * particles[i].rho) + 1.0 / (particles[j].rho * particles[j].rho));

            force_visc_x += m_j * visc_term * dW * v_dot_e * ex;
            force_visc_y += m_j * visc_term * dW * v_dot_e * ey;
        }
    }

    // Update Acceleration (Forward Euler)
    particles[i].a[0] = force_pressure_x + force_visc_x;
    particles[i].a[1] = force_pressure_y + force_visc_y + G_D;

    // Update Velocity and Position
    particles[i].v[0] += delta_t * particles[i].a[0];
    particles[i].v[1] += delta_t * particles[i].a[1];
    
    particles[i].x[0] += delta_t * particles[i].v[0];
    particles[i].x[1] += delta_t * particles[i].v[1];

    // Mathematical Boundaries Backup
    double min_x = 0.0, max_x = 30.0;
    double min_y = 0.0, max_y = 50.0;

    if (particles[i].x[0] < min_x) { particles[i].x[0] = min_x; particles[i].v[0] *= -WALL_RESTITUTION_D; }
    if (particles[i].x[0] > max_x) { particles[i].x[0] = max_x; particles[i].v[0] *= -WALL_RESTITUTION_D; }
    if (particles[i].x[1] < min_y) { particles[i].x[1] = min_y; particles[i].v[1] *= -WALL_RESTITUTION_D; }
    if (particles[i].x[1] > max_y) { particles[i].x[1] = max_y; particles[i].v[1] *= -WALL_RESTITUTION_D; }
}

// -----------------------------------------------------------------------------
// CPU Controller
// -----------------------------------------------------------------------------
int main() {
    double DX = 0.2;       
    double T_MAX = 10.0;   
    double h_factor = 1.3;
    double h = DX * h_factor;
    double delta_t = 0.01 * h / 20.0; 

    cout << "=========================================" << endl;
    cout << "   SPH Wave Generation Simulator (CUDA)  " << endl;
    cout << "=========================================" << endl;

    // Generate Geometry
    vector<SPH_particle_cuda> temp_particles;
    for (double x = -3.0 * DX; x <= 30.0 + 3.0 * DX; x += DX) {
        for (double y = -3.0 * DX; y <= 50.0; y += DX) {
            bool is_boundary = (x < 0.0 || x > 30.0 || y < 0.0);
            bool is_fluid = (x > 0.0 && x <= 10.0 && y > 0.0 && y <= 20.0);

            if (is_boundary || is_fluid) {
                SPH_particle_cuda p;
                p.x[0] = x; p.x[1] = y;
                p.v[0] = 0.0; p.v[1] = 0.0;
                p.rho = 1000.0; p.p = 0.0;
                p.a[0] = 0.0; p.a[1] = 0.0;
                p.is_boundary = is_boundary ? 1 : 0;
                temp_particles.push_back(p);
            }
        }
    }

    int num_particles = temp_particles.size();
    cout << "Total particles generated: " << num_particles << endl;

    // Allocate Unified Memory for GPU
    SPH_particle_cuda* d_particles;
    cudaMallocManaged(&d_particles, num_particles * sizeof(SPH_particle_cuda));

    // Copy setup to Unified Memory
    for(int i = 0; i < num_particles; i++) {
        d_particles[i] = temp_particles[i];
    }

    // Calculate Grid and Block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    double current_time = 0;
    int step = 0;
    clock_t start = clock();

    while (current_time < T_MAX) {
        // Launch GPU Kernels
        kernel_density_pressure<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, h, DX);
        cudaDeviceSynchronize();

        kernel_forces_integrate<<<blocksPerGrid, threadsPerBlock>>>(d_particles, num_particles, h, DX, delta_t);
        cudaDeviceSynchronize();

        if (step % 50 == 0) {
            cout << "Step: " << step << " | Sim Time: " << current_time << "s" << endl;
            
            // Re-pack into your existing file_writer format to output .vtp frames
            vector<SPH_particle> output_list;
            for(int i = 0; i < num_particles; i++) {
                SPH_particle p;
                p.x[0] = d_particles[i].x[0]; p.x[1] = d_particles[i].x[1];
                p.v[0] = d_particles[i].v[0]; p.v[1] = d_particles[i].v[1];
                p.p = d_particles[i].p;
                p.is_boundary = (d_particles[i].is_boundary == 1);
                output_list.push_back(p);
            }
            string filename = "output/sph_output_" + to_string(step / 50) + ".vtp";
            write_file(filename.c_str(), &output_list);
        }

        step++;
        current_time += delta_t;
    }

    clock_t end = clock();
    cudaFree(d_particles);
    
    cout << "-----------------------------------------" << endl;
    cout << "Simulation complete!" << endl;
    cout << "Execution time: " << (end - start) / (double)CLOCKS_PER_SEC << " seconds." << endl;

    return 0;
}