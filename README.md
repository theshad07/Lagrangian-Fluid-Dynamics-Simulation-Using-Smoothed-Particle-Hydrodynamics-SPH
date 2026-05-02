# Lagrangian 2D Fluid Simulation Using SPH
Overview
This project implements a 2D fluid simulation from scratch using Smoothed Particle Hydrodynamics (SPH). It models a dynamic wave collapse within a constrained tank, treating the water as a weakly compressible fluid via the stiff Tait equation of state.

To overcome the traditional O(N^2) computational bottleneck of SPH, the simulator utilizes a Linked-Cell spatial hashing algorithm, reducing neighbor search complexity to O(N). Furthermore, the solver ensures mathematical stability using a Predictor-Corrector second-order integration scheme.

# Execution Instructions

# 1. Sequential & OpenMP (Shared Memory)
# Compile
g++ -fopenmp -O3 sph_openmp.cpp -o sph_openmp.out
# Execute
./sph_openmp.out

# 2. MPI (Distributed Memory)
# Compile
mpicxx -O3 sph_mpi.cpp -o sph_mpi.out
# Execute (Replace '4' with desired number of processes)
mpirun -np 4 ./sph_mpi.out

# 3. CUDA (GPU Acceleration)
# Compile
nvcc -O3 sph_cuda.cu -o sph_cuda.out
# Execute
./sph_cuda.out
./sph_cuda.out
