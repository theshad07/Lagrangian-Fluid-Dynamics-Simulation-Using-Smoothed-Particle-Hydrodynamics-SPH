#include "SPH_2D.h"
#include "file_writer.h"
#include <mpi.h>
#include <string>

SPH_main domain;

int main(int argc, char* argv[])
{
    // Initialize MPI Environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double h_factor = 1.3; 
    double DX = 0.2;       
    double T_MAX = 15.0;   

    if (rank == 0) {
        cout << "=========================================" << endl;
        cout << "   SPH Simulator (MPI Implementation)    " << endl;
        cout << "   Running on " << size << " processes." << endl;
        cout << "=========================================" << endl;
    }

    domain.set_values(h_factor, DX, T_MAX);
    domain.initialise_grid();
    domain.place_points(domain.min_x, domain.max_x);

    // Start MPI Timer!
    double start_time = MPI_Wtime();

    bool smooth;
    double current_time = 0;
    int step = 0;

    while (current_time < domain.t_max)
    {
        smooth = (step % 20 == 0); 
        
        // Pass rank and size into the physics calculation!
        domain.predictor_corrector(smooth, rank, size);

        // Only let Rank 0 print and write files to avoid hard drive corruption
        if (rank == 0 && step % 50 == 0)
        {
            cout << "Step: " << step << " | Sim Time: " << current_time << "s" << endl;

            std::string filename = "output/sph_frame_" + std::to_string(step) + ".vtp";
            write_file(filename.c_str(), &domain.particle_list);
        }

        step++;
        current_time += domain.delta_t;
    }

    // Stop MPI Timer!
    double end_time = MPI_Wtime();

    if (rank == 0) {
        cout << "-----------------------------------------" << endl;
        cout << "Simulation complete!" << endl;
        cout << "Execution time: " << (end_time - start_time) << " seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}