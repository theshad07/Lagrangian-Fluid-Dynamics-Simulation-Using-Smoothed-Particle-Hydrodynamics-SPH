#include "SPH_2D.h"
#include <mpi.h>
#include <algorithm> // For std::min

SPH_main* SPH_particle::main_data;

void SPH_particle::calc_index(void)
{
    for (int i = 0; i < 2; i++)
        list_num[i] = int((x[i] - main_data->min_x[i]) / (2.0 * main_data->h));
}

SPH_main::SPH_main()
{
    SPH_particle::main_data = this;
}

void SPH_main::set_values(double h_factor, double DX, double T_MAX)
{
    inner_min_x[0] = min_x[0] = 0.0;
    inner_min_x[1] = min_x[1] = 0.0;

    inner_max_x[0] = max_x[0] = 20.0;
    inner_max_x[1] = max_x[1] = 10.0;

    dx = DX;
    h_fac = h_factor;
    h = dx * h_fac; // Characteristic smoothing length (from PDF)
    t_max = T_MAX;
    
    // Initial time step calculation (from PDF: dt = 0.1 * h / c0)
    delta_t = 0.01 * h / C0; 
    cout << "Calculated Smoothing Length (h): " << h << endl;
}

void SPH_main::initialise_grid(void)
{
    for (int i = 0; i < 2; i++)
    {
        // Enlarge the region to set 2h-wide boundary particles as per PDF
        min_x[i] -= 3.0 * h;
        max_x[i] += 3.0 * h;

        // Calculate the number of grid cells in each dimension for Linked-Cell
        max_list[i] = int((max_x[i] - min_x[i]) / (2.0 * h) + 1.0);
    }

    search_grid.resize(max_list[0]);
    for (int i = 0; i < max_list[0]; i++)
        search_grid[i].resize(max_list[1]);
}

void SPH_main::place_points(double* min, double* max)
{
    double x_pos[2] = { min[0], min[1] };
    SPH_particle particle;

    while (x_pos[0] <= max[0])
    {
        x_pos[1] = min[1];
        while (x_pos[1] <= max[1])
        {
            // --- DAM BREAK GEOMETRY ---
            // 1. Check if the current coordinate is inside the actual tank (not a wall)
            bool inside_tank = (x_pos[0] > inner_min_x[0] && x_pos[0] < inner_max_x[0] && 
                                x_pos[1] > inner_min_x[1] && x_pos[1] < inner_max_x[1]);
            
            // 2. Define our Dam: A column of water 4 meters wide and 8 meters tall on the left.
            // If a point is inside the tank, but OUTSIDE this column, we leave it empty.
            bool outside_water_column = (x_pos[0] > 4.0 || x_pos[1] > 8.0);

            if (inside_tank && outside_water_column)
            {
                x_pos[1] += dx;
                continue; // Skip placing a fluid particle here
            }
            // --------------------------

            for (int i = 0; i < 2; i++)
                particle.x[i] = x_pos[i];

            // Assign boundary status using our clean inside_tank boolean
            if (!inside_tank)
                particle.is_boundary = true;
            else
                particle.is_boundary = false;

            particle.calc_index();
            particle_list.push_back(particle);

            x_pos[1] += dx;
        }
        x_pos[0] += dx;
    }
}

void SPH_main::allocate_to_grid(void)
{
    // 1. Bulletproof Clear Loop
    // Uses the actual size of the grid in memory, so it can never overreach
    for (size_t i = 0; i < search_grid.size(); i++)
        for (size_t j = 0; j < search_grid[i].size(); j++)
            search_grid[i][j].clear();

    // 2. Bulletproof Allocation Loop
    for (unsigned int cnt = 0; cnt < particle_list.size(); cnt++)
    {
        int ix = particle_list[cnt].list_num[0];
        int iy = particle_list[cnt].list_num[1];

        // Get safe maximum bounds
        int max_x = search_grid.size();
        int max_y = (max_x > 0) ? search_grid[0].size() : 0;

        // Force a strict clamp right before grid access
        if (ix < 0) ix = 0;
        else if (ix >= max_x) ix = max_x - 1;

        if (iy < 0) iy = 0;
        else if (iy >= max_y) iy = max_y - 1;

        // Safely lock the corrected index back into the particle
        particle_list[cnt].list_num[0] = ix;
        particle_list[cnt].list_num[1] = iy;

        // Safely push to the grid
        particle_list[cnt].grid_index = search_grid[ix][iy].size();
        search_grid[ix][iy].push_back(&particle_list[cnt]);
    }
}

// --------------------------------------------------------------------------------------
// CORE PHYSICS: NAVIER-STOKES & CONTINUITY EQUATIONS
// This function calculates acceleration (dv/dt) and density change (drho/dt)
// --------------------------------------------------------------------------------------
void SPH_main::update_a_D(SPH_particle* part, SPH_particle* other_part, double dist, bool stencil)
{
    double m_j = dx * dx * RHO0; // Mass of neighboring particle j
    double dW = calculate_dW(dist); // Derivative of smoothing kernel

    // Unit vector e_ij (Direction from particle to neighbor)
    double ex = (part->x[0] - other_part->x[0]) / dist;
    double ey = (part->x[1] - other_part->x[1]) / dist;

    // Relative velocity v_ij
    double dvx = part->v[0] - other_part->v[0];
    double dvy = part->v[1] - other_part->v[1];
    
    // Dot product v_ij * e_ij
    double v_dot_e = (dvx * ex) + (dvy * ey);

    // Symmetric Pressure gradient approximation (PDF equation)
    double pressure_term = (part->p / (part->rho * part->rho)) + (other_part->p / (other_part->rho * other_part->rho));
    
    // Viscous term approximation
    double visc_term = MU * (1.0 / (part->rho * part->rho) + 1.0 / (other_part->rho * other_part->rho));

    // Calculate Acceleration (dv/dt)
    if (!part->is_boundary)
    {
        part->a[0] += m_j * (-pressure_term * dW * ex + visc_term * dW * v_dot_e * ex);
        part->a[1] += m_j * (-pressure_term * dW * ey + visc_term * dW * v_dot_e * ey);
    }
    // Newton's 3rd Law (Symmetry)
    if (stencil && !other_part->is_boundary)
    {
        other_part->a[0] -= m_j * (-pressure_term * dW * ex + visc_term * dW * v_dot_e * ex);
        other_part->a[1] -= m_j * (-pressure_term * dW * ey + visc_term * dW * v_dot_e * ey);
    }

    // Continuity Equation (dRho/dt)
    double rho_change = m_j * dW * v_dot_e;
    part->drho_dt += rho_change;
    
    if (stencil)
        other_part->drho_dt += rho_change;
}

void SPH_main::update_dynamical_t(SPH_particle* part, SPH_particle* other_part)
{
    // CFL Conditions from PDF
    double v_ij = sqrt(pow(part->v[0] - other_part->v[0], 2) + pow(part->v[1] - other_part->v[1], 2));
    if (v_ij > 0) {
        double tmp_cfl = h / v_ij;
        if (tmp_cfl < dt_cfl) dt_cfl = tmp_cfl;
    }

    double a_i = sqrt(pow(part->a[0], 2) + pow(part->a[1], 2));
    if (a_i > 0) {
        double tmp_f = sqrt(h / a_i);
        if (tmp_f < dt_f) dt_f = tmp_f;
    }

    double tmp_a = h / (C0 * sqrt(pow(part->rho / RHO0, GAMMA_TAIT - 1.0)));
    if (tmp_a < dt_a) dt_a = tmp_a;
}

// --------------------------------------------------------------------------------------
// NEIGHBOR SEARCHING (O(N) linked cell algorithm)
// --------------------------------------------------------------------------------------
void SPH_main::neighbour_iterate_non_stencil(SPH_particle* part, bool stencil, bool change_delta_t)
{
    SPH_particle* other_part;
    double dist, dn[2];

    for (int i = part->list_num[0] - 1; i <= part->list_num[0] + 1; i++)
        if (i >= 0 && i < max_list[0])
            for (int j = part->list_num[1] - 1; j <= part->list_num[1] + 1; j++)
                if (j >= 0 && j < max_list[1])
                {
                    for (unsigned int cnt = 0; cnt < search_grid[i][j].size(); cnt++)
                    {
                        other_part = search_grid[i][j][cnt];
                        if (part != other_part) // MUST NOT include i=j to avoid div by zero
                        {
                            dn[0] = part->x[0] - other_part->x[0];
                            dn[1] = part->x[1] - other_part->x[1];
                            dist = sqrt(dn[0] * dn[0] + dn[1] * dn[1]);

                            // Only interact if within 2h compact support radius
                            if (dist > 0 && dist < 2.0 * h)
                            {
                                update_a_D(part, other_part, dist);
                                update_dynamical_t(part, other_part);
                            }
                        }
                    }
                }
    
   // if (change_delta_t)
    //    delta_t = 0.3 * min(min(dt_cfl, dt_f), dt_a); // CFL Coefficient applied
}

void SPH_main::neighbour_iterate(SPH_particle* part, bool stencil, bool change_delta_t)
{
    //dt_cfl = 10.0; dt_f = 10.0; dt_a = 10.0;
    
    if (stencil == false) {
        neighbour_iterate_non_stencil(part, stencil, change_delta_t);
    } else {
        // Stencil version logic remains (omitted for brevity, falls back to non_stencil for safety if modified)
        neighbour_iterate_non_stencil(part, false, change_delta_t); 
    }
}

// --------------------------------------------------------------------------------------
// CUBIC SPLINE SMOOTHING KERNELS (Eq 19 from PDF)
// --------------------------------------------------------------------------------------
double SPH_main::calculate_W(double r)
{
    double q = r / h;
    double alpha = 10.0 / (7.0 * PI * h * h);

    if (q >= 0 && q <= 1)
        return alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    else if (q > 1 && q <= 2)
        return alpha * 0.25 * pow((2.0 - q), 3);
    
    return 0.0;
}

double SPH_main::calculate_dW(double r)
{
    double q = r / h;
    double alpha = 10.0 / (7.0 * PI * h * h * h); // Derivative extracts an extra 1/h

    if (q >= 0 && q <= 1)
        return alpha * (-3.0 * q + 2.25 * q * q);
    else if (q > 1 && q <= 2)
        return alpha * -0.75 * pow((2.0 - q), 2);
    
    return 0.0;
}

double distance(SPH_particle& i, SPH_particle& j)
{
    return sqrt(pow((i.x[0] - j.x[0]), 2) + pow((i.x[1] - j.x[1]), 2));
}

// --------------------------------------------------------------------------------------
// DENSITY SMOOTHING (To prevent numerical roughness - PDF Eq 103)
// --------------------------------------------------------------------------------------
void SPH_main::smoothing()
{
    SPH_particle* other_part;
    double dn[2], dist, W;
    #pragma omp parallel for private(other_part, dn, dist, W)schedule(runtime)

   for (size_t ii = 0; ii < particle_list.size(); ii++)
    {
        double numerator_sum = 0;
        double denominator_sum = 0;
        
        for (int i = particle_list[ii].list_num[0] - 1; i <= particle_list[ii].list_num[0] + 1; i++)
            if (i >= 0 && i < max_list[0])
                for (int j = particle_list[ii].list_num[1] - 1; j <= particle_list[ii].list_num[1] + 1; j++)
                    if (j >= 0 && j < max_list[1])
                    {
                        for (unsigned int cnt = 0; cnt < search_grid[i][j].size(); cnt++)
                        {
                            other_part = search_grid[i][j][cnt];
                            dn[0] = particle_list[ii].x[0] - other_part->x[0];
                            dn[1] = particle_list[ii].x[1] - other_part->x[1];
                            dist = sqrt(dn[0] * dn[0] + dn[1] * dn[1]);

                            // MUST include i=j for smoothing (dist < 2h)
                            if (dist < 2.0 * h)
                            {
                                W = calculate_W(dist);
                                numerator_sum += W;
                                denominator_sum += W / other_part->rho;
                            }
                        }
                    }
        if (denominator_sum > 0)
            particle_list[ii].rho = numerator_sum / denominator_sum;
    }
}

// --------------------------------------------------------------------------------------
// TIME INTEGRATION: FORWARD EULER
// --------------------------------------------------------------------------------------
void SPH_main::forward_euler(bool smooth, int rank, int size)
{
    allocate_to_grid();

    if (smooth) smoothing();

    for (size_t i = 0; i != particle_list.size(); i++)
    {
        particle_list[i].a[0] = 0.0;
        particle_list[i].a[1] = particle_list[i].is_boundary ? 0.0 : G;
        particle_list[i].drho_dt = 0.0;

        neighbour_iterate(&(particle_list[i]), false);
    }

    for (size_t i = 0; i != particle_list.size(); i++)
    {
        if (particle_list[i].is_boundary)
        {
            particle_list[i].v[0] = 0;
            particle_list[i].v[1] = 0;
        }
        else
        {
            for (int k = 0; k < 2; k++)
            {
                particle_list[i].x[k] += delta_t * particle_list[i].v[k];
                
                // Wall leakage check
                if (particle_list[i].x[k] < inner_min_x[k] || particle_list[i].x[k] > inner_max_x[k])
                {
                    particle_list[i].x[k] -= delta_t * particle_list[i].v[k];
                    particle_list[i].v[k] = -WALL_RESTITUTION * particle_list[i].v[k];
                }
                else {
                    particle_list[i].v[k] += delta_t * particle_list[i].a[k];
                }
            }
        }
        particle_list[i].rho += delta_t * particle_list[i].drho_dt;
        particle_list[i].calculate_P();
        particle_list[i].calc_index();
    }
}

// --------------------------------------------------------------------------------------
// TIME INTEGRATION: PREDICTOR-CORRECTOR (2nd Order)
// --------------------------------------------------------------------------------------
void SPH_main::predictor_corrector(bool smooth, int rank, int size)
{
    if (smooth) smoothing();
    allocate_to_grid(); // Everyone builds the full grid to find neighbors

    // 1. Array Slicing (Domain Decomposition)
    int n_total = particle_list.size();
    int local_n = n_total / size;
    int remainder = n_total % size;
    
    int start_idx = rank * local_n + std::min(rank, remainder);
    int end_idx = start_idx + local_n + (rank < remainder ? 1 : 0);

    // 2. Physics & Half-Step (ONLY calculate my chunk!)
    for (int i = start_idx; i < end_idx; i++)
    {
        particle_list[i].a[0] = 0.0;
        particle_list[i].a[1] = particle_list[i].is_boundary ? 0.0 : G;
        particle_list[i].drho_dt = 0.0;
        neighbour_iterate(&(particle_list[i]), smooth, false);
    }

    for (int step = 0; step < 2; step++)
    {
        if (step == 0) // HALF-STEP
        {
            for (int i = start_idx; i < end_idx; i++)
            {
                if (particle_list[i].is_boundary) {
                    particle_list[i].v[0] = 0; particle_list[i].v[1] = 0;
                    particle_list[i].prev_v[0] = 0; particle_list[i].prev_v[1] = 0;
                    particle_list[i].prev_rho = particle_list[i].rho;
                }
                else {
                    particle_list[i].prev_x[0] = particle_list[i].x[0];
                    particle_list[i].prev_x[1] = particle_list[i].x[1];
                    particle_list[i].prev_v[0] = particle_list[i].v[0];
                    particle_list[i].prev_v[1] = particle_list[i].v[1];
                    particle_list[i].prev_rho = particle_list[i].rho;

                    for (int k = 0; k < 2; k++) {
                        particle_list[i].x[k] += 0.5 * delta_t * particle_list[i].v[k];
                        if (particle_list[i].x[k] < inner_min_x[k] || particle_list[i].x[k] > inner_max_x[k]) {
                            particle_list[i].x[k] -= 0.5 * delta_t * particle_list[i].v[k];
                            particle_list[i].v[k] = -WALL_RESTITUTION * particle_list[i].prev_v[k];
                        } else {
                            particle_list[i].v[k] += 0.5 * delta_t * particle_list[i].a[k];
                        }
                    }
                }
                particle_list[i].rho += 0.5 * delta_t * particle_list[i].drho_dt;
                particle_list[i].calculate_P();
                particle_list[i].calc_index();
            }
        }
        else // FULL-STEP
        {
            for (int i = start_idx; i < end_idx; i++)
            {
                if (!particle_list[i].is_boundary)
                {
                    for (int k = 0; k < 2; k++) {
                        double temp_half_x = particle_list[i].prev_x[k] + 0.5 * delta_t * particle_list[i].v[k];
                        particle_list[i].x[k] = 2.0 * temp_half_x - particle_list[i].prev_x[k];

                        if (particle_list[i].x[k] < inner_min_x[k] || particle_list[i].x[k] > inner_max_x[k]) {
                            particle_list[i].x[k] = particle_list[i].prev_x[k];
                            particle_list[i].v[k] = -WALL_RESTITUTION * particle_list[i].prev_v[k];
                        } else {
                            double temp_half_v = particle_list[i].prev_v[k] + 0.5 * delta_t * particle_list[i].a[k];
                            particle_list[i].v[k] = 2.0 * temp_half_v - particle_list[i].prev_v[k];
                        }
                    }
                }
                double temp_half_rho = particle_list[i].prev_rho + 0.5 * delta_t * particle_list[i].drho_dt;
                particle_list[i].rho = 2.0 * temp_half_rho - particle_list[i].prev_rho;
                particle_list[i].calculate_P();
                particle_list[i].calc_index();
            }
        }
    }

    // 3. THE MPI MAGIC: Synchronize the array so everyone sees the updated particle positions
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    int current_disp = 0;
    
    for (int i = 0; i < size; i++) {
        int count = (n_total / size) + (i < remainder ? 1 : 0);
        recvcounts[i] = count * sizeof(SPH_particle);
        displs[i] = current_disp;
        current_disp += recvcounts[i];
    }

    // Pass the raw bytes of the C++ struct across the MPI communicator
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                   particle_list.data(), recvcounts.data(), displs.data(), 
                   MPI_BYTE, MPI_COMM_WORLD);
}