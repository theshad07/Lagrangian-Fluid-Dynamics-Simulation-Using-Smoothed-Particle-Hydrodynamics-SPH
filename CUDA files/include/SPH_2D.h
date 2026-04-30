#pragma once
#ifndef SPH_2D_hpp
#define SPH_2D_hpp

#include <vector>
#include <cmath>
#include <iostream>
#include <string>

// -----------------------------------------------------------------------------
// Physical and Simulation Constants 
// (Refactored from macros to modern C++ constants)
// -----------------------------------------------------------------------------
const double MU = 0;          // Dynamic viscosity of water (Pa.s)
const double G = -9.81;           // Gravity in y-direction (m/s^2)
const double PI = 3.14159265359;
const double GAMMA_TAIT = 7.0;    // Stiff equation of state exponent
const double C0 = 20.0;           // Artificial numerical speed of sound (m/s)
const double RHO0 = 1000.0;       // Reference fluid density (kg/m^3)
const double WALL_RESTITUTION = 0.5; // Velocity lost rate upon wall impact

using namespace std;

class SPH_main; // Forward declaration

// -----------------------------------------------------------------------------
// SPH_particle
// Represents an individual fluid parcel (Lagrangian point) associated with a given mass
// -----------------------------------------------------------------------------
class SPH_particle
{
public:
    // Core state variables (2D arrays for x and y components)
    double x[2];    // Position vector
    double v[2];    // Velocity vector
    
    // Macroscopic fluid properties
    double rho = RHO0;  // Density (\rho)
    double p;           // Pressure (P)
    
    // Rate of change variables from Navier-Stokes
    double a[2];        // Acceleration vector (dv/dt)
    double drho_dt;     // Rate of change of density (D\rho/Dt from continuity eq)
    
    // Pointer to simulation environment for grid calculations
    static SPH_main *main_data;
    
    // -------------------------------------------------------------------------
    // Predictor-Corrector Variables 
    // Stores intermediate states for 2nd-order time integration
    // -------------------------------------------------------------------------
    double prev_x[2];
    double prev_v[2];
    double prev_rho;
    
    // Linked-cell neighbor search grid data
    int list_num[2];
    unsigned int grid_index = 0;
    
    // Identifies if particle is fixed wall layer (2h wide) or moving fluid
    bool is_boundary = false;
    
    // Calculates which cell this particle belongs to
    void calc_index(void);
    
    // -------------------------------------------------------------------------
    // Tait Equation of State
    // Closes the system by relating pressure to density variations
    // Formula: P = B * ((rho / rho0)^gamma - 1)
    // -------------------------------------------------------------------------
    void calculate_P() { 
        double B = (RHO0 * C0 * C0) / GAMMA_TAIT;
        p = B * (pow((rho / RHO0), GAMMA_TAIT) - 1.0); 
    }
};

// -----------------------------------------------------------------------------
// SPH_main
// The core simulation environment and solver
// -----------------------------------------------------------------------------
class SPH_main
{
public:
    // Smoothing properties
    double h;           // Characteristic smoothing length
    double h_fac;       // Proportionality factor (e.g., 1.3)
    double dx;          // Initial particle spacing (\Delta x)
    
    // Time stepping properties
    double delta_t;     // Current time step (\Delta t)
    double t_max;       // Total simulation time (e.g., 30s)
    
    // CFL stability constraints
    double dt_cfl = 10, dt_f = 10, dt_a = 10;
    
    // Domain boundaries
    double min_x[2], max_x[2];
    double inner_max_x[2], inner_min_x[2];
    int max_list[2]; // Max grid cells
    
    // Data structures
    vector<SPH_particle> particle_list;
    vector<vector<vector<SPH_particle*> > > search_grid; // Linked cell grid
    
public:
    SPH_main();

    // Setup and Initialization
    void set_values(double h_factor, double DX, double T_MAX);
    void initialise_grid(void);
    void place_points(double *min, double *max);
    
    // Linked Cell Method
    void allocate_to_grid(void);
    
    // Core Physics Calculations
    // Approximates the Navier-Stokes momentum eq and Continuity eq
    void update_a_D(SPH_particle * part, SPH_particle * other_part, double dist, bool stencil = false);
    
    // Neighbor searching
    void neighbour_iterate(SPH_particle *part, bool stencil = false, bool change_delta_t = false);
    void neighbour_iterate_non_stencil(SPH_particle *part, bool stencil, bool change_delta_t);

    // -------------------------------------------------------------------------
    // Smoothing Kernels
    // -------------------------------------------------------------------------
    double calculate_W(double r);    // Cubic spline kernel W(r, h)
    double calculate_dW(double r);   // Derivative of kernel dW/dr
    
    // Time step constraint calculator
    void update_dynamical_t(SPH_particle * part, SPH_particle * other_part);
    friend double distance(SPH_particle & i, SPH_particle & j);
    
    // Density smoothing to prevent numerical "roughness"
    void smoothing();
    
    // Time Integration Schemes
    void forward_euler(bool smooth = false, bool stencil = false);
    void predictor_corrector(bool smooth = false, bool change_delta_t = false);
};
#endif /* SPH_2D_hpp */