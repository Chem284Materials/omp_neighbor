#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <list>
#include <omp.h>

// Typedefs used for the neighborlist evaluation
typedef std::pair<int,int> pairT; // A pair of interacting particles
typedef std::list<pairT> neighborsT; // A list of pairs

class MDSimulation {
public:
  MDSimulation(double box_size_in, int nparticles_in);
  void run(int nsteps, double dt);
private:
  double cutoff;                 // Cutoff radius for Lennard-Jones potential
  double cutoff2;                // Square of the cutoff radius
  double lj_potential_at_cutoff; // Value of the Lennard-Jones potential at the cutoff
  double box_size;               // Length of each side of the periodic simulation cell, which is cubic.
  int nparticles;                // Number of particles in the simulation
  std::vector<std::array<double, 3>> positions;  // Position of the particles
  std::vector<std::array<double, 3>> velocities; // Velocities of the particles
  std::vector<std::array<double, 3>> forces;     // Forces on the particles

  neighborsT neighbors;           // The list of neighbor particles
  double neighbor_cutoff;         // Cutoff radius for the neighborlist evaluation
  double neighbor_cutoff2;        // Square of the neighborlist cutoff
  int neighbor_steps;             // Number of steps between neighborlist evaluations

  double minimum_image(double dist);
  double lj_potential(double r2);
  double lj_potential_with_cutoff(double r2);
  double lj_force(double r2);
  double lj_force_with_cutoff(double r2);
  void evaluate_neighbors();
};

/*! \brief Initialize a molecular dynamics simulation
 *
 * \param [in]  box_size_in
 *                   Length of each side of the periodic simulation cell, which is cubic.
 * \param [in]  nparticles_in
 *                   Number of particles in the simulation.
 */
MDSimulation::MDSimulation(double box_size_in, int nparticles_in) {
  box_size = box_size_in;
  nparticles = nparticles_in;
  cutoff = 2.5;
  cutoff2 = cutoff * cutoff;

  // The neighborlist cutoff should be larger than the cutoff
  neighbor_cutoff = 1.3 * cutoff;
  neighbor_cutoff2 = neighbor_cutoff * neighbor_cutoff;
  neighbor_steps = 50;

  // Initialize the particles on a rough grid
  int particles_per_side = std::ceil( std::pow(nparticles, 1.0/3.0) );
  double particle_spacing = box_size / (particles_per_side + 1);
  for (int iparticle = 0; iparticle < nparticles; iparticle++) {
    int ix = iparticle % particles_per_side;
    int iy = (iparticle / particles_per_side) % particles_per_side;
    int iz = iparticle / (particles_per_side * particles_per_side);
    positions.push_back({
      particle_spacing * ix + ( 0.5 * particle_spacing ),
      particle_spacing * iy + ( 0.5 * particle_spacing ),
      particle_spacing * iz + ( 0.5 * particle_spacing )
    });
  }

  // Initialize the velocities randomly
  for (int iparticle = 0; iparticle < nparticles; ++iparticle) {
    /* The random number generators here use the particle index as the seed.
           This isn't something you would normally do, but it is quite helpful
           in this case for the purpose of ensuring that the velocities are 
           reproducible with respect to parallelization. */
    std::mt19937 gen(iparticle);
    std::uniform_real_distribution<double> random_vel(-0.5, 0.5);
    velocities.push_back({random_vel(gen), random_vel(gen), random_vel(gen)});
  }

  // Initialize the forces
  for (int iparticle = 0; iparticle < nparticles; ++iparticle) {
    forces.push_back({0.0, 0.0, 0.0});
  }

  // Determine the Lennard-Jones potential at the cutoff
  lj_potential_at_cutoff = lj_potential(cutoff2);
}

/*! \brief Evaluate the Lennard-Jones potential associated with a specific particle separation.
 *
 * \param [in]  r2
 *                   Square of the distance between two particles.
 */
double MDSimulation::lj_potential(double r2) {
  double inv_r2 = 1.0 / r2;
  double inv_r6 = inv_r2 * inv_r2 * inv_r2;
  return 4.0 * (inv_r6 * inv_r6 - inv_r6);
}

/*! \brief Evaluate the Lennard-Jones potential associated with a specific particle separation, with a cutoff.
 *
 * \param [in]  r2
 *                   Square of the distance between two particles.
 */
double MDSimulation::lj_potential_with_cutoff(double r2) {
  if (r2 < cutoff2) {
    return lj_potential(r2) - lj_potential_at_cutoff;
  }
  else {
    return 0.0;
  }
}

/*! \brief Evaluate the Lennard-Jones force for a specific particle separation.
 *
 * \param [in]  r2
 *                   Square of the distance between two particles.
 */
double MDSimulation::lj_force(double r2) {
  double inv_r2 = 1.0 / r2;
  double inv_r6 = inv_r2 * inv_r2 * inv_r2;
  return 24.0 * inv_r2 * (2.0 * inv_r6 * inv_r6 - inv_r6);
}

/*! \brief Evaluate the Lennard-Jones force for a specific particle separation, with a cutoff.
 *
 * \param [in]  r2
 *                   Square of the distance between two particles.
 */
double MDSimulation::lj_force_with_cutoff(double r2) {
  if (r2 < cutoff2) {
    return lj_force(r2);
  }
  else {
    return 0.0;
  }
}

/*! \brief Account for periodic boundary conditions by returning the smallest magnitude distance between two particles.
 *
 * \param [in]  dist
 *                   Distance between the particles
 */
double MDSimulation::minimum_image(double dist) {
  if (dist > 0.5 * box_size) dist -= box_size;
  if (dist < -0.5 * box_size) dist += box_size;
  return dist;
}

/*! \brief Evaluate the list of particles that are close enough to potentially interact via the Lennard-Jones potential
 */
void MDSimulation::evaluate_neighbors() {
  neighbors.clear();
  for (int iparticle = 0; iparticle < nparticles; iparticle++) {
    for (int jparticle = iparticle + 1; jparticle < nparticles; jparticle++) {

      double dx = minimum_image(positions[iparticle][0] - positions[jparticle][0]);
      double dy = minimum_image(positions[iparticle][1] - positions[jparticle][1]);
      double dz = minimum_image(positions[iparticle][2] - positions[jparticle][2]);
      double r2 = (dx * dx) + (dy * dy) + (dz * dz);


      if (r2 < neighbor_cutoff2) {
        neighbors.push_back( pairT(iparticle, jparticle) );
      }
    }
  }
}

/*! \brief Run a molecular dynamics simulation.
 *
 * \param [in]  nsteps
 *                   Number of time integration steps to perform.
 * \param [in]  dt
 *                   Size of the timestep (reduced Lennard-Jones units).
 */
void MDSimulation::run(int nsteps, double dt) {

  double start_time = omp_get_wtime();
  double neighborlist_time = 0.0;
  double forces_time = 0.0;

  // Main simulation loop
  for (int istep = 0; istep < nsteps; ++istep) {

    // Update the particle velocities and positions
    for (int iparticle = 0; iparticle < nparticles; ++iparticle) {

      // Update the positions
      positions[iparticle][0] += velocities[iparticle][0] * dt;
      positions[iparticle][1] += velocities[iparticle][1] * dt;
      positions[iparticle][2] += velocities[iparticle][2] * dt;

      // Apply periodic boundary conditions; ensure that particles outside the box wrap to the other side
      for (int idimension = 0; idimension < 3; ++idimension) {
        if (positions[iparticle][idimension] < 0.0) positions[iparticle][idimension] += box_size;
        if (positions[iparticle][idimension] >= box_size) positions[iparticle][idimension] -= box_size;
      }

    }

    // Check if we should re-evaluate the neighborlist
    if ( istep % neighbor_steps == 0 ) {
      double neighborlist_start_time = omp_get_wtime();
      evaluate_neighbors();
      neighborlist_time += omp_get_wtime() - neighborlist_start_time;
    }

    // Zero the energy and forces
    double forces_start_time = omp_get_wtime();
    double potential_energy = 0.0;
    double kinetic_energy = 0.0;

    for (int iparticle = 0; iparticle < nparticles; ++iparticle) {
      forces[iparticle] = {0.0, 0.0, 0.0};
    }

    // Evaluate the energy and forces, using a neighborlist
    for (auto pair = neighbors.begin(); pair != neighbors.end(); pair++) {
      int iparticle = pair->first;
      int jparticle = pair->second;

      double dx = minimum_image(positions[iparticle][0] - positions[jparticle][0]);
      double dy = minimum_image(positions[iparticle][1] - positions[jparticle][1]);
      double dz = minimum_image(positions[iparticle][2] - positions[jparticle][2]);
      double r2 = (dx * dx) + (dy * dy) + (dz * dz);

      double f = lj_force_with_cutoff(r2);
      forces[iparticle][0] += f * dx;
      forces[iparticle][1] += f * dy;
      forces[iparticle][2] += f * dz;
      forces[jparticle][0] -= f * dx;
      forces[jparticle][1] -= f * dy;
      forces[jparticle][2] -= f * dz;

      potential_energy += lj_potential_with_cutoff(r2);
    }
    forces_time += omp_get_wtime() - forces_start_time;

    // Compute the kinetic energy
    for (int iparticle = 0; iparticle < nparticles; ++iparticle) {
      kinetic_energy += 0.5 * velocities[iparticle][0] * velocities[iparticle][0];
      kinetic_energy += 0.5 * velocities[iparticle][1] * velocities[iparticle][1];
      kinetic_energy += 0.5 * velocities[iparticle][2] * velocities[iparticle][2];
    }

    // Update the particle velocities
    for (int iparticle = 0; iparticle < nparticles; ++iparticle) {
      velocities[iparticle][0] += forces[iparticle][0] * dt;
      velocities[iparticle][1] += forces[iparticle][1] * dt;
      velocities[iparticle][2] += forces[iparticle][2] * dt;
    }

    // Print output
    std::cout << "Iteration " << istep << '\n';
    std::cout << "    Potential Energy: " << potential_energy << '\n';
    std::cout << "    Kinetic Energy:   " << kinetic_energy << '\n';
    std::cout << "    Total Energy:     " << potential_energy + kinetic_energy << "\n\n";
  }

  std::cout << "Simulation completed." << '\n';

  std::cout << "Total runtime:        " << omp_get_wtime() - start_time << '\n';
  std::cout << "   Neighborlist time: " << neighborlist_time << '\n';
  std::cout << "   Forces time:       " << forces_time << '\n';
}

int main(int argc, char** argv) {
  MDSimulation mysimulation(20.0, 1000);
  mysimulation.run(100, 0.005);
  return 0;
}
