#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N_PARTICLES 1000  // Number of particles
#define TIME_STEPS 1000   // Number of simulation time steps
#define DT 0.001          // Time step
#define L 10.0            // Box size
#define CUTOFF 2.5        // Cutoff radius
#define EPSILON 1.0       // Depth of LJ potential
#define SIGMA 1.0         // Distance at which potential is zero

typedef struct {
    double x, y;       // Position
    double vx, vy;     // Velocity
    double fx, fy;     // Force
} Particle;

// Lennard-Jones force calculation
void compute_force(Particle *particles, int start, int end, int n) {
    for (int i = start; i < end; i++) {
        particles[i].fx = particles[i].fy = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;

                // Periodic boundary conditions
                if (dx > L / 2) dx -= L;
                if (dx < -L / 2) dx += L;
                if (dy > L / 2) dy -= L;
                if (dy < -L / 2) dy += L;

                double r2 = dx * dx + dy * dy;
                if (r2 < CUTOFF * CUTOFF) {
                    double r2_inv = 1.0 / r2;
                    double r6_inv = r2_inv * r2_inv * r2_inv;
                    double force = 24 * EPSILON * r6_inv * (2 * r6_inv - 1) * r2_inv;

                    particles[i].fx += force * dx;
                    particles[i].fy += force * dy;
                }
            }
        }
    }
}

// Velocity Verlet integration
void integrate(Particle *particles, int start, int end) {
    for (int i = start; i < end; i++) {
        particles[i].vx += 0.5 * particles[i].fx * DT;
        particles[i].vy += 0.5 * particles[i].fy * DT;

        particles[i].x += particles[i].vx * DT;
        particles[i].y += particles[i].vy * DT;

        // Periodic boundary conditions
        if (particles[i].x < 0) particles[i].x += L;
        if (particles[i].x >= L) particles[i].x -= L;
        if (particles[i].y < 0) particles[i].y += L;
        if (particles[i].y >= L) particles[i].y -= L;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = N_PARTICLES / size;
    int start = rank * local_n;
    int end = start + local_n;

    Particle *particles = (Particle *)malloc(N_PARTICLES * sizeof(Particle));

    if (rank == 0) {
        // Initialize particles with random positions and velocities
        for (int i = 0; i < N_PARTICLES; i++) {
            particles[i].x = L * (rand() / (double)RAND_MAX);
            particles[i].y = L * (rand() / (double)RAND_MAX);
            particles[i].vx = 0.0;
            particles[i].vy = 0.0;
        }
    }

    // Broadcast particle data to all processes
    MPI_Bcast(particles, N_PARTICLES * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    for (int t = 0; t < TIME_STEPS; t++) {
        compute_force(particles, start, end, N_PARTICLES);

        // Gather forces from all processes
        MPI_Allgather(MPI_IN_PLACE, local_n * sizeof(Particle), MPI_BYTE,
                      particles, local_n * sizeof(Particle), MPI_BYTE, MPI_COMM_WORLD);

        integrate(particles, start, end);
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Simulation completed in %f seconds.\n", end_time - start_time);
    }

    free(particles);
    MPI_Finalize();
    return 0;
}
