#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

extern double size;
__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}


__global__ void compute_forces_gpu(particle_t * particles, int n)
{

  // Get thread (particle) ID
  //int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = (blockId.y*gridDim.x+blockId.x)*blockDim.y*blockDim.x +
	threadId.y*blockDim.x + threadId.x;
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
  for(int j = 0 ; j < n ; j++)
    apply_force_gpu(particles[tid], particles[j]);

}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }
}


int main( int argc, char **argv )
{    
    cudaThreadSynchronize(); 

    int n = read_int( argc, argv, "-n", 1000 );
	int BlockDimX = read_int( argc, argv, "-bx", 16 );
	int BlockDimY = read_int( argc, argv, "-by", 8 );
	int NUM_THREADS = read_int( argc, argv, "-t", 64 );
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    set_size( n );

	dim3 bd(BlockDimY, BlockDimX);
	dim3 gd((n+BlockDimY-1)/BlockDimY, (n+BlockDimX-1)/BlockDimX);
	int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
		
    init_particles( n, particles );
	//cudaMalloc((void **) &shuffle, n * sizeof(int));
	
    cudaThreadSynchronize();
    double copy_time = read_timer( );
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();
	copy_time = read_timer( ) - copy_time;
    
	//init_particles <<gd, bd>> (n, particles);
	
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
		compute_forces_gpu <<< gd, bd >>> (d_particles, n);
		//compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);
     
		move_gpu <<< gd, bd >>> (d_particles, n, size);
		//move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
    }
	
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
	
    return 0;
}
