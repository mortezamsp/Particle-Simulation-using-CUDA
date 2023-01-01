#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 128

extern double size;
__global__ void run_simulation (particle_t * particles, int n, double size)
{

	for(int step=0; step<n; step++)
	{
		// Get thread (particle) ID
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= n) return;

		// compute force
		particles[tid].ax = particles[tid].ay = 0;
		for(int j = 0 ; j < n ; j++)
		{
		  double dx = particles[j].x - particles[tid].x;
		  double dy = particles[j].y - particles[tid].y;
		  double r2 = dx * dx + dy * dy;
		  if( r2 > cutoff*cutoff )
			  return;
		  //r2 = fmax( r2, min_r*min_r );
		  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
		  double r = sqrt( r2 );

		  double coef = ( 1 - cutoff / r ) / r2 / mass;
		  particles[tid].ax += coef * dx;
		  particles[tid].ay += coef * dy;
		}
		
		// move
		particle_t * p = &particles[tid];
		p->vx += p->ax * dt;
		p->vy += p->ay * dt;
		p->x  += p->vx * dt;
		p->y  += p->vy * dt;

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
		
		__syncthreads();
	}
}




int main( int argc, char **argv )
{    
    cudaThreadSynchronize(); 
    int n = read_int( argc, argv, "-n", 1000 );
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    set_size( n );
    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    cudaThreadSynchronize();
	int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    double simulation_time = read_timer( );
	run_simulation <<<blks, NUM_THREADS >>> (d_particles, n, size);
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
	
    return 0;
}
