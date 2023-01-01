#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include<mem.h>
#include "common.h"

extern double size;
void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

void compute_forces(particle_t * particles, int n, int iteration)
{
    for(int tid=0;tid<n;tid++)
    {
        particles[tid].ax = particles[tid].ay = 0;
        for(int j = 0 ; j < n ; j++)
            apply_force_gpu(particles[tid], particles[j]);
    }
}

#define MaxDistance 20
#define DENS	300

int *threadworks;
int *threadworksl;
void compute_forces2(particle_t * particles, int n, int iteration)
{
	    int maxidx=0;
	if(iteration%5 == 0)
	{
	    for(int tid=0;tid<n;tid++)
	    {
            int idx = 0;
            for(int j=0;j<n;j++)
            {
                double dx = particles[tid].x - particles[j].x;
                double dy = particles[tid].y - particles[j].y;
                double r2 = dx * dx + dy * dy;
                if( r2 <= cutoff*cutoff*MaxDistance*MaxDistance )
                {
                    threadworks[tid*DENS+idx] = j;
                    idx ++;
                    if(idx == DENS)
                        break;
                }
            }
            threadworksl[tid] = idx;
            if(idx>maxidx)maxidx=idx;
	    }
	}

    for(int tid=0;tid<n;tid++)
    {
        particles[tid].ax = particles[tid].ay = 0;
        for(int j = 0 ; j < threadworksl[tid] ; j++)
            apply_force_gpu(particles[tid], particles[ threadworks[tid*DENS+j] ]);
    }
}
//double maxx,maxy;
void move(particle_t * particles, int n, double size)
{

    for(int tid=0;tid<n;tid++)
    {
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

/*
        if(p->x>maxx)
            maxx=p->x;
        if(p->y>maxy)
            maxy=p->y;*/
    }
}


int main( int argc, char **argv )
{
    //maxx=0;maxy=0;
    int n = 1000;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );
    threadworks = new int[n*DENS];
    threadworksl = new int[n];

    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
		compute_forces2(particles, n, step);
		move(particles, n, size);
    }
    simulation_time = read_timer( ) - simulation_time;
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    free( particles );

    return 0;
}
