#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "common.h"

double size;
double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    //return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
	return (end.tv_sec - start.tv_sec);
}

//
//  keep density constant
//
void set_size( int n )
{
    size = sqrt( density * n );
}

//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p )
{
    srand( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = rand()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
        p[i].x = size*(1.+(k%sx))/(1+sx);
        p[i].y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        p[i].vx = rand()*2-1;
        p[i].vy = rand()*2-1;
    }
    free( shuffle );
}

