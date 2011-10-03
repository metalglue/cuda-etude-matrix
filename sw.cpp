
#include <time.h>
#include <stdio.h>
#include "sw.h"

void stopwatch::start()
{
    ::clock_gettime(CLOCK_REALTIME, &starttime);
}

void stopwatch::get_lap()
{
    ::clock_gettime(CLOCK_REALTIME, &laptime);
}

void stopwatch::show(long flop_count) const
{
    double elapsed = laptime.tv_sec + (double)laptime.tv_nsec / 1000000000;
    elapsed -= starttime.tv_sec + (double)starttime.tv_nsec / 1000000000;
    printf("Real: %.2f s\n", elapsed);
    printf("%.2f GFLOPS\n", flop_count / elapsed / 1000000000);
}

