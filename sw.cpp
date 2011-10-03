
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include "sw.h"

void stopwatch::start()
{
    clock_start = ::clock();
    ::times(&tms_start);
}

void stopwatch::get_lap()
{
    clock_lap = ::clock();
    ::times(&tms_lap);
}

void stopwatch::show(long flop_count) const
{
    long ticks = ::sysconf(_SC_CLK_TCK);
    printf("Real: %.2f s\n", (double)(clock_lap - clock_start) / CLOCKS_PER_SEC);
    printf("User: %.2f s\n", (double)(tms_lap.tms_utime - tms_start.tms_utime) / ticks);
    printf(" Sys: %.2f s\n", (double)(tms_lap.tms_stime - tms_start.tms_stime) / ticks);
    printf("%.2f GFLOPS\n", (double)flop_count / (clock_lap - clock_start) * CLOCKS_PER_SEC / 1000 / 1000 / 1000);
}

