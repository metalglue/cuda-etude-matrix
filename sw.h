
#ifndef _SW_H
#define _SW_H

#include <sys/times.h>

class stopwatch {
public:
    void start();
    void get_lap();
    void show(long flop_count) const ;
private:
    clock_t clock_start, clock_lap;
    tms tms_start, tms_lap;
};

#endif

