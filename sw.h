
#ifndef _SW_H
#define _SW_H

#include <time.h>

class stopwatch {
public:
    void start();
    void get_lap();
    void show(long flop_count) const ;
private:
    timespec starttime, laptime;
};

#endif

