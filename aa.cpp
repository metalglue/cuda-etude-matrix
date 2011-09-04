/* vim: set sw=4 sts=4 sta : */

#include <cstdlib>
#include <stdio.h>

typedef double number;

/*        */
/* matrix */
/*        */

class matrix {
public:
    static matrix *new_random_filled(int height, int width);
    static matrix *new_garbage(int height, int width);
    static void delete_matrix(matrix *m);
    class iter_all {
    public:
        iter_all(const matrix *m) { pix = &m->items[0]; beyond = &m->items[m->height * m->width]; }
        void operator ++(int) { pix++; }
        number *operator *() { return pix == beyond ? 0 : pix; }
    private:
        number *pix;
        number *beyond;
    };
    void size(int *height_, int *width_) const { *height_ = height; *width_ = width; }
    void show() const ;
    class iter_row {
    public:
        iter_row(const matrix *m, int row) { pix = &m->items[row * m->width]; beyond = &m->items[(row + 1) * m->width]; }
        void operator ++(int) { pix++; }
        number *operator *() { return pix == beyond ? 0 : pix; }
    private:
        number *pix;
        number *beyond;
    };
    class iter_col {
    public:
        iter_col(const matrix *m, int col) : width(m->width) { pix = &m->items[col]; beyond = &m->items[m->height * m->width + col]; }
        void operator ++(int) { pix += width; }
        number *operator *() { return pix == beyond ? 0 : pix; }
    private:
        const int width;
        number *pix;
        number *beyond;
    };
    static int mul_ijk(const matrix *a, const matrix *b, matrix *r);
private:
    matrix();
    matrix(int height_, int width_) : height(height_), width(width_) {}
private:
    const int height, width;
    number *items;
};

matrix *matrix::new_garbage(int height, int width)
{
    matrix *m = new matrix(height, width);
    m->items = new number[height * width];
    return m;
}

matrix *matrix::new_random_filled(int height, int width)
{
    matrix *m = new_garbage(height, width);
    for (iter_all i(m); *i != 0; i++) {
        number *n = *i;
        *n = rand() % 10;
    }
    return m;
}

void matrix::delete_matrix(matrix *m)
{
    delete[] m->items;
    delete m;
}

void matrix::show() const
{
    int height, width;
    size(&height, &width);
    matrix::iter_all i(this);
    printf("(\n");
    for (int line = 0; line < height; line++) {
        for (int column = 0; column < width; column++) {
            printf("%2.0f ", **i);
            i++;
        }
        printf("\n");
    }
    printf(")\n");
}

int matrix::mul_ijk(const matrix *a, const matrix *b, matrix *r)
{
    int width, height;
    a->size(&height, &width);
    for (int i = 0; i < height; i++) {
        iter_row r_i(r, i);
        for (int j = 0; j < height; j++) {
            iter_row a_i(a, i);
            iter_col b_i(b, j);
            double n = 0;
            for (int k = 0; k < width; k++) {
                n += **a_i * **b_i;
                a_i++; b_i++;
            }
            **r_i = n;
            r_i++;
        }
    }
    return width * width * height * 2;
}

/*           */
/* stopwatch */
/*           */

#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/times.h>

class stopwatch {
public:
    void start();
    void get_lap();
    void show(int flop_count) const ;
private:
    clock_t clock_start, clock_lap;
    tms tms_start, tms_lap;
};

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

void stopwatch::show(int flop_count) const
{
    long ticks = ::sysconf(_SC_CLK_TCK);
    printf("Real: %.2f s\n", (double)(clock_lap - clock_start) / CLOCKS_PER_SEC);
    printf("User: %.2f s\n", (double)(tms_lap.tms_utime - tms_start.tms_utime) / ticks);
    printf(" Sys: %.2f s\n", (double)(tms_lap.tms_stime - tms_start.tms_stime) / ticks);
    printf("%.2f GFLOPS\n", (double)flop_count / (clock_lap - clock_start) * CLOCKS_PER_SEC / 1000 / 1000 / 1000);
}

/*      */
/* main */
/*      */

void
test_0001()
{
    matrix *a = matrix::new_random_filled(2, 3);
    matrix *b = matrix::new_random_filled(3, 2);
    matrix *r = matrix::new_garbage(2, 2);
    stopwatch sw;
    sw.start();
    int flop_count = matrix::mul_ijk(a, b, r);
    sw.get_lap();
    a->show(); b->show(); r->show();
    sw.show(flop_count);
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r);
}

void
test_0002()
{
    matrix *a = matrix::new_random_filled(1000, 1000);
    matrix *b = matrix::new_random_filled(1000, 1000);
    matrix *r = matrix::new_garbage(1000, 1000);
    stopwatch sw;
    sw.start();
    int flop_count = matrix::mul_ijk(a, b, r);
    sw.get_lap();
    sw.show(flop_count);
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r);
}

int main()
{
    test_0002();
}

