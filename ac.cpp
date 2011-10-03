
#include <cstdlib>
#include <stdio.h>
#include <assert.h>
#include "cf.h"
#include "mx.h"
#include "sw.h"

/*        */
/* matrix */
/*        */

long matrix::mul_openmp(const matrix *a, const matrix *b, matrix *r)
{
    int width, height, h_div, h_mod;
    a->size(&height, &width);
    h_div = height / 16;
    h_mod = height % 16;
#   pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (iter_row r_i(r, i); *r_i; r_i++)
            **r_i = 0;
        iter_row a_i(a, i);
        for (int k = 0; k < width; k++) {
            number n = **a_i; a_i++;
            iter_row r_i(r, i);
            iter_row b_i(b, k);
            for (int j = 0; j < h_div; j++) {
                (*r_i)[0] += (*b_i)[0] * n;
                (*r_i)[1] += (*b_i)[1] * n;
                (*r_i)[2] += (*b_i)[2] * n;
                (*r_i)[3] += (*b_i)[3] * n;
                (*r_i)[4] += (*b_i)[4] * n;
                (*r_i)[5] += (*b_i)[5] * n;
                (*r_i)[6] += (*b_i)[6] * n;
                (*r_i)[7] += (*b_i)[7] * n;
                (*r_i)[8] += (*b_i)[8] * n;
                (*r_i)[9] += (*b_i)[9] * n;
                (*r_i)[10] += (*b_i)[10] * n;
                (*r_i)[11] += (*b_i)[11] * n;
                (*r_i)[12] += (*b_i)[12] * n;
                (*r_i)[13] += (*b_i)[13] * n;
                (*r_i)[14] += (*b_i)[14] * n;
                (*r_i)[15] += (*b_i)[15] * n;
                r_i += 16; b_i += 16;
            }
            for (int j = 0; j < h_mod; j++) {
                **r_i += **b_i * n;
                r_i++; b_i++;
            }
        }
    }
    return (long)width * width * height * 2;
}

/*      */
/* main */
/*      */

void
test_0008()
{
    matrix *a = matrix::new_random_filled(DIM, DIM);
    matrix *b = matrix::new_random_filled(DIM, DIM);
    matrix *r1 = matrix::new_garbage(DIM, DIM);
    matrix *r2 = matrix::new_garbage(DIM, DIM);
    matrix *r3 = matrix::new_garbage(DIM, DIM);
    matrix *r4 = matrix::new_garbage(DIM, DIM);
    stopwatch sw;
    sw.start();
    long flop_count;
    flop_count = matrix::mul_ijk(a, b, r1);
    sw.get_lap();
    sw.show(flop_count);
    sw.start();
    flop_count = matrix::mul_kij(a, b, r2);
    sw.get_lap();
    sw.show(flop_count);
    sw.start();
    flop_count = matrix::mul_ikj(a, b, r3);
    sw.get_lap();
    sw.show(flop_count);
    sw.start();
    flop_count = matrix::mul_openmp(a, b, r4);
    sw.get_lap();
    sw.show(flop_count);
    assert(matrix::eq(r1, r2));
    assert(matrix::eq(r2, r3));
    assert(matrix::eq(r3, r4));
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r1);
    matrix::delete_matrix(r2);
}

int main()
{
    test_0008();
}

