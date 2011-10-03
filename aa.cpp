/* vim: set sw=4 sts=4 sta : */

#include <cstdlib>
#include <stdio.h>
#include <assert.h>
#include "cf.h"
#include "mx.h"
#include "sw.h"

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
    long flop_count = matrix::mul_ijk(a, b, r);
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
    matrix *a = matrix::new_random_filled(DIM, DIM);
    matrix *b = matrix::new_random_filled(DIM, DIM);
    matrix *r = matrix::new_garbage(DIM, DIM);
    stopwatch sw;
    sw.start();
    long flop_count = matrix::mul_ijk(a, b, r);
    sw.get_lap();
    sw.show(flop_count);
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r);
}

void
test_0003()
{
    matrix *a = matrix::new_random_filled(2, 3);
    matrix *b = matrix::new_random_filled(3, 2);
    matrix *r = matrix::new_garbage(2, 2);
    stopwatch sw;
    sw.start();
    long flop_count = matrix::mul_kij(a, b, r);
    sw.get_lap();
    a->show(); b->show(); r->show();
    sw.show(flop_count);
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r);
}

void
test_0004()
{
    matrix *a = matrix::new_random_filled(DIM, DIM);
    matrix *b = matrix::new_random_filled(DIM, DIM);
    matrix *r = matrix::new_garbage(DIM, DIM);
    stopwatch sw;
    sw.start();
    long flop_count = matrix::mul_kij(a, b, r);
    sw.get_lap();
    sw.show(flop_count);
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r);
}

void
test_0005()
{
    matrix *a = matrix::new_random_filled(DIM, DIM);
    matrix *b = matrix::new_random_filled(DIM, DIM);
    matrix *r1 = matrix::new_garbage(DIM, DIM);
    matrix *r2 = matrix::new_garbage(DIM, DIM);
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
    assert(matrix::eq(r1, r2));
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r1);
    matrix::delete_matrix(r2);
}

void
test_0006()
{
    matrix *a = matrix::new_random_filled(2, 3);
    matrix *b = matrix::new_random_filled(3, 2);
    matrix *r = matrix::new_garbage(2, 2);
    stopwatch sw;
    sw.start();
    long flop_count = matrix::mul_ikj(a, b, r);
    sw.get_lap();
    a->show(); b->show(); r->show();
    sw.show(flop_count);
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r);
}

void
test_0007()
{
    matrix *a = matrix::new_random_filled(DIM, DIM);
    matrix *b = matrix::new_random_filled(DIM, DIM);
    matrix *r1 = matrix::new_garbage(DIM, DIM);
    matrix *r2 = matrix::new_garbage(DIM, DIM);
    matrix *r3 = matrix::new_garbage(DIM, DIM);
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
    assert(matrix::eq(r1, r2));
    assert(matrix::eq(r2, r3));
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r1);
    matrix::delete_matrix(r2);
}

int main()
{
    test_0007();
}

