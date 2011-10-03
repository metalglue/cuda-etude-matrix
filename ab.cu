/* vim: set sw=4 sts=4 sta : */

//#include <cstdlib>
//#include <stdio.h>
//#include <assert.h>
#include <cutil.h>
#include <cutil_inline.h>
#include "cf.h"
#include "mx.h"
#include "sw.h"

/*        */
/* matrix */
/*        */

__global__ void mul_cuda_1_kernel(number *a, number *b, number *r, int height, int width)
{
    int tx = blockIdx.x;
    int ty = blockIdx.y;
    number n = 0;
    for (int k = 0; k < width; k++) {
        n += a[ty * width + k] * b[k * height + tx];
    }
    r[ty * height + tx] = n;
}

long matrix::mul_cuda_1(const matrix *a, const matrix *b, matrix *r)
{
    int height, width;
    a->size(&height, &width);
    void *aa, *bb, *rr;
    CUDA_SAFE_CALL( cudaMalloc(&aa, height * width * sizeof(number)) );
    matrix::iter_all ia(a);
    CUDA_SAFE_CALL( cudaMemcpy(aa, *ia, height * width * sizeof(number), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMalloc(&bb, height * width * sizeof(number)) );
    matrix::iter_all ib(b);
    CUDA_SAFE_CALL( cudaMemcpy(bb, *ib, height * width * sizeof(number), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMalloc(&rr, height * height * sizeof(number)) );
    dim3 grid(height, height);
    dim3 block(1, 1);
    mul_cuda_1_kernel<<<grid, block>>>((number *)aa, (number *)bb, (number *)rr, height, width);
    matrix::iter_all ir(r);
    CUDA_SAFE_CALL( cudaMemcpy(*ir, rr, height * height * sizeof(number), cudaMemcpyDeviceToHost) );
    cudaFree(rr); cudaFree(bb); cudaFree(aa);
    return (long)width * width * height * 2;
}

__global__ void mul_cuda_2_kernel(number *a, number *b, number *r, int height, int width)
{
    int h_div, h_mod;
    h_div = height / 16;
    h_mod = height % 16;
    for (int i = 0; i < height; i++) {
        {
            number *r_i = &r[i * height];
            number *r_i_beyond = &r[(i + 1) * height];
            for (; r_i != r_i_beyond; ++r_i)
                *r_i = 0;
        }
        number *a_i = &a[i * width];
        for (int k = 0; k < width; k++) {
            number n = *a_i++;
            number *r_i = &r[i * height];
            number *b_i = &b[k * height];
            for (int j = 0; j < h_div; j++) {
                r_i[0] += b_i[0] * n;
                r_i[1] += b_i[1] * n;
                r_i[2] += b_i[2] * n;
                r_i[3] += b_i[3] * n;
                r_i[4] += b_i[4] * n;
                r_i[5] += b_i[5] * n;
                r_i[6] += b_i[6] * n;
                r_i[7] += b_i[7] * n;
                r_i[8] += b_i[8] * n;
                r_i[9] += b_i[9] * n;
                r_i[10] += b_i[10] * n;
                r_i[11] += b_i[11] * n;
                r_i[12] += b_i[12] * n;
                r_i[13] += b_i[13] * n;
                r_i[14] += b_i[14] * n;
                r_i[15] += b_i[15] * n;
                r_i += 16; b_i += 16;
            }
            for (int j = 0; j < h_mod; j++) {
                *r_i += *b_i * n;
                ++r_i; ++b_i;
            }
        }
    }
}

long matrix::mul_cuda_2(const matrix *a, const matrix *b, matrix *r)
{
    int height, width;
    a->size(&height, &width);
    void *aa, *bb, *rr;
    CUDA_SAFE_CALL( cudaMalloc(&aa, height * width * sizeof(number)) );
    matrix::iter_all ia(a);
    CUDA_SAFE_CALL( cudaMemcpy(aa, *ia, height * width * sizeof(number), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMalloc(&bb, height * width * sizeof(number)) );
    matrix::iter_all ib(b);
    CUDA_SAFE_CALL( cudaMemcpy(bb, *ib, height * width * sizeof(number), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMalloc(&rr, height * height * sizeof(number)) );
    dim3 grid(1, 1);
    dim3 block(1, 1);
    mul_cuda_2_kernel<<<grid, block>>>((number *)aa, (number *)bb, (number *)rr, height, width);
    matrix::iter_all ir(r);
    CUDA_SAFE_CALL( cudaMemcpy(*ir, rr, height * height * sizeof(number), cudaMemcpyDeviceToHost) );
    cudaFree(rr); cudaFree(bb); cudaFree(aa);
    return (long)width * width * height * 2;
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
    long flop_count = matrix::mul_cuda_1(a, b, r);
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
    matrix *r1 = matrix::new_garbage(DIM, DIM);
    matrix *r2 = matrix::new_garbage(DIM, DIM);
    matrix *r3 = matrix::new_garbage(DIM, DIM);
    stopwatch sw;
    sw.start();
    long flop_count;
    flop_count = matrix::mul_kij(a, b, r1);
    sw.get_lap();
    sw.show(flop_count);
    sw.start();
    flop_count = matrix::mul_ikj(a, b, r2);
    sw.get_lap();
    sw.show(flop_count);
    sw.start();
    flop_count = matrix::mul_cuda_1(a, b, r3);
    sw.get_lap();
    sw.show(flop_count);
    assert(matrix::eq(r1, r2));
    assert(matrix::eq(r2, r3));
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r1);
    matrix::delete_matrix(r2);
    matrix::delete_matrix(r3);
}

void
test_0003()
{
    matrix *a = matrix::new_random_filled(2, 3);
    matrix *b = matrix::new_random_filled(3, 2);
    matrix *r = matrix::new_garbage(2, 2);
    stopwatch sw;
    sw.start();
    long flop_count = matrix::mul_cuda_2(a, b, r);
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
    matrix *a = matrix::new_random_filled(DIM_SMALL, DIM_SMALL);
    matrix *b = matrix::new_random_filled(DIM_SMALL, DIM_SMALL);
    matrix *r1 = matrix::new_garbage(DIM_SMALL, DIM_SMALL);
    matrix *r2 = matrix::new_garbage(DIM_SMALL, DIM_SMALL);
    matrix *r3 = matrix::new_garbage(DIM_SMALL, DIM_SMALL);
    matrix *r4 = matrix::new_garbage(DIM_SMALL, DIM_SMALL);
    stopwatch sw;
    sw.start();
    long flop_count;
    flop_count = matrix::mul_kij(a, b, r1);
    sw.get_lap();
    sw.show(flop_count);
    sw.start();
    flop_count = matrix::mul_ikj(a, b, r2);
    sw.get_lap();
    sw.show(flop_count);
    sw.start();
    flop_count = matrix::mul_cuda_1(a, b, r3);
    sw.get_lap();
    sw.show(flop_count);
    sw.start();
    flop_count = matrix::mul_cuda_2(a, b, r4);
    sw.get_lap();
    sw.show(flop_count);
    assert(matrix::eq(r1, r2));
    assert(matrix::eq(r2, r3));
    assert(matrix::eq(r3, r4));
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r1);
    matrix::delete_matrix(r2);
    matrix::delete_matrix(r3);
    matrix::delete_matrix(r4);
}

int main(int argc, char *argv[])
{
    CUT_DEVICE_INIT(argc, argv);

    {
        void *dummy;
        CUDA_SAFE_CALL( cudaMalloc(&dummy, 1) );
        CUDA_SAFE_CALL( cudaFree(dummy) );
    }

    test_0004();

    CUT_EXIT(argc, argv);
    return 0;
}

