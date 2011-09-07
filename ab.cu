/* vim: set sw=4 sts=4 sta : */

#include <cstdlib>
#include <stdio.h>
#include <assert.h>
#include <cutil.h>
#include <cutil_inline.h>

const int DIM = 1000;
const int DIM_SMALL = 200;
typedef float number;

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
        void ok() { assert(pix <= beyond); }
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
        void operator +=(int d) { pix += d; }
        number *operator *() { return pix == beyond ? 0 : pix; }
        void ok() { assert(pix <= beyond); }
    private:
        number *pix;
        number *beyond;
    };
    class iter_col {
    public:
        iter_col(const matrix *m, int col) : width(m->width) { pix = &m->items[col]; beyond = &m->items[m->height * m->width + col]; }
        void operator ++(int) { pix += width; }
        number *operator *() { return pix == beyond ? 0 : pix; }
        void ok() { assert(pix <= beyond); }
    private:
        const int width;
        number *pix;
        number *beyond;
    };
    static bool eq(const matrix *a, const matrix *b);
    static long mul_ijk(const matrix *a, const matrix *b, matrix *r);
    static long mul_kij(const matrix *a, const matrix *b, matrix *r);
    static long mul_ikj(const matrix *a, const matrix *b, matrix *r);
    static long mul_cuda_1(const matrix *a, const matrix *b, matrix *r);
    static long mul_cuda_2(const matrix *a, const matrix *b, matrix *r);
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
            printf("%2.0f ", (double)**i);
            i++;
        }
        printf("\n");
    }
    printf(")\n");
}

bool matrix::eq(const matrix *a, const matrix *b)
{
    assert(a->height == b->height && a->width == b->width);
    iter_all ia(a);
    iter_all ib(b);
    for (; *ia; ia++, ib++)
        if (**ia != **ib)
            return false;
    return true;
}

long matrix::mul_ijk(const matrix *a, const matrix *b, matrix *r)
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
    return (long)width * width * height * 2;
}

long matrix::mul_kij(const matrix *a, const matrix *b, matrix *r)
{
    int width, height, h_div, h_mod;
    a->size(&height, &width);
    h_div = height / 16;
    h_mod = height % 16;
    for (iter_all i(r); *i; i++)
        **i = 0;
    for (int k = 0; k < width; k++) {
        iter_col a_i(a, k);
        for (int i = 0; i < height; i++) {
            number n = **a_i; a_i++;
            iter_row b_i(b, k);
            iter_row r_i(r, i);
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

long matrix::mul_ikj(const matrix *a, const matrix *b, matrix *r)
{
    int width, height, h_div, h_mod;
    a->size(&height, &width);
    h_div = height / 16;
    h_mod = height % 16;
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
    void show(long flop_count) const ;
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

void stopwatch::show(long flop_count) const
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

