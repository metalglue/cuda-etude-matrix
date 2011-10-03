
#ifndef MX_H
#define MX_H

#include <assert.h>
#include "cf.h"

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
    static long mul_openmp(const matrix *a, const matrix *b, matrix *r);
private:
    matrix();
    matrix(int height_, int width_) : height(height_), width(width_) {}
private:
    const int height, width;
    number *items;
};

#endif

