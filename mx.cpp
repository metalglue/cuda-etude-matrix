
#include <stdlib.h>
#include <stdio.h>
#include "mx.h"

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

