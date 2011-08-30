
#include <cstdlib>
#include <stdio.h>

typedef double number;

class matrix {
public:
    static matrix *new_random_filled(int height, int width);
    static matrix *new_garbage(int height, int width);
    static void delete_matrix(matrix *m);
    class iter_all {
    public:
        iter_all(const matrix &m) : matrix_(m) { pix = &m.items[0]; }
        iter_all &operator ++(int) { pix++; return *this; }
        number *operator *() { return pix == &matrix_.items[matrix_.height * matrix_.width] ? 0 : pix; }
    private:
        const matrix &matrix_;
        number *pix;
    };
    void size(int *height_, int *width_) { *height_ = height; *width_ = width; }
    void show();
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
    iter_all i(*m);
    for (iter_all i(*m); *i != 0; i++) {
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

void matrix::show()
{
    int height, width;
    size(&height, &width);
    matrix::iter_all i(*this);
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

void
test_0001()
{
    matrix *a = matrix::new_random_filled(2, 3);
    matrix *b = matrix::new_random_filled(3, 2);
    matrix *r = matrix::new_garbage(2, 2);
    a->show(); b->show();
    matrix::delete_matrix(a);
    matrix::delete_matrix(b);
    matrix::delete_matrix(r);
}

int main()
{
    test_0001();
}

