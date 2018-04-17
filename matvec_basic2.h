#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

//#include <time.h>
//#include <sys/time.h> // for clock_gettime()

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))


typedef struct {
    int nrows, ncols;
    double complex * d; 
} mat;


typedef struct {
    int nrows;
    double complex * d;
} vec;


/* initialize new matrix and set all entries to zero */
void matrix_new(mat **M, int nrows, int ncols);


/* initialize new vector and set all entries to zero */
void vector_new(vec **v, int nrows);


void matrix_delete(mat *M);


void vector_delete(vec *v);


// column major format
void matrix_set_element(mat *M, int row_num, int col_num, double complex val);


double complex matrix_get_element(mat *M, int row_num, int col_num);


void vector_set_element(vec *v, int row_num, double complex val);


double complex vector_get_element(vec *v, int row_num);


/* set matrix elements from array */
void matrix_init_from_array(mat **M, int m, int n, double complex *d);



void matrix_print(mat * M);

void vector_print(vec * v);

/* v(:) = data */
void vector_set_data(vec *v, double complex *data);
 

/* set all vector elems to a constant */
void vector_set_elems_constant(vec *v, double complex scalar);


/* scale vector by a constant */
void vector_scale(vec *v, double complex scalar);
    

/* scale matrix by a constant */
void matrix_scale(mat *M, double complex scalar);


/* s = x + alpha*y */
void vector_add(vec *x, vec *y, double complex alpha, vec *s);

/* S = X + alpha*Y */
void matrix_add(mat *X, mat *Y, double complex alpha, mat *S);


/* copy contents of vec s to d  */
void vector_copy(vec *d, vec *s);


/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S);


/* build transpose of matrix : Mt = M^T */
void matrix_build_transpose(mat *Mt, mat *M);


/* compute euclidean norm of vector */
double vector_get2norm(vec *v);


/* vector min/max */
void vector_get_min_element(vec *v, int *minindex, double *minval); 

void vector_get_max_element(vec *v, int *maxindex, double *maxval);
  
void vector_get_absmax_element(vec *v, int *maxindex, double *maxval);


/* matrix frobenius norm */
double complex get_matrix_frobenius_norm(mat *M);
 

/* matrix max abs val */
double get_matrix_max_abs_element(mat *M);




/* initialize diagonal matrix from vector data */
void initialize_diagonal_matrix(mat *D, vec *data);


/* initialize identity */
void initialize_identity_matrix(mat *D);


/* y = M*x ; column major */
void matrix_vec_mult(mat *M, vec *x, vec **y);

/* y = M^T*x ; column major */
void matrix_transpose_vec_mult(mat *M, vec *x, vec **y);

/* M = x x^T */
void vector_vector_transpose_mult(vec *x, mat **M);


