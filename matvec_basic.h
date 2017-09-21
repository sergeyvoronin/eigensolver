#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include <time.h>
#include <sys/time.h> // for clock_gettime()

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))


typedef struct {
    int nrows, ncols;
    double * d; 
} mat;


typedef struct {
    int nrows;
    double * d;
} vec;



/* timing function */
double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval);


/* initialize new matrix and set all entries to zero */
void matrix_new(mat **M, int nrows, int ncols);


/* initialize new vector and set all entries to zero */
void vector_new(vec **v, int nrows);


void matrix_delete(mat *M);


void vector_delete(vec *v);


// column major format
void matrix_set_element(mat *M, int row_num, int col_num, double val);


double matrix_get_element(mat *M, int row_num, int col_num);


void vector_set_element(vec *v, int row_num, double val);


double vector_get_element(vec *v, int row_num);


/* set matrix elements from array */
void matrix_init_from_array(mat **M, int m, int n, double *d);

/* matrix from text file 
 * the nonzeros are in a loop over columns and rows (column major)
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
void matrix_load_from_text_file(char *fname, mat **M);


void matrix_load_from_binary_file(char *fname, mat **M);


void vector_load_from_binary_file(char *fname, vec **v);


void matrix_print(mat * M);

void vector_print(vec * v);

/* v(:) = data */
void vector_set_data(vec *v, double *data);
 

/* set all vector elems to a constant */
void vector_set_elems_constant(vec *v, double scalar);


/* scale vector by a constant */
void vector_scale(vec *v, double scalar);
    

/* scale matrix by a constant */
void matrix_scale(mat *M, double scalar);

/* v = S_tau(v) */
void vector_soft_threshold(vec *v, double tau);

/* s = x + alpha*y */
void vector_add(vec *x, vec *y, double alpha, vec *s);

/* S = X + alpha*Y */
void matrix_add(mat *X, mat *Y, double alpha, mat *S);


/* copy contents of vec s to d  */
void vector_copy(vec *d, vec *s);


/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S);


/* hard threshold matrix entries  */
void matrix_hard_threshold(mat *M, double TOL);


/* build transpose of matrix : Mt = M^T */
void matrix_build_transpose(mat *Mt, mat *M);



/* subtract b from a and save result in a  */
void vector_sub(vec *a, vec *b);


/* subtract B from A and save result in A  */
void matrix_sub(mat *A, mat *B);


/* A = A - u*v where u is a column vec and v is a row vec */
void matrix_sub_column_times_row_vector(mat *A, vec *u, vec *v);


/* compute euclidean norm of vector */
double vector_get2norm(vec *v);


/* vector min/max */
void vector_get_min_element(vec *v, int *minindex, double *minval); 

void vector_get_max_element(vec *v, int *maxindex, double *maxval);
  
void vector_get_absmax_element(vec *v, int *maxindex, double *maxval);


/* returns the dot product of two vectors */
double vector_dot_product(vec *u, vec *v); 


/* matrix frobenius norm */
double get_matrix_frobenius_norm(mat *M);
 

/* matrix max abs val */
double get_matrix_max_abs_element(mat *M);



/* calculate percent error between A and B 
in terms of Frobenius norm: 100*norm(A - B)/norm(A) */
double get_percent_error_between_two_mats(mat *A, mat *B);


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


/* set column of matrix to vector */
void matrix_set_col(mat *M, int j, vec *column_vec);


/* extract column of a matrix into a vector */
void matrix_get_col(mat *M, int j, vec *column_vec);


/* extract row i of a matrix into a vector */
void matrix_get_row(mat *M, int i, vec *row_vec);


/* put vector row_vec as row i of a matrix */
void matrix_set_row(mat *M, int i, vec *row_vec);

