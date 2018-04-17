/* basic blas functions, complex support, single threaded */
/* Sergey Voronin, 2017 */

#include "matvec_basic2.h"


/* initialize new matrix and set all entries to zero */
void matrix_new(mat **M, int nrows, int ncols)
{
    *M = malloc(sizeof(mat));
    (*M)->d = (double complex*)calloc(nrows*ncols, sizeof(double complex));
    (*M)->nrows = nrows;
    (*M)->ncols = ncols;
}


/* initialize new vector and set all entries to zero */
void vector_new(vec **v, int nrows)
{
    *v = malloc(sizeof(vec));
    (*v)->d = (double complex*)calloc(nrows,sizeof(double complex));
    (*v)->nrows = nrows;
}


void matrix_delete(mat *M)
{
    free(M->d);
    free(M);
}


void vector_delete(vec *v)
{
    free(v->d);
    free(v);
}


// column major format
void matrix_set_element(mat *M, int row_num, int col_num, double complex val){
    M->d[col_num*(M->nrows) + row_num] = val;
}

double complex matrix_get_element(mat *M, int row_num, int col_num){
    return M->d[col_num*(M->nrows) + row_num];
}


void vector_set_element(vec *v, int row_num, double complex val){
    v->d[row_num] = val;
}


double complex vector_get_element(vec *v, int row_num){
    return v->d[row_num];
}


/* set matrix elements from array */
void matrix_init_from_array(mat **M, int m, int n, double complex *d){
	int i,j;
    printf("initializing M of size %d by %d\n", m, n);
    matrix_new(M,m,n);
    printf("done..\n");

    // set nnzs in col major order
    for(j=0; j<n; j++){
    	for(i=0; i<m; i++){
            matrix_set_element(*M,i,j, d[j*m + i]);
        }
    }
}


void matrix_print(mat * M){
    int i,j;
    double complex val;
    for(i=0; i<M->nrows; i++){
        for(j=0; j<M->ncols; j++){
            val = matrix_get_element(M, i, j);
            //printf("%f  ", val);
            printf("%.2f %+.2fi  ", creal(val), cimag(val));
        }
        printf("\n");
    }
}


void vector_print(vec * v){
    int i;
    double complex val;
    for(i=0; i<v->nrows; i++){
        val = vector_get_element(v, i);
        printf("%.2f %+.2fi\n", creal(val), cimag(val));
    }
}


/* v(:) = data */
void vector_set_data(vec *v, double complex *data){
    int i;
    for(i=0; i<(v->nrows); i++){
        v->d[i] = data[i];
    }
}


/* set all vector elems to a constant */
void vector_set_elems_constant(vec *v, double complex scalar){
    int i;
    for(i=0; i<(v->nrows); i++){
        v->d[i] = scalar;
    }
}


/* scale vector by a constant */
void vector_scale(vec *v, double complex scalar){
    int i;
    for(i=0; i<(v->nrows); i++){
        v->d[i] = scalar*(v->d[i]);
    }
}


/* scale matrix by a constant */
void matrix_scale(mat *M, double complex scalar){
    int i;
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        M->d[i] = scalar*(M->d[i]);
    }
}


/* s = x + alpha*y */
void vector_add(vec *x, vec *y, double complex alpha, vec *s){
	int i, n;
	n = x->nrows;
	#pragma omp parallel for private(i)
	for(i=0; i<n; i++){
		vector_set_element(s,i,vector_get_element(x,i) + alpha*vector_get_element(y,i));
	}
}


/* S = X + alpha*Y */
void matrix_add(mat *X, mat *Y, double complex alpha, mat *S){
	int i,j,m,n;
	m = X->nrows;
	n = X->ncols;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			matrix_set_element(S,i,j,matrix_get_element(X,i,j) + alpha*matrix_get_element(Y,i,j));
		}
	}
}



/* copy contents of vec s to d  */
void vector_copy(vec *d, vec *s){
    int i;
    for(i=0; i<(s->nrows); i++){
        d->d[i] = s->d[i];
    }
}


/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S){
    int i;
    for(i=0; i<((S->nrows)*(S->ncols)); i++){
        D->d[i] = S->d[i];
    }
}




/* build transpose of matrix : Mt = M^T */
void matrix_build_transpose(mat *Mt, mat *M){
    int i,j;
    for(i=0; i<(M->nrows); i++){
        for(j=0; j<(M->ncols); j++){
            matrix_set_element(Mt,j,i,matrix_get_element(M,i,j)); 
        }
    }
}


/* compute euclidean norm of vector */
double vector_get2norm(vec *v){
    int i;
    double val, normval = 0;
    for(i=0; i<(v->nrows); i++){
        val = cabs(v->d[i]);
        normval += val*val;
    }
    return sqrt(normval);
}


void vector_get_min_element(vec *v, int *minindex, double *minval){
    int i;
    double val;
    *minindex = 0;
    *minval = cabs(v->d[0]);
    for(i=0; i<(v->nrows); i++){
        val = cabs(v->d[i]);
        if(val < *minval){
            *minval = val;
            *minindex = i;
        }
    }
}


void vector_get_max_element(vec *v, int *maxindex, double *maxval){
    int i;
    double val;
    *maxindex = 0;
    *maxval = cabs(v->d[0]);
    for(i=0; i<(v->nrows); i++){
        val = cabs(v->d[i]);
        if(val > *maxval){
            *maxval = val;
            *maxindex = i;
        }
    }
}


void vector_get_absmax_element(vec *v, int *maxindex, double *maxval){
    int i;
    double val;
    *maxindex = 0;
    *maxval = cabs(v->d[0]);
    for(i=0; i<(v->nrows); i++){
        val = cabs(v->d[i]);
        if(val > *maxval){
            *maxval = val;
            *maxindex = i;
        }
    }
}



/* matrix frobenius norm */
double complex get_matrix_frobenius_norm(mat *M){
    int i;
    double complex val, normval = 0;
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        val = M->d[i];
        normval += val*val;
    }
    return csqrt(normval);
}


/* matrix max abs val */
double get_matrix_max_abs_element(mat *M){
    int i;
    double val, max = 0;
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        val = cabs(M->d[i]);
        if( val > max )
            max = val;
    }
    return max;
}



/* Multiplies matrix M by vector x; returns resulting vector y 
 * no OpenMP version */
void matrix_vec_mult(mat *M, vec *x, vec **y){
    int i,j;
    double complex val;
    vector_new(y,M->nrows);
    for (i = 0; i < M->nrows; i++){   
        for (j = 0; j < M->ncols; j++){
            val = matrix_get_element(M,i,j);
            vector_set_element(*y,i,vector_get_element(*y,i) + val*vector_get_element(x,j));
        }
    }
}


/* Multiplies matrix transpose M by vector x; returns resulting vector y */
/* y_i = sum Aji xj */
void matrix_transpose_vec_mult(mat *M, vec *x, vec **y)
{
    int i,j;
    double complex val;
    vector_new(y,M->ncols);
    for (i = 0; i < M->ncols; i++){   
        for (j = 0; j < M->nrows; j++){
            val = matrix_get_element(M,j,i);
            vector_set_element(*y,i,vector_get_element(*y,i) + val*vector_get_element(x,j));
        }
    }
}


/* vector outer product: M = x x^t ; M_ij = xi*xj */
void vector_vector_transpose_mult(vec *x, mat **M){
	int i,j;
	matrix_new(M,x->nrows,x->nrows);	
	for(i=0; i<(x->nrows); i++){
		for(j=0; j<(x->nrows); j++){
			matrix_set_element(*M,i,j,vector_get_element(x,i)*vector_get_element(x,j));
		}
	}
}


