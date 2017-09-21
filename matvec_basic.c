/* basic OpenMP accelerated blas functions */
/* Sergey Voronin, 2017 */

#include "matvec_basic.h"


/* timing function */
double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval){
    long secs_used, micros_used;
    secs_used= end_timeval.tv_sec - start_timeval.tv_sec;
    micros_used= end_timeval.tv_usec - start_timeval.tv_usec;
	
    return (double)(secs_used + micros_used/1000000.0); 
}


/* initialize new matrix and set all entries to zero */
void matrix_new(mat **M, int nrows, int ncols)
{
    *M = malloc(sizeof(mat));
    (*M)->d = (double*)calloc(nrows*ncols, sizeof(double));
    (*M)->nrows = nrows;
    (*M)->ncols = ncols;
}


/* initialize new vector and set all entries to zero */
void vector_new(vec **v, int nrows)
{
    *v = malloc(sizeof(vec));
    (*v)->d = (double*)calloc(nrows,sizeof(double));
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
void matrix_set_element(mat *M, int row_num, int col_num, double val){
    M->d[col_num*(M->nrows) + row_num] = val;
}

double matrix_get_element(mat *M, int row_num, int col_num){
    return M->d[col_num*(M->nrows) + row_num];
}


void vector_set_element(vec *v, int row_num, double val){
    v->d[row_num] = val;
}


double vector_get_element(vec *v, int row_num){
    return v->d[row_num];
}


/* set matrix elements from array */
void matrix_init_from_array(mat **M, int m, int n, double *d){
	int i,j;
    printf("initializing M of size %d by %d\n", m, n);
    matrix_new(M,m,n);
    printf("done..\n");

    // read and set elements
    for(j=0; j<n; j++){
    	for(i=0; i<m; i++){
            matrix_set_element(*M,i,j, d[j*m + i]);
        }
    }
}



/* set matrix elements from text file 
 * the nonzeros are in column major order 
*/
void matrix_load_from_text_file(char *fname, mat **M){
	int i,m,n;
	double val;
	FILE *fp;
	fp = fopen(fname,"r");
	fscanf(fp,"%d\n",&m);
	fscanf(fp,"%d\n",&n);
	printf("setting up matrix with m = %d and n = %d\n", m, n);
	matrix_new(M,m,n);
	for(i=0; i<(m*n); i++){
		fscanf(fp,"%lf\n",&val);
		(*M)->d[i] = val;
	}
	fclose(fp);
}



/* load matrix from binary file 
 * the nonzeros are in column major order 
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
void matrix_load_from_binary_file(char *fname, mat **M){
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;

    fp = fopen(fname,"r");
    fread(&num_rows,sizeof(int),one,fp); //read m
    fread(&num_columns,sizeof(int),one,fp); //read n
    printf("initializing M of size %d by %d\n", num_rows, num_columns);
    matrix_new(M,num_rows,num_columns);
    printf("done..\n");

    // read and set elements
    for(j=0; j<num_columns; j++){
    	for(i=0; i<num_rows; i++){
            fread(&nnz_val,sizeof(double),one,fp); //read nnz
            matrix_set_element(*M,i,j,nnz_val);
        }
    }
    fclose(fp);
}
 


/* write matrix to binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
void matrix_write_to_binary_file(mat *M, char *fname){
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    num_rows = M->nrows; num_columns = M->ncols;
    
    fp = fopen(fname,"w");
    fwrite(&num_rows,sizeof(int),one,fp); //write m
    fwrite(&num_columns,sizeof(int),one,fp); //write n

    // write the elements
    for(j=0; j<num_columns; j++){
    	for(i=0; i<num_rows; i++){
            nnz_val = matrix_get_element(M,i,j);
            fwrite(&nnz_val,sizeof(double),one,fp); //write nnz
        }
    }
    fclose(fp);
}



/* column vector load from binary file 
 * the nonzeros are in order of a loop over all rows  
format:
num_rows (int) 
nnz (double)
...
nnz (double)
*/
void vector_load_from_binary_file(char *fname, vec **v){
    int i, j, num_rows, row_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;

    fp = fopen(fname,"r");
    fread(&num_rows,sizeof(int),one,fp); //read num_rows
    printf("initializing v of length %d\n", num_rows);
    vector_new(v,num_rows);
    printf("done..\n");

    // read and set elements
	for(i=0; i<num_rows; i++){
		fread(&nnz_val,sizeof(double),one,fp); //read nnz
		vector_set_element(*v,i,nnz_val);
	}
    fclose(fp);
}
 


void matrix_print(mat * M){
    int i,j;
    double val;
    for(i=0; i<M->nrows; i++){
        for(j=0; j<M->ncols; j++){
            val = matrix_get_element(M, i, j);
            printf("%f  ", val);
        }
        printf("\n");
    }
}


void vector_print(vec * v){
    int i;
    double val;
    for(i=0; i<v->nrows; i++){
        val = vector_get_element(v, i);
        printf("%f\n", val);
    }
}


/* v(:) = data */
void vector_set_data(vec *v, double *data){
    int i;
    #pragma omp parallel shared(v) private(i) 
    {
    #pragma omp for
    for(i=0; i<(v->nrows); i++){
        v->d[i] = data[i];
    }
    }
}


/* set all vector elems to a constant */
void vector_set_elems_constant(vec *v, double scalar){
    int i;
    #pragma omp parallel shared(v,scalar) private(i) 
    {
    #pragma omp for
    for(i=0; i<(v->nrows); i++){
        v->d[i] = scalar;
    }
    }
}


/* scale vector by a constant */
void vector_scale(vec *v, double scalar){
    int i;
    #pragma omp parallel shared(v,scalar) private(i) 
    {
    #pragma omp for
    for(i=0; i<(v->nrows); i++){
        v->d[i] = scalar*(v->d[i]);
    }
    }
}


/* scale matrix by a constant */
void matrix_scale(mat *M, double scalar){
    int i;
    #pragma omp parallel shared(M,scalar) private(i) 
    {
    #pragma omp for
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        M->d[i] = scalar*(M->d[i]);
    }
    }
}


/* v = S_tau(v) */
void vector_soft_threshold(vec *v, double tau){
	int i,n;
	double val;
	n = v->nrows;
	#pragma omp parallel for default(shared) private(i,val)
	for(i=0; i<n; i++){
		val = vector_get_element(v,i);
		if(val > tau){
			vector_set_element(v,i,val - tau);
		}
		else if(val < -tau){
			vector_set_element(v,i,val + tau);
		}
		else{
			vector_set_element(v,i,0);
		}
	}
}


/* s = x + alpha*y */
void vector_add(vec *x, vec *y, double alpha, vec *s){
	int i, n;
	n = x->nrows;
	#pragma omp parallel for private(i)
	for(i=0; i<n; i++){
		vector_set_element(s,i,vector_get_element(x,i) + alpha*vector_get_element(y,i));
	}
}


/* S = X + alpha*Y */
void matrix_add(mat *X, mat *Y, double alpha, mat *S){
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
    #pragma omp parallel shared(d,s) private(i) 
    {
    #pragma omp for 
    for(i=0; i<(s->nrows); i++){
        d->d[i] = s->d[i];
    }
    }
}


/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S){
    int i;
    #pragma omp parallel shared(D,S) private(i) 
    {
    #pragma omp for 
    for(i=0; i<((S->nrows)*(S->ncols)); i++){
        D->d[i] = S->d[i];
    }
    }
}



/* hard threshold matrix entries  */
void matrix_hard_threshold(mat *M, double TOL){
    int i;
    #pragma omp parallel shared(M) private(i) 
    {
    #pragma omp for 
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        if(fabs(M->d[i]) < TOL){
            M->d[i] = 0;
        }
    }
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



/* subtract b from a and save result in a  */
void vector_sub(vec *a, vec *b){
    int i;
    //#pragma omp parallel for
    #pragma omp parallel shared(a,b) private(i) 
    {
    #pragma omp for 
    for(i=0; i<(a->nrows); i++){
        a->d[i] = a->d[i] - b->d[i];
    }
    }
}


/* subtract B from A and save result in A  */
void matrix_sub(mat *A, mat *B){
    int i;
    #pragma omp parallel shared(A,B) private(i) 
    {
    #pragma omp for 
    for(i=0; i<((A->nrows)*(A->ncols)); i++){
        A->d[i] = A->d[i] - B->d[i];
    }
    }
}


/* A = A - u*v where u is a column vec and v is a row vec */
void matrix_sub_column_times_row_vector(mat *A, vec *u, vec *v){
    int i,j;
    #pragma omp parallel for shared(A,u,v) private(j)
    for(i=0; i<(A->nrows); i++){
        for(j=0; j<(A->ncols); j++){
            matrix_set_element(A,i,j,matrix_get_element(A,i,j) - vector_get_element(u,i)*vector_get_element(v,j));
        }
    }
}


/* compute euclidean norm of vector */
double vector_get2norm(vec *v){
    int i;
    double val, normval = 0;
    #pragma omp parallel shared(v,normval) private(i,val) 
    {
    #pragma omp for reduction(+:normval)
    for(i=0; i<(v->nrows); i++){
        val = v->d[i];
        normval += val*val;
    }
    }
    return sqrt(normval);
}


void vector_get_min_element(vec *v, int *minindex, double *minval){
    int i, val;
    *minindex = 0;
    *minval = v->d[0];
    for(i=0; i<(v->nrows); i++){
        val = v->d[i];
        if(val < *minval){
            *minval = val;
            *minindex = i;
        }
    }
}


void vector_get_max_element(vec *v, int *maxindex, double *maxval){
    int i, val;
    *maxindex = 0;
    *maxval = v->d[0];
    for(i=0; i<(v->nrows); i++){
        val = v->d[i];
        if(val > *maxval){
            *maxval = val;
            *maxindex = i;
        }
    }
}



void vector_get_absmax_element(vec *v, int *maxindex, double *maxval){
    int i, val;
    *maxindex = 0;
    *maxval = fabs(v->d[0]);
    for(i=0; i<(v->nrows); i++){
        val = fabs(v->d[i]);
        if(val > *maxval){
            *maxval = val;
            *maxindex = i;
        }
    }
}



/* returns the dot product of two vectors */
double vector_dot_product(vec *u, vec *v){
    int i;
    double dotval = 0;
    #pragma omp parallel shared(u,v,dotval) private(i) 
    {
    #pragma omp for reduction(+:dotval)
    for(i=0; i<u->nrows; i++){
        dotval += (u->d[i])*(v->d[i]);
    }
    }
    return dotval;
}



/* matrix frobenius norm */
double get_matrix_frobenius_norm(mat *M){
    int i;
    double val, normval = 0;
    #pragma omp parallel shared(M,normval) private(i,val) 
    {
    #pragma omp for reduction(+:normval)
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        val = M->d[i];
        normval += val*val;
    }
    }
    return sqrt(normval);
}


/* matrix max abs val */
double get_matrix_max_abs_element(mat *M){
    int i;
    double val, max = 0;
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        val = fabs(M->d[i]);
        if( val > max )
            max = val;
    }
    return max;
}



/* calculate percent error between A and B 
in terms of Frobenius norm: 100*norm(A - B)/norm(A) */
double get_percent_error_between_two_mats(mat *A, mat *B){
    int m,n;
    double normA, normB, normA_minus_B;
    mat *A_minus_B;
    m = A->nrows;
    n = A->ncols;
    matrix_new(&A_minus_B,m,n);
    matrix_copy(A_minus_B, A);
    matrix_sub(A_minus_B, B);
    normA = get_matrix_frobenius_norm(A);
    normB = get_matrix_frobenius_norm(B);
    normA_minus_B = get_matrix_frobenius_norm(A_minus_B);
    matrix_delete(A_minus_B);
    return 100.0*normA_minus_B/normA;
}



/* initialize diagonal matrix from vector data */
void initialize_diagonal_matrix(mat *D, vec *data){
    int i;
    #pragma omp parallel shared(D) private(i)
    { 
    #pragma omp parallel for
    for(i=0; i<(D->nrows); i++){
        matrix_set_element(D,i,i,data->d[i]);
    }
    }
}



/* initialize identity */
void initialize_identity_matrix(mat *D){
    int i;
    matrix_scale(D, 0);
    #pragma omp parallel shared(D) private(i)
    { 
    #pragma omp parallel for
    for(i=0; i<(D->nrows); i++){
        matrix_set_element(D,i,i,1.0);
    }
    }
}


/* Multiplies matrix M by vector x; returns resulting vector y 
 * OpenMP version */
void matrix_vec_mult(mat *M, vec *x, vec **y){
    int i,j;
    double val;
    vector_new(y,M->nrows);
    #pragma omp parallel for private(j,val) shared(M,x,y)
    for (i = 0; i < M->nrows; i++){   
        //y[i] = 0; 
        //vector_set_element(y,i, 0);
        for (j = 0; j < M->ncols; j++){
            val = matrix_get_element(M,i,j);
            //y[i] += M->v[i][j] * x[j];
            vector_set_element(*y,i,vector_get_element(*y,i) + val*vector_get_element(x,j));
        }
    }
}


/* Multiplies matrix transpose M by vector x; returns resulting vector y */
/* y_i = sum Aji xj */
void matrix_transpose_vec_mult(mat *M, vec *x, vec **y)
{
    int i,j;
    double val;
    vector_new(y,M->ncols);
    // must declare j to be private! otherwise all threads will have access to j
    // j should be private for each thread
    #pragma omp parallel for private(j,val)
    for (i = 0; i < M->ncols; i++){   
        for (j = 0; j < M->nrows; j++){
            val = matrix_get_element(M,j,i);
            vector_set_element(*y,i,vector_get_element(*y,i) + val*vector_get_element(x,j));
        }
    }
}


/* M = x x^t ; M_ij = xi*xj */
void vector_vector_transpose_mult(vec *x, mat **M){
	int i,j;
	matrix_new(M,x->nrows,x->nrows);	
	for(i=0; i<(x->nrows); i++){
		for(j=0; j<(x->nrows); j++){
			matrix_set_element(*M,i,j,vector_get_element(x,i)*vector_get_element(x,j));
		}
	}
}



/* set column of matrix to vector */
void matrix_set_col(mat *M, int j, vec *column_vec){
    int i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp for
    for(i=0; i<M->nrows; i++){
        matrix_set_element(M,i,j,vector_get_element(column_vec,i));
    }
    }
}


/* extract column of a matrix into a vector */
void matrix_get_col(mat *M, int j, vec *column_vec){
    int i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp parallel for
    for(i=0; i<M->nrows; i++){ 
        vector_set_element(column_vec,i,matrix_get_element(M,i,j));
    }
    }
}


/* extract row i of a matrix into a vector */
void matrix_get_row(mat *M, int i, vec *row_vec){
    int j;
    #pragma omp parallel shared(row_vec,M,i) private(j) 
    {
    #pragma omp parallel for
    for(j=0; j<M->ncols; j++){ 
        vector_set_element(row_vec,j,matrix_get_element(M,i,j));
    }
    }
}


/* put vector row_vec as row i of a matrix */
void matrix_set_row(mat *M, int i, vec *row_vec){
    int j;
    #pragma omp parallel shared(row_vec,M,i) private(j) 
    {
    #pragma omp parallel for
    for(j=0; j<M->ncols; j++){ 
        matrix_set_element(M,i,j,vector_get_element(row_vec,j));
    }
    }
}


