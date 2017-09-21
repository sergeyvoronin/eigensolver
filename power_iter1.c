#include "matvec_basic.h"

// Sergey Voronin, 2017.

void deflation_scheme(mat *A, double tol, int maxiters);
void power_iter(mat *A, double tol, int maxiters, double *eval, vec *evec);	

int main()
{
    int i, j, m, n, maxiters;
    mat *A;
    time_t start_time, end_time;

	// set parameters ---> 
	char *A_file = "data/mat1.txt";
	maxiters = 10000; // num iterations needed depends on spacing between evals
	double tol = 1e-15; // tolerance to break out

	printf("loading matrix A\n");
	matrix_load_from_text_file(A_file, &A);

	// print matrix
	printf("matrix A:\n");
	matrix_print(A);	

	// call deflation scheme
	printf("running deflation scheme..\n");
	deflation_scheme(A, tol, maxiters);

    // delete and exit
    printf("delete and exit..\n");
    matrix_delete(A);

    return 0;
}

void deflation_scheme(mat *A, double tol, int maxiters){
	int i,m,n;
	double eval;
	vec *x0, *evec, *Av, *dvec;
	mat *M;
	m = A->nrows;
	n = A->ncols;

	if(m!=n){
		printf("code currently written for square matrix; feed A^T A\n");
	}

	vector_new(&evec,n);
	vector_new(&dvec,n);
	
	// loop over the eigenvals
	for(i=0; i<min(m,n); i++){
		power_iter(A,tol,maxiters,&eval,evec);	
		printf("eigenpair %d of %d:\n", i+1, min(m,n));
		printf("eval = %f\n", eval);
		printf("evec = \n");
		vector_print(evec);

		matrix_vec_mult(A, evec, &Av);
		vector_copy(dvec, Av);
		vector_add(Av, evec, -eval, dvec);
		printf("norm(dvec) = %f\n", vector_get2norm(dvec)); 

		// spectral deflation 
		vector_vector_transpose_mult(evec,&M);
		
		/* A = A - eval*M */
		matrix_add(A, M, -eval, A);
	}
}


void power_iter(mat *A, double tol, int maxiters, double *eval, vec *evec){
	int iter, maxind, n;
	double maxval;
	vec *x0, *X, *Xn, *AXn, *err;
	n = A->nrows;
	vector_new(&x0,n);
	vector_new(&err,n);
	vector_new(&X,n);
	vector_set_elems_constant(x0, 1.0);
	vector_copy(X,x0);
	for(iter=0; iter <  maxiters; iter++){
		matrix_vec_mult(A,X,&Xn);
		matrix_vec_mult(A,Xn,&AXn);

		vector_get_absmax_element(AXn, &maxind, &maxval);

		*eval = vector_get_element(Xn,maxind);

		vector_scale(Xn,1.0/(*eval));
		
		vector_copy(err, Xn);

		/* err = Xn - X */
		vector_add(err, X, -1, err); 
	
		vector_copy(X,Xn);
	}
	
	vector_copy(evec,X);
	vector_scale(evec,1/vector_get2norm(evec));
}

