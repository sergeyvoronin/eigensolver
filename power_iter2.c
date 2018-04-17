#include "matvec_basic2.h"

void power_iter(mat *A, double tol, int maxiters, double *eval, vec *evec);	

int main()
{
    int i, j, m, n, maxiters;
    mat *A;
    vec *evec;
    double eval;

	// set parameters ---> 
	double complex Ad[9] = {2,1,3,1,1,2+I,3,2-I,2};
	double tol = 1e-8;
	maxiters = 100;
	 
	printf("loading A and b\n");
	m = 3; n = 3;
	matrix_new(&A,m,n);
	matrix_init_from_array(&A, m, n, Ad);

	// print matrix
	printf("matrix A:\n");
	matrix_print(A);	

	// call deflation scheme
	printf("running power iteration..\n");
	vector_new(&evec,n);
	power_iter(A,tol,maxiters,&eval,evec);	

    printf("eval = %f\n", eval);
    printf("evec = \n");
    vector_print(evec);

    // delete and exit
    printf("delete and exit..\n");
    matrix_delete(A);

    return 0;
}


void power_iter(mat *A, double tol, int maxiters, double *eval, vec *evec){
	int i,iter, maxind, n;
	double maxval;
	vec *x0, *X, *Xn, *AXn, *err;
	n = A->nrows;
	vector_new(&x0,n);
	vector_new(&err,n);
	vector_new(&X,n);
	vector_new(&Xn,n);
	vector_new(&AXn,n);
	vector_set_elems_constant(x0, 1.0);
    vector_scale(x0,1/vector_get2norm(x0));
	vector_copy(X,x0);
	for(iter=0; iter <  maxiters; iter++){
		matrix_vec_mult(A,X,&Xn);
        vector_scale(Xn,1/vector_get2norm(Xn));
        matrix_vec_mult(A,Xn,&AXn);
        vector_copy(X,Xn);
        *eval= 0;
        for(i=0; i<n; i++){
            *eval += conj(vector_get_element(Xn,i))*vector_get_element(AXn,i);
        }
        printf("eval = %f\n",*eval);
	}
	
	vector_copy(evec,Xn);
	vector_scale(evec,1/vector_get2norm(evec));
}

