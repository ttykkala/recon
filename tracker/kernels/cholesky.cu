#include <math.h>
#include <assert.h>
#include <helper_cuda.h>


/*****************************************************************
*  Inversion of a symmetric matrix by Cholesky decomposition.    *
*  The matrix must be positive definite.                         * 
* -------------------------------------------------------------- *
* REFERENCE:                                                     *
*             From a Java Library Created by Vadim Kutsyy,       *
*             "http://www.kutsyy.com".                           *
* -------------------------------------------------------------- * 
* SAMPLE RUN:                                                    *
*                                                                *
* Inversion of a square real symetric matrix by Cholevsky method *
* (The matrix must positive definite).                           *
*                                                                *
* Size = 4                                                       *
*                                                                *
* Determinant = 432.000000                                       *
*                                                                *
* Matrix A:                                                      *
* 5.000000 -1.000000 -1.000000 -1.000000                         *
* -1.000000 5.000000 -1.000000 -1.000000                         *
* -1.000000 -1.000000 5.000000 -1.000000                         *
* -1.000000 -1.000000 -1.000000 5.000000                         *
*                                                                *
* Matrix Inv(A):                                                 *
* 0.250000 0.083333 0.083333 0.083333                            *
* 0.083333 0.250000 0.083333 0.083333                            *
* 0.083333 0.083333 0.250000 0.083333                            *
* 0.083333 0.083333 0.083333 0.250000                            *
*                                                                *
*                      C++ Release By Jean-Pierre Moreau, Paris. *
* -------------------------------------------------------------- *
* Release 1.1 : added verification Inv(A) * A = I.               *
*****************************************************************/

/* ----------------------------------------------------
        main method for Cholesky decomposition.

        input         n  size of matrix
        input/output  a  Symmetric positive def. matrix
        output        p  vector of resulting diag of a
        author:       <Vadum Kutsyy, kutsyy@hotmail.com>
   ----------------------------------------------------- */
        __device__ void choldc1(int n, CHOLMAT &a, CHOLVEC &p) {
          int i,j,k;
          doublereal sum;
          // X X X X X X
          // 0 X X X X X
          // 0 0 X X X X
          // 0 0 0 X X X
          // 0 0 0 0 X X
          // 0 0 0 0 0 X
	  for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
              sum = a[i][j];
              for (k = i - 1; k >= 0; k--) {
                sum -= a[i][k] * a[j][k];
	      }
              if (i == j) {
                if (sum <= 0) {
                  //printf(" a is not positive definite!\n");
		}
                p[i] = sqrt(sum);
	      }
              else {
                a[j][i] = sum / p[i];
	      }
	    }
	  }
	}
        /* ----------------------------------------------------
                main method for Cholesky decomposition for one element.

                input         n  size of matrix
                input/output  a  Symmetric positive def. matrix
                output        p  vector of resulting diag of a
                author:       <Vadum Kutsyy, kutsyy@hotmail.com>
           ----------------------------------------------------- */
        __device__ void choldc1Element(int i, int j, CHOLMAT &a, CHOLVEC &p) {
            int k;
            doublereal sum = a[i][j];
            for (k = i - 1; k >= 0; k--) {
                sum -= a[i][k] * a[j][k];
            }
            if (i == j) {
                if (sum <= 0) {
                    //printf(" a is not positive definite!\n");
                }
                p[i] = sqrt(sum);
            }
            else {
                a[j][i] = sum / p[i];
            }
        }
        /* -----------------------------------------------
        Cholesky decomposition.

        input    n  size of matrix
        input    A  Symmetric positive def. matrix
        output   a  lower deomposed matrix
        uses        choldc1(int,MAT,VEC)
   ----------------------------------------------- */
        __device__ void choldc(int n,CHOLMAT &A, CHOLMAT &a) {
	  int i,j;
      CHOLVEC p;
          for (i = 0; i < n; i++) 
	    for (j = 0; j < n; j++) 
	      a[i][j] = A[i][j];
	  choldc1(n, a, p);
          for (i = 0; i < n; i++) {
            a[i][i] = p[i];
            for (j = i + 1; j < n; j++) {
              a[i][j] = 0;
	    }
	  }
	}
 
/* -----------------------------------------------------
         Inverse of Cholesky decomposition.

         input    n  size of matrix
         input    A  Symmetric positive def. matrix
         output   a  inverse of lower deomposed matrix
         uses        choldc1(int,MAT,CHOLVEC)
   ----------------------------------------------------- */
        __device__ void choldcsl(int n, CHOLMAT &A, CHOLMAT &a) {
      int i,j,k; doublereal sum;
      CHOLVEC p;
          for (i = 0; i < n; i++) 
	    for (j = 0; j < n; j++) 
	      a[i][j] = A[i][j];
          choldc1(n, a, p);
          for (i = 0; i < n; i++) {
            a[i][i] = 1 / p[i];
            for (j = i + 1; j < n; j++) {
              sum = 0;
              for (k = i; k < j; k++) {
                sum -= a[j][k] * a[k][i];
	      }
              a[j][i] = sum / p[j];
	    }
	  }
        }
        /* -----------------------------------------------------
                 Inverse of Cholesky decomposition (fast).

                 input    n  size of matrix
                 input    A  Symmetric positive def. matrix
                 input    p  vector of diag of a
                 output   a  inverse of lower decomposed matrix
                 uses        choldc1(int,MAT,VEC)
                 note! replaces A by decomposition
           ----------------------------------------------------- */
        __device__ void choldcsl2(int n, CHOLMAT &a, CHOLVEC &p) {
            int i,j,k; doublereal sum;
            for (i = 0; i < n; i++) {
                a[i][i] = 1 / p[i];
                for (j = i + 1; j < n; j++) {
                    sum = 0;
                    for (k = i; k < j; k++) {
                        sum -= a[j][k] * a[k][i];
                    }
                    a[j][i] = sum / p[j];
                }
            }
        }


/* -----------------------------------------------------------------------------
        Computation of Determinant of the matrix using Cholesky decomposition

        input    n  size of matrix
        input    a  Symmetric positive def. matrix
        return      det(a)
        uses        choldc(int,MAT,MAT)
   ------------------------------------------------------------------------------ */
        __device__ doublereal choldet(int n, CHOLMAT &a) {
           CHOLMAT c; doublereal d=1; int i;
           choldc(n,a,c);
           for (i = 0; i < n; i++)  d *= c[i][i];
           return d * d;
	}
 
/* ---------------------------------------------------
        Matrix inverse using Cholesky decomposition

        input    n  size of matrix
        input	  A  Symmetric positive def. matrix
        output   a  inverse of A
        uses        choldc1(MAT, VEC)
   --------------------------------------------------- */
        __device__ void cholsl(int n, CHOLMAT &A, CHOLMAT &a) {
	  int i,j,k;
          choldcsl(n,A,a);
          for (i = 0; i < n; i++) {
            for (j = i + 1; j < n; j++) {
              a[i][j] = 0.0;
	    }
	  }
          for (i = 0; i < n; i++) {
            a[i][i] *= a[i][i];
            for (k = i + 1; k < n; k++) {
              a[i][i] += a[k][i] * a[k][i];
	    }
            for (j = i + 1; j < n; j++) {
              for (k = j; k < n; k++) {
                a[i][j] += a[k][i] * a[k][j];
	      }
	    }
	  }
          for (i = 0; i < n; i++) {
            for (j = 0; j < i; j++) {
              a[i][j] = a[j][i];
	    }
	  }
	}
        __device__ void choleskyInverse(int n, CHOLMAT &a) {
            int i,j,k;
            for (i = 0; i < n; i++) {
              for (j = i + 1; j < n; j++) {
                a[i][j] = 0.0;
              }
            }
            for (i = 0; i < n; i++) {
              a[i][i] *= a[i][i];
              for (k = i + 1; k < n; k++) {
                a[i][i] += a[k][i] * a[k][i];
              }
              for (j = i + 1; j < n; j++) {
                for (k = j; k < n; k++) {
                  a[i][j] += a[k][i] * a[k][j];
                }
              }
            }
            for (i = 0; i < n; i++) {
              for (j = 0; j < i; j++) {
                a[i][j] = a[j][i];
              }
            }
        }

