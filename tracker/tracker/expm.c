/*
Copyright 2016 Tommi M. Tykk‰l‰

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <assert.h>
#include <f2c.h>
#include <stdio.h>
#include <math.h>
///////////////////////////////////////////
//F77 routines
///////////////////////////////////////////
static int s_stop(char *s, ftnlen n);
static void d_cnjg(doublecomplex *r, doublecomplex *z);
static integer pow_ii(integer *ap, integer *bp);
static void z_div(doublecomplex *c, doublecomplex *a, doublecomplex *b);
static doublereal f__cabs(doublereal real, doublereal imag);
static doublereal z_abs(doublecomplex *z);
//lapack
static int zgefa_(doublecomplex *a, integer *lda, integer *n, integer *ipvt, integer *info);
static int zgesl_(doublecomplex *a, integer *lda, integer *n, integer *ipvt, doublecomplex *b, integer *job);
static int zgesv_(integer *n, integer *m, doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *b, integer *ldb, integer *iflag);
//blas
static int xerbla2_(char *srname, integer *info, ftnlen srname_len);
static long lsame_(char *ca, char *cb, ftnlen ca_len, ftnlen cb_len);
static int zgemm_(char *transa, char *transb, integer *m, integer *
    n, integer *k, doublecomplex *alpha, doublecomplex *a, integer *lda,
    doublecomplex *b, integer *ldb, doublecomplex *beta, doublecomplex *
    c__, integer *ldc, ftnlen transa_len, ftnlen transb_len);
static int zscal_(integer *n, doublecomplex *za, doublecomplex *zx, integer *incx);
static doublereal dcabs1_(doublecomplex *z__);
static integer izamax_(integer *n, doublecomplex *zx, integer *incx);
static int zaxpy_(integer *n, doublecomplex *za, doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy);
static void zdotc_(doublecomplex * ret_val, integer *n, doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy);
static int zdscal2_(integer *n, doublereal *da, doublecomplex *zx, integer *incx);

/*
 int s_stop(char *s, ftnlen n) {
	//printf("%s\n",s);
    return 0;
}
*/

static void d_cnjg(doublecomplex *r, doublecomplex *z) {
	doublereal zi = z->i;
	r->r = z->r;
	r->i = -zi;
}

static integer pow_ii(integer *ap, integer *bp)
{
	integer pow, x, n;
	unsigned long u;

	x = *ap;
	n = *bp;

	if (n <= 0) {
		if (n == 0 || x == 1)
			return 1;
		if (x != -1)
			/*			return x == 0 ? 1/x : 0; */
			return x == 0 ? 0 : 1/x;	/* rbd */
		n = -n;
	}
	u = n;
	for(pow = 1; ; )
	{
		if(u & 01)
			pow *= x;
		if(u >>= 1)
			x *= x;
		else
			break;
	}
	return(pow);
}

static void z_div(doublecomplex *c, doublecomplex *a, doublecomplex *b)
{
	doublereal ratio, den;
	doublereal abr, abi, cr;

	if( (abr = b->r) < 0.)
		abr = - abr;
	if( (abi = b->i) < 0.)
		abi = - abi;
	if( abr <= abi )
	{
		if(abi == 0) {
			//			printf("complex division by zero\n");
			return;
		}
		ratio = b->r / b->i ;
		den = b->i * (1 + ratio*ratio);
		cr = (a->r*ratio + a->i) / den;
		c->i = (a->i*ratio - a->r) / den;
	}

	else
	{
		ratio = b->i / b->r ;
		den = b->r * (1 + ratio*ratio);
		cr = (a->r + a->i*ratio) / den;
		c->i = (a->i - a->r*ratio) / den;
	}
	c->r = cr;
}

static doublereal f__cabs(doublereal real, doublereal imag)
{
	doublereal temp;
	if(real < 0)
		real = -real;
	if(imag < 0)
		imag = -imag;
	if(imag > real){
		temp = real;
		real = imag;
		imag = temp;
	}
	if((real+imag) == real)
		return(real);

	temp = imag/real;
    temp = real*sqrtf((double)1.0 + temp*temp);  /*overflow!!*/
	return(temp);
}

static doublereal z_abs(doublecomplex *z) {
	return( f__cabs( z->r, z->i ) );
}

///////////////////////////////////////////
//LAPACK routines
///////////////////////////////////////////

/* Table of constant values */
 integer c__0 = 0;
 integer c__1 = 1;
 doublecomplex c_b113 = {-1.,-0.};

static int zgefa_(doublecomplex *a, integer *lda, integer *n, integer *ipvt, integer *info)
{
	/* System generated locals */
	integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
	doublereal d__1, d__2;
	doublecomplex z__1;

	/* Local variables */
	integer j, k, l;
	doublecomplex t;
	integer kp1, nm1;


	/*     zgefa factors a complex*16 matrix by gaussian elimination. */

	/*     zgefa is usually called by zgeco, but it can be called */
	/*     directly with a saving in time if  rcond  is not needed. */
	/*     (time for zgeco) = (1 + 9/n)*(time for zgefa) . */

	/*     on entry */

	/*        a       complex*16(lda, n) */
	/*                the matrix to be factored. */

	/*        lda     integer */
	/*                the leading dimension of the array  a . */

	/*        n       integer */
	/*                the order of the matrix  a . */

	/*     on return */

	/*        a       an upper triangular matrix and the multipliers */
	/*                which were used to obtain it. */
	/*                the factorization can be written  a = l*u  where */
	/*                l  is a product of permutation and unit lower */
	/*                triangular matrices and  u  is upper triangular. */

	/*        ipvt    integer(n) */
	/*                an integer vector of pivot indices. */

	/*        info    integer */
	/*                = 0  normal value. */
	/*                = k  if  u(k,k) .eq. 0.0 .  this is not an error */
	/*                     condition for this subroutine, but it does */
	/*                     indicate that zgesl or zgedi will divide by zero */
	/*                     if called.  use  rcond  in zgeco for a reliable */
	/*                     indication of singularity. */

	/*     linpack. this version dated 08/14/78 . */
	/*     cleve moler, university of new mexico, argonne national lab. */

	/*     subroutines and functions */

	/*     blas zaxpy,zscal,izamax */
	/*     fortran dabs */

	/*     internal variables */



	/*     gaussian elimination with partial pivoting */

	/* Parameter adjustments */
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;
	--ipvt;

	/* Function Body */
	*info = 0;
	nm1 = *n - 1;
	if (nm1 < 1) {
		goto L70;
	}
	i__1 = nm1;
	for (k = 1; k <= i__1; ++k) {
		kp1 = k + 1;

		/*        find l = pivot index */

		i__2 = *n - k + 1;
		l = izamax_(&i__2, &a[k + k * a_dim1], &c__1) + k - 1;
		ipvt[k] = l;

		/*        zero pivot implies this column already triangularized */

		i__2 = l + k * a_dim1;
		i__3 = l + k * a_dim1;
		z__1.r = a[i__3].r * 0. - a[i__3].i * -1., z__1.i = a[i__3].i * 0. + 
			a[i__3].r * -1.;
		if ((d__1 = a[i__2].r, fabs(d__1)) + (d__2 = z__1.r, fabs(d__2)) == 0.) 
		{
			goto L40;
		}

		/*           interchange if necessary */

		if (l == k) {
			goto L10;
		}
		i__2 = l + k * a_dim1;
		t.r = a[i__2].r, t.i = a[i__2].i;
		i__2 = l + k * a_dim1;
		i__3 = k + k * a_dim1;
		a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
		i__2 = k + k * a_dim1;
		a[i__2].r = t.r, a[i__2].i = t.i;
L10:

		/*           compute multipliers */

		z_div(&z__1, &c_b113, &a[k + k * a_dim1]);
		t.r = z__1.r, t.i = z__1.i;
		i__2 = *n - k;
		zscal_(&i__2, &t, &a[k + 1 + k * a_dim1], &c__1);

		/*           row elimination with column indexing */

		i__2 = *n;
		for (j = kp1; j <= i__2; ++j) {
			i__3 = l + j * a_dim1;
			t.r = a[i__3].r, t.i = a[i__3].i;
			if (l == k) {
				goto L20;
			}
			i__3 = l + j * a_dim1;
			i__4 = k + j * a_dim1;
			a[i__3].r = a[i__4].r, a[i__3].i = a[i__4].i;
			i__3 = k + j * a_dim1;
			a[i__3].r = t.r, a[i__3].i = t.i;
L20:
			i__3 = *n - k;
			zaxpy_(&i__3, &t, &a[k + 1 + k * a_dim1], &c__1, &a[k + 1 + j * 
				a_dim1], &c__1);
			/* L30: */
		}
		goto L50;
L40:
		*info = k;
L50:
		/* L60: */
		;
	}
L70:
	ipvt[*n] = *n;
	i__1 = *n + *n * a_dim1;
	i__2 = *n + *n * a_dim1;
	z__1.r = a[i__2].r * 0. - a[i__2].i * -1., z__1.i = a[i__2].i * 0. + a[
		i__2].r * -1.;
		if ((d__1 = a[i__1].r, fabs(d__1)) + (d__2 = z__1.r, fabs(d__2)) == 0.) {
			*info = *n;
		}
		return 0;
} /* zgefa_ */

static int zgesl_(doublecomplex *a, integer *lda, integer *n, integer *ipvt, doublecomplex *b, integer *job)
{
	/* System generated locals */
	integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
	doublecomplex z__1, z__2, z__3;

	/* Local variables */
	integer k, l;
	doublecomplex t;
	integer kb, nm1;

	/*     zgesl solves the complex*16 system */
	/*     a * x = b  or  ctrans(a) * x = b */
	/*     using the factors computed by zgeco or zgefa. */

	/*     on entry */

	/*        a       complex*16(lda, n) */
	/*                the output from zgeco or zgefa. */

	/*        lda     integer */
	/*                the leading dimension of the array  a . */

	/*        n       integer */
	/*                the order of the matrix  a . */

	/*        ipvt    integer(n) */
	/*                the pivot vector from zgeco or zgefa. */

	/*        b       complex*16(n) */
	/*                the right hand side vector. */

	/*        job     integer */
	/*                = 0         to solve  a*x = b , */
	/*                = nonzero   to solve  ctrans(a)*x = b  where */
	/*                            ctrans(a)  is the conjugate transpose. */

	/*     on return */

	/*        b       the solution vector  x . */

	/*     error condition */

	/*        a division by zero will occur if the input factor contains a */
	/*        zero on the diagonal.  technically this indicates singularity */
	/*        but it is often caused by improper arguments or improper */
	/*        setting of lda .  it will not occur if the subroutines are */
	/*        called correctly and if zgeco has set rcond .gt. 0.0 */
	/*        or zgefa has set info .eq. 0 . */

	/*     to compute  inverse(a) * c  where  c  is a matrix */
	/*     with  p  columns */
	/*           call zgeco(a,lda,n,ipvt,rcond,z) */
	/*           if (rcond is too small) go to ... */
	/*           do 10 j = 1, p */
	/*              call zgesl(a,lda,n,ipvt,c(1,j),0) */
	/*        10 continue */

	/*     linpack. this version dated 08/14/78 . */
	/*     cleve moler, university of new mexico, argonne national lab. */

	/*     subroutines and functions */

	/*     blas zaxpy,zdotc */
	/*     fortran dconjg */

	/*     internal variables */


	/* Parameter adjustments */
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;
	--ipvt;
	--b;

	/* Function Body */
	nm1 = *n - 1;
	if (*job != 0) {
		goto L50;
	}

	/*        job = 0 , solve  a * x = b */
	/*        first solve  l*y = b */

	if (nm1 < 1) {
		goto L30;
	}
	i__1 = nm1;
	for (k = 1; k <= i__1; ++k) {
		l = ipvt[k];
		i__2 = l;
		t.r = b[i__2].r, t.i = b[i__2].i;
		if (l == k) {
			goto L10;
		}
		i__2 = l;
		i__3 = k;
		b[i__2].r = b[i__3].r, b[i__2].i = b[i__3].i;
		i__2 = k;
		b[i__2].r = t.r, b[i__2].i = t.i;
L10:
		i__2 = *n - k;
		zaxpy_(&i__2, &t, &a[k + 1 + k * a_dim1], &c__1, &b[k + 1], &c__1);
		/* L20: */
	}
L30:

	/*        now solve  u*x = y */

	i__1 = *n;
	for (kb = 1; kb <= i__1; ++kb) {
		k = *n + 1 - kb;
		i__2 = k;
		z_div(&z__1, &b[k], &a[k + k * a_dim1]);
		b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		i__2 = k;
		z__1.r = -b[i__2].r, z__1.i = -b[i__2].i;
		t.r = z__1.r, t.i = z__1.i;
		i__2 = k - 1;
		zaxpy_(&i__2, &t, &a[k * a_dim1 + 1], &c__1, &b[1], &c__1);
		/* L40: */
	}
	goto L100;
L50:

	/*        job = nonzero, solve  ctrans(a) * x = b */
	/*        first solve  ctrans(u)*y = b */

	i__1 = *n;
	for (k = 1; k <= i__1; ++k) {
		i__2 = k - 1;
		zdotc_(&z__1, &i__2, &a[k * a_dim1 + 1], &c__1, &b[1], &c__1);
		t.r = z__1.r, t.i = z__1.i;
		i__2 = k;
		i__3 = k;
		z__2.r = b[i__3].r - t.r, z__2.i = b[i__3].i - t.i;
		d_cnjg(&z__3, &a[k + k * a_dim1]);
		z_div(&z__1, &z__2, &z__3);
		b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		/* L60: */
	}

	/*        now solve ctrans(l)*x = y */

	if (nm1 < 1) {
		goto L90;
	}
	i__1 = nm1;
	for (kb = 1; kb <= i__1; ++kb) {
		k = *n - kb;
		i__2 = k;
		i__3 = k;
		i__4 = *n - k;
		zdotc_(&z__2, &i__4, &a[k + 1 + k * a_dim1], &c__1, &b[k + 1], &c__1);
		z__1.r = b[i__3].r + z__2.r, z__1.i = b[i__3].i + z__2.i;
		b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		l = ipvt[k];
		if (l == k) {
			goto L70;
		}
		i__2 = l;
		t.r = b[i__2].r, t.i = b[i__2].i;
		i__2 = l;
		i__3 = k;
		b[i__2].r = b[i__3].r, b[i__2].i = b[i__3].i;
		i__2 = k;
		b[i__2].r = t.r, b[i__2].i = t.i;
L70:
		/* L80: */
		;
	}
L90:
L100:
	return 0;
} /* zgesl_ */


static int zgesv_(integer *n, integer *m, doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *b, integer *ldb, integer *iflag)
{
	/* System generated locals */
	integer a_dim1, a_offset, b_dim1, b_offset, i__1;

	/* Local variables */
	integer j;

	/* Parameter adjustments */
	--ipiv;
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;
	b_dim1 = *ldb;
	b_offset = 1 + b_dim1;
	b -= b_offset;

	/* Function Body */
	zgefa_(&a[a_offset], lda, n, &ipiv[1], iflag);
	if (*iflag != 0) {
    //	s_stop("Error in ZGESV (LU factorisation)", (ftnlen)33);
        return 1;
	}
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
		zgesl_(&a[a_offset], lda, n, &ipiv[1], &b[j * b_dim1 + 1], &c__0);
	}
	return 0;
} /* zgesv_ */


///////////////////////////////////////////
//BLAS routines
///////////////////////////////////////////

// dummy error handler
static int xerbla2_(char *srname, integer *info, ftnlen srname_len)
{
	return 0;
} /* xerbla2_ */


/* ----------------------------------------------------------------------| */
static long lsame_(char *ca, char *cb, ftnlen ca_len, ftnlen cb_len)
{
	/* System generated locals */
	long ret_val;

	/* Local variables */
	integer inta, intb, zcode;


	/*  -- LAPACK auxiliary routine (version 1.1) -- */
	/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
	/*     Courant Institute, Argonne National Lab, and Rice University */
	/*     February 29, 1992 */

	/*     .. Scalar Arguments .. */
	/*     .. */

	/*  Purpose */
	/*  ======= */

	/*  LSAME returns .TRUE. if CA is the same letter as CB regardless of */
	/*  case. */

	/*  Arguments */
	/*  ========= */

	/*  CA      (input) CHARACTER*1 */
	/*  CB      (input) CHARACTER*1 */
	/*          CA and CB specify the single characters to be compared. */

	/*     .. Intrinsic Functions .. */
	/*     .. */
	/*     .. Local Scalars .. */
	/*     .. */
	/*     .. Executable Statements .. */

	/*     Test if the characters are equal */

	ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
	if (ret_val) {
		return ret_val;
	}

	/*     Now test for equivalence if both characters are alphabetic. */

	zcode = 'Z';

	/*     Use 'Z' rather than 'A' so that ASCII can be detected on Prime */
	/*     machines, on which ICHAR returns a value with bit 8 set. */
	/*     ICHAR('A') on Prime machines returns 193 which is the same as */
	/*     ICHAR('A') on an EBCDIC machine. */

	inta = *(unsigned char *)ca;
	intb = *(unsigned char *)cb;

	if (zcode == 90 || zcode == 122) {

		/*        ASCII is assumed - ZCODE is the ASCII code of either lower or */
		/*        upper case 'Z'. */

		if (inta >= 97 && inta <= 122) {
			inta += -32;
		}
		if (intb >= 97 && intb <= 122) {
			intb += -32;
		}

	} else if (zcode == 233 || zcode == 169) {

		/*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or */
		/*        upper case 'Z'. */

		if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta 
			>= 162 && inta <= 169) {
				inta += 64;
		}
		if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb 
			>= 162 && intb <= 169) {
				intb += 64;
		}

	} else if (zcode == 218 || zcode == 250) {

		/*        ASCII is assumed, on Prime machines - ZCODE is the ASCII code */
		/*        plus 128 of either lower or upper case 'Z'. */

		if (inta >= 225 && inta <= 250) {
			inta += -32;
		}
		if (intb >= 225 && intb <= 250) {
			intb += -32;
		}
	}
	ret_val = inta == intb;

	/*     RETURN */

	/*     End of LSAME */

	return ret_val;
} /* lsame_ */


static int zgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, doublecomplex *alpha, doublecomplex *a, integer *lda, 
	doublecomplex *b, integer *ldb, doublecomplex *beta, doublecomplex *
	c__, integer *ldc, ftnlen transa_len, ftnlen transb_len)
{
	/* System generated locals */
	integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, 
		i__3, i__4, i__5, i__6;
	doublecomplex z__1, z__2, z__3, z__4;


	/* Local variables */
	integer i__, j, l, info;
	long nota, notb;
	doublecomplex temp;
	long conja, conjb;
//    integer ncola = 0;
    integer nrowa, nrowb;

	/*     .. Scalar Arguments .. */
	/*     .. Array Arguments .. */
	/*     .. */

	/*  Purpose */
	/*  ======= */

	/*  ZGEMM  performs one of the matrix-matrix operations */

	/*     C := alpha*op( A )*op( B ) + beta*C, */

	/*  where  op( X ) is one of */

	/*     op( X ) = X   or   op( X ) = X'   or   op( X ) = conjg( X' ), */

	/*  alpha and beta are scalars, and A, B and C are matrices, with op( A ) */
	/*  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix. */

	/*  Parameters */
	/*  ========== */

	/*  TRANSA - CHARACTER*1. */
	/*           On entry, TRANSA specifies the form of op( A ) to be used in */
	/*           the matrix multiplication as follows: */

	/*              TRANSA = 'N' or 'n',  op( A ) = A. */

	/*              TRANSA = 'T' or 't',  op( A ) = A'. */

	/*              TRANSA = 'C' or 'c',  op( A ) = conjg( A' ). */

	/*           Unchanged on exit. */

	/*  TRANSB - CHARACTER*1. */
	/*           On entry, TRANSB specifies the form of op( B ) to be used in */
	/*           the matrix multiplication as follows: */

	/*              TRANSB = 'N' or 'n',  op( B ) = B. */

	/*              TRANSB = 'T' or 't',  op( B ) = B'. */

	/*              TRANSB = 'C' or 'c',  op( B ) = conjg( B' ). */

	/*           Unchanged on exit. */

	/*  M      - INTEGER. */
	/*           On entry,  M  specifies  the number  of rows  of the  matrix */
	/*           op( A )  and of the  matrix  C.  M  must  be at least  zero. */
	/*           Unchanged on exit. */

	/*  N      - INTEGER. */
	/*           On entry,  N  specifies the number  of columns of the matrix */
	/*           op( B ) and the number of columns of the matrix C. N must be */
	/*           at least zero. */
	/*           Unchanged on exit. */

	/*  K      - INTEGER. */
	/*           On entry,  K  specifies  the number of columns of the matrix */
	/*           op( A ) and the number of rows of the matrix op( B ). K must */
	/*           be at least  zero. */
	/*           Unchanged on exit. */

	/*  ALPHA  - COMPLEX*16      . */
	/*           On entry, ALPHA specifies the scalar alpha. */
	/*           Unchanged on exit. */

	/*  A      - COMPLEX*16       array of DIMENSION ( LDA, ka ), where ka is */
	/*           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise. */
	/*           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k */
	/*           part of the array  A  must contain the matrix  A,  otherwise */
	/*           the leading  k by m  part of the array  A  must contain  the */
	/*           matrix A. */
	/*           Unchanged on exit. */

	/*  LDA    - INTEGER. */
	/*           On entry, LDA specifies the first dimension of A as declared */
	/*           in the calling (sub) program. When  TRANSA = 'N' or 'n' then */
	/*           LDA must be at least  max( 1, m ), otherwise  LDA must be at */
	/*           least  max( 1, k ). */
	/*           Unchanged on exit. */

	/*  B      - COMPLEX*16       array of DIMENSION ( LDB, kb ), where kb is */
	/*           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise. */
	/*           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n */
	/*           part of the array  B  must contain the matrix  B,  otherwise */
	/*           the leading  n by k  part of the array  B  must contain  the */
	/*           matrix B. */
	/*           Unchanged on exit. */

	/*  LDB    - INTEGER. */
	/*           On entry, LDB specifies the first dimension of B as declared */
	/*           in the calling (sub) program. When  TRANSB = 'N' or 'n' then */
	/*           LDB must be at least  max( 1, k ), otherwise  LDB must be at */
	/*           least  max( 1, n ). */
	/*           Unchanged on exit. */

	/*  BETA   - COMPLEX*16      . */
	/*           On entry,  BETA  specifies the scalar  beta.  When  BETA  is */
	/*           supplied as zero then C need not be set on input. */
	/*           Unchanged on exit. */

	/*  C      - COMPLEX*16       array of DIMENSION ( LDC, n ). */
	/*           Before entry, the leading  m by n  part of the array  C must */
	/*           contain the matrix  C,  except when  beta  is zero, in which */
	/*           case C need not be set on entry. */
	/*           On exit, the array  C  is overwritten by the  m by n  matrix */
	/*           ( alpha*op( A )*op( B ) + beta*C ). */

	/*  LDC    - INTEGER. */
	/*           On entry, LDC specifies the first dimension of C as declared */
	/*           in  the  calling  (sub)  program.   LDC  must  be  at  least */
	/*           max( 1, m ). */
	/*           Unchanged on exit. */


	/*  Level 3 Blas routine. */

	/*  -- Written on 8-February-1989. */
	/*     Jack Dongarra, Argonne National Laboratory. */
	/*     Iain Duff, AERE Harwell. */
	/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
	/*     Sven Hammarling, Numerical Algorithms Group Ltd. */


	/*     .. External Functions .. */
	/*     .. External Subroutines .. */
	/*     .. Intrinsic Functions .. */
	/*     .. Local Scalars .. */
	/*     .. Parameters .. */
	/*     .. */
	/*     .. Executable Statements .. */

	/*     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not */
	/*     conjugated or transposed, set  CONJA and CONJB  as true if  A  and */
	/*     B  respectively are to be  transposed but  not conjugated  and set */
	/*     NROWA, NCOLA and  NROWB  as the number of rows and  columns  of  A */
	/*     and the number of rows of  B  respectively. */

	/* Parameter adjustments */
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;
	b_dim1 = *ldb;
	b_offset = 1 + b_dim1;
	b -= b_offset;
	c_dim1 = *ldc;
	c_offset = 1 + c_dim1;
	c__ -= c_offset;

	/* Function Body */
	nota = lsame_(transa, "N", (ftnlen)1, (ftnlen)1);
	notb = lsame_(transb, "N", (ftnlen)1, (ftnlen)1);
	conja = lsame_(transa, "C", (ftnlen)1, (ftnlen)1);
	conjb = lsame_(transb, "C", (ftnlen)1, (ftnlen)1);
	if (nota) {
		nrowa = *m;
    //	ncola = *k;
	} else {
		nrowa = *k;
    //	ncola = *m;
	}
	if (notb) {
		nrowb = *k;
	} else {
		nrowb = *n;
	}

	/*     Test the input parameters. */

	info = 0;
	if (! nota && ! conja && ! lsame_(transa, "T", (ftnlen)1, (ftnlen)1)) {
		info = 1;
	} else if (! notb && ! conjb && ! lsame_(transb, "T", (ftnlen)1, (ftnlen)
		1)) {
			info = 2;
	} else if (*m < 0) {
		info = 3;
	} else if (*n < 0) {
		info = 4;
	} else if (*k < 0) {
		info = 5;
	} else if (*lda < max(1,nrowa)) {
		info = 8;
	} else if (*ldb < max(1,nrowb)) {
		info = 10;
	} else if (*ldc < max(1,*m)) {
		info = 13;
	}
	if (info != 0) {
		xerbla2_("ZGEMM ", &info, (ftnlen)6);
		return 0;
	}

	/*     Quick return if possible. */

	if (*m == 0 || *n == 0 || (alpha->r == 0. && alpha->i == 0. || *k == 0) &&
		(beta->r == 1. && beta->i == 0.)) {
			return 0;
	}

	/*     And when  alpha.eq.zero. */

	if (alpha->r == 0. && alpha->i == 0.) {
		if (beta->r == 0. && beta->i == 0.) {
			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				i__2 = *m;
				for (i__ = 1; i__ <= i__2; ++i__) {
					i__3 = i__ + j * c_dim1;
					c__[i__3].r = 0., c__[i__3].i = 0.;
					/* L10: */
				}
				/* L20: */
			}
		} else {
			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				i__2 = *m;
				for (i__ = 1; i__ <= i__2; ++i__) {
					i__3 = i__ + j * c_dim1;
					i__4 = i__ + j * c_dim1;
					z__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i, 
						z__1.i = beta->r * c__[i__4].i + beta->i * c__[
							i__4].r;
							c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
							/* L30: */
				}
				/* L40: */
			}
		}
		return 0;
	}

	/*     Start the operations. */

	if (notb) {
		if (nota) {

			/*           Form  C := alpha*A*B + beta*C. */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				if (beta->r == 0. && beta->i == 0.) {
					i__2 = *m;
					for (i__ = 1; i__ <= i__2; ++i__) {
						i__3 = i__ + j * c_dim1;
						c__[i__3].r = 0., c__[i__3].i = 0.;
						/* L50: */
					}
				} else if (beta->r != 1. || beta->i != 0.) {
					i__2 = *m;
					for (i__ = 1; i__ <= i__2; ++i__) {
						i__3 = i__ + j * c_dim1;
						i__4 = i__ + j * c_dim1;
						z__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__1.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
						/* L60: */
					}
				}
				i__2 = *k;
				for (l = 1; l <= i__2; ++l) {
					i__3 = l + j * b_dim1;
					if (b[i__3].r != 0. || b[i__3].i != 0.) {
						i__3 = l + j * b_dim1;
						z__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i, 
							z__1.i = alpha->r * b[i__3].i + alpha->i * b[
								i__3].r;
								temp.r = z__1.r, temp.i = z__1.i;
								i__3 = *m;
								for (i__ = 1; i__ <= i__3; ++i__) {
									i__4 = i__ + j * c_dim1;
									i__5 = i__ + j * c_dim1;
									i__6 = i__ + l * a_dim1;
									z__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i, 
										z__2.i = temp.r * a[i__6].i + temp.i * a[
											i__6].r;
											z__1.r = c__[i__5].r + z__2.r, z__1.i = c__[i__5]
											.i + z__2.i;
											c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
											/* L70: */
								}
					}
					/* L80: */
				}
				/* L90: */
			}
		} else if (conja) {

			/*           Form  C := alpha*conjg( A' )*B + beta*C. */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				i__2 = *m;
				for (i__ = 1; i__ <= i__2; ++i__) {
					temp.r = 0., temp.i = 0.;
					i__3 = *k;
					for (l = 1; l <= i__3; ++l) {
						d_cnjg(&z__3, &a[l + i__ * a_dim1]);
						i__4 = l + j * b_dim1;
						z__2.r = z__3.r * b[i__4].r - z__3.i * b[i__4].i, 
							z__2.i = z__3.r * b[i__4].i + z__3.i * b[i__4]
						.r;
						z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
						temp.r = z__1.r, temp.i = z__1.i;
						/* L100: */
					}
					if (beta->r == 0. && beta->i == 0.) {
						i__3 = i__ + j * c_dim1;
						z__1.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__1.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					} else {
						i__3 = i__ + j * c_dim1;
						z__2.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__2.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						i__4 = i__ + j * c_dim1;
						z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__3.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					}
					/* L110: */
				}
				/* L120: */
			}
		} else {

			/*           Form  C := alpha*A'*B + beta*C */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				i__2 = *m;
				for (i__ = 1; i__ <= i__2; ++i__) {
					temp.r = 0., temp.i = 0.;
					i__3 = *k;
					for (l = 1; l <= i__3; ++l) {
						i__4 = l + i__ * a_dim1;
						i__5 = l + j * b_dim1;
						z__2.r = a[i__4].r * b[i__5].r - a[i__4].i * b[i__5]
						.i, z__2.i = a[i__4].r * b[i__5].i + a[i__4]
						.i * b[i__5].r;
						z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
						temp.r = z__1.r, temp.i = z__1.i;
						/* L130: */
					}
					if (beta->r == 0. && beta->i == 0.) {
						i__3 = i__ + j * c_dim1;
						z__1.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__1.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					} else {
						i__3 = i__ + j * c_dim1;
						z__2.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__2.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						i__4 = i__ + j * c_dim1;
						z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__3.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					}
					/* L140: */
				}
				/* L150: */
			}
		}
	} else if (nota) {
		if (conjb) {

			/*           Form  C := alpha*A*conjg( B' ) + beta*C. */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				if (beta->r == 0. && beta->i == 0.) {
					i__2 = *m;
					for (i__ = 1; i__ <= i__2; ++i__) {
						i__3 = i__ + j * c_dim1;
						c__[i__3].r = 0., c__[i__3].i = 0.;
						/* L160: */
					}
				} else if (beta->r != 1. || beta->i != 0.) {
					i__2 = *m;
					for (i__ = 1; i__ <= i__2; ++i__) {
						i__3 = i__ + j * c_dim1;
						i__4 = i__ + j * c_dim1;
						z__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__1.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
						/* L170: */
					}
				}
				i__2 = *k;
				for (l = 1; l <= i__2; ++l) {
					i__3 = j + l * b_dim1;
					if (b[i__3].r != 0. || b[i__3].i != 0.) {
						d_cnjg(&z__2, &b[j + l * b_dim1]);
						z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, 
							z__1.i = alpha->r * z__2.i + alpha->i * 
							z__2.r;
						temp.r = z__1.r, temp.i = z__1.i;
						i__3 = *m;
						for (i__ = 1; i__ <= i__3; ++i__) {
							i__4 = i__ + j * c_dim1;
							i__5 = i__ + j * c_dim1;
							i__6 = i__ + l * a_dim1;
							z__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i, 
								z__2.i = temp.r * a[i__6].i + temp.i * a[
									i__6].r;
									z__1.r = c__[i__5].r + z__2.r, z__1.i = c__[i__5]
									.i + z__2.i;
									c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
									/* L180: */
						}
					}
					/* L190: */
				}
				/* L200: */
			}
		} else {

			/*           Form  C := alpha*A*B'          + beta*C */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				if (beta->r == 0. && beta->i == 0.) {
					i__2 = *m;
					for (i__ = 1; i__ <= i__2; ++i__) {
						i__3 = i__ + j * c_dim1;
						c__[i__3].r = 0., c__[i__3].i = 0.;
						/* L210: */
					}
				} else if (beta->r != 1. || beta->i != 0.) {
					i__2 = *m;
					for (i__ = 1; i__ <= i__2; ++i__) {
						i__3 = i__ + j * c_dim1;
						i__4 = i__ + j * c_dim1;
						z__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__1.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
						/* L220: */
					}
				}
				i__2 = *k;
				for (l = 1; l <= i__2; ++l) {
					i__3 = j + l * b_dim1;
					if (b[i__3].r != 0. || b[i__3].i != 0.) {
						i__3 = j + l * b_dim1;
						z__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i, 
							z__1.i = alpha->r * b[i__3].i + alpha->i * b[
								i__3].r;
								temp.r = z__1.r, temp.i = z__1.i;
								i__3 = *m;
								for (i__ = 1; i__ <= i__3; ++i__) {
									i__4 = i__ + j * c_dim1;
									i__5 = i__ + j * c_dim1;
									i__6 = i__ + l * a_dim1;
									z__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i, 
										z__2.i = temp.r * a[i__6].i + temp.i * a[
											i__6].r;
											z__1.r = c__[i__5].r + z__2.r, z__1.i = c__[i__5]
											.i + z__2.i;
											c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
											/* L230: */
								}
					}
					/* L240: */
				}
				/* L250: */
			}
		}
	} else if (conja) {
		if (conjb) {

			/*           Form  C := alpha*conjg( A' )*conjg( B' ) + beta*C. */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				i__2 = *m;
				for (i__ = 1; i__ <= i__2; ++i__) {
					temp.r = 0., temp.i = 0.;
					i__3 = *k;
					for (l = 1; l <= i__3; ++l) {
						d_cnjg(&z__3, &a[l + i__ * a_dim1]);
						d_cnjg(&z__4, &b[j + l * b_dim1]);
						z__2.r = z__3.r * z__4.r - z__3.i * z__4.i, z__2.i = 
							z__3.r * z__4.i + z__3.i * z__4.r;
						z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
						temp.r = z__1.r, temp.i = z__1.i;
						/* L260: */
					}
					if (beta->r == 0. && beta->i == 0.) {
						i__3 = i__ + j * c_dim1;
						z__1.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__1.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					} else {
						i__3 = i__ + j * c_dim1;
						z__2.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__2.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						i__4 = i__ + j * c_dim1;
						z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__3.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					}
					/* L270: */
				}
				/* L280: */
			}
		} else {

			/*           Form  C := alpha*conjg( A' )*B' + beta*C */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				i__2 = *m;
				for (i__ = 1; i__ <= i__2; ++i__) {
					temp.r = 0., temp.i = 0.;
					i__3 = *k;
					for (l = 1; l <= i__3; ++l) {
						d_cnjg(&z__3, &a[l + i__ * a_dim1]);
						i__4 = j + l * b_dim1;
						z__2.r = z__3.r * b[i__4].r - z__3.i * b[i__4].i, 
							z__2.i = z__3.r * b[i__4].i + z__3.i * b[i__4]
						.r;
						z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
						temp.r = z__1.r, temp.i = z__1.i;
						/* L290: */
					}
					if (beta->r == 0. && beta->i == 0.) {
						i__3 = i__ + j * c_dim1;
						z__1.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__1.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					} else {
						i__3 = i__ + j * c_dim1;
						z__2.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__2.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						i__4 = i__ + j * c_dim1;
						z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__3.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					}
					/* L300: */
				}
				/* L310: */
			}
		}
	} else {
		if (conjb) {

			/*           Form  C := alpha*A'*conjg( B' ) + beta*C */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				i__2 = *m;
				for (i__ = 1; i__ <= i__2; ++i__) {
					temp.r = 0., temp.i = 0.;
					i__3 = *k;
					for (l = 1; l <= i__3; ++l) {
						i__4 = l + i__ * a_dim1;
						d_cnjg(&z__3, &b[j + l * b_dim1]);
						z__2.r = a[i__4].r * z__3.r - a[i__4].i * z__3.i, 
							z__2.i = a[i__4].r * z__3.i + a[i__4].i * 
							z__3.r;
						z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
						temp.r = z__1.r, temp.i = z__1.i;
						/* L320: */
					}
					if (beta->r == 0. && beta->i == 0.) {
						i__3 = i__ + j * c_dim1;
						z__1.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__1.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					} else {
						i__3 = i__ + j * c_dim1;
						z__2.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__2.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						i__4 = i__ + j * c_dim1;
						z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__3.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					}
					/* L330: */
				}
				/* L340: */
			}
		} else {

			/*           Form  C := alpha*A'*B' + beta*C */

			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
				i__2 = *m;
				for (i__ = 1; i__ <= i__2; ++i__) {
					temp.r = 0., temp.i = 0.;
					i__3 = *k;
					for (l = 1; l <= i__3; ++l) {
						i__4 = l + i__ * a_dim1;
						i__5 = j + l * b_dim1;
						z__2.r = a[i__4].r * b[i__5].r - a[i__4].i * b[i__5]
						.i, z__2.i = a[i__4].r * b[i__5].i + a[i__4]
						.i * b[i__5].r;
						z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
						temp.r = z__1.r, temp.i = z__1.i;
						/* L350: */
					}
					if (beta->r == 0. && beta->i == 0.) {
						i__3 = i__ + j * c_dim1;
						z__1.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__1.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					} else {
						i__3 = i__ + j * c_dim1;
						z__2.r = alpha->r * temp.r - alpha->i * temp.i, 
							z__2.i = alpha->r * temp.i + alpha->i * 
							temp.r;
						i__4 = i__ + j * c_dim1;
						z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
						.i, z__3.i = beta->r * c__[i__4].i + beta->i *
							c__[i__4].r;
						z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
						c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
					}
					/* L360: */
				}
				/* L370: */
			}
		}
	}

	return 0;

	/*     End of ZGEMM . */

} /* zgemm_ */


static  int zscal_(integer *n, doublecomplex *za, doublecomplex *zx, integer *incx)
{
	/* System generated locals */
	integer i__1, i__2, i__3;
	doublecomplex z__1;

	/* Local variables */
	integer i__, ix;


	/*     scales a vector by a constant. */
	/*     jack dongarra, 3/11/78. */
	/*     modified 3/93 to return if incx .le. 0. */


	/* Parameter adjustments */
	--zx;

	/* Function Body */
	if (*n <= 0 || *incx <= 0) {
		return 0;
	}
	if (*incx == 1) {
		goto L20;
	}

	/*        code for increment not equal to 1 */

	ix = 1;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = ix;
		i__3 = ix;
		z__1.r = za->r * zx[i__3].r - za->i * zx[i__3].i, z__1.i = za->r * zx[
			i__3].i + za->i * zx[i__3].r;
			zx[i__2].r = z__1.r, zx[i__2].i = z__1.i;
			ix += *incx;
			/* L10: */
	}
	return 0;

	/*        code for increment equal to 1 */

L20:
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		i__3 = i__;
		z__1.r = za->r * zx[i__3].r - za->i * zx[i__3].i, z__1.i = za->r * zx[
			i__3].i + za->i * zx[i__3].r;
			zx[i__2].r = z__1.r, zx[i__2].i = z__1.i;
			/* L30: */
	}
	return 0;
} /* zscal_ */

static doublereal dcabs1_(doublecomplex *z__)
{
	/* System generated locals */
	doublereal ret_val;
	doublecomplex equiv_0[1];

	/* Local variables */
#define t ((doublereal *)equiv_0)
#define zz (equiv_0)

	zz->r = z__->r, zz->i = z__->i;
	ret_val = fabs(t[0]) + fabs(t[1]);
	return ret_val;
} /* dcabs1_ */

#undef zz
#undef t

static integer izamax_(integer *n, doublecomplex *zx, integer *incx)
{
	/* System generated locals */
	integer ret_val, i__1;

	/* Local variables */
	integer i__, ix;
	doublereal smax;

	/*     finds the index of element having max. absolute value. */
	/*     jack dongarra, 1/15/85. */
	/*     modified 3/93 to return if incx .le. 0. */

	/* Parameter adjustments */
	--zx;

	/* Function Body */
	ret_val = 0;
	if (*n < 1 || *incx <= 0) {
		return ret_val;
	}
	ret_val = 1;
	if (*n == 1) {
		return ret_val;
	}
	if (*incx == 1) {
		goto L20;
	}

	/*        code for increment not equal to 1 */

	ix = 1;
	smax = dcabs1_(&zx[1]);
	ix += *incx;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
		if (dcabs1_(&zx[ix]) <= smax) {
			goto L5;
		}
		ret_val = i__;
		smax = dcabs1_(&zx[ix]);
L5:
		ix += *incx;
		/* L10: */
	}
	return ret_val;

	/*        code for increment equal to 1 */

L20:
	smax = dcabs1_(&zx[1]);
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
		if (dcabs1_(&zx[i__]) <= smax) {
			goto L30;
		}
		ret_val = i__;
		smax = dcabs1_(&zx[i__]);
L30:
		;
	}
	return ret_val;
} /* izamax_ */


static int zaxpy_(integer *n, doublecomplex *za, doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy)
{
	/* System generated locals */
	integer i__1, i__2, i__3, i__4;
	doublecomplex z__1, z__2;

	/* Local variables */
	integer i__, ix, iy;

	/*     constant times a vector plus a vector. */
	/*     jack dongarra, 3/11/78. */

	/* Parameter adjustments */
	--zy;
	--zx;

	/* Function Body */
	if (*n <= 0) {
		return 0;
	}
	if (dcabs1_(za) == 0.) {
		return 0;
	}
	if (*incx == 1 && *incy == 1) {
		goto L20;
	}

	/*        code for unequal increments or equal increments */
	/*          not equal to 1 */

	ix = 1;
	iy = 1;
	if (*incx < 0) {
		ix = (-(*n) + 1) * *incx + 1;
	}
	if (*incy < 0) {
		iy = (-(*n) + 1) * *incy + 1;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = iy;
		i__3 = iy;
		i__4 = ix;
		z__2.r = za->r * zx[i__4].r - za->i * zx[i__4].i, z__2.i = za->r * zx[
			i__4].i + za->i * zx[i__4].r;
			z__1.r = zy[i__3].r + z__2.r, z__1.i = zy[i__3].i + z__2.i;
			zy[i__2].r = z__1.r, zy[i__2].i = z__1.i;
			ix += *incx;
			iy += *incy;
			/* L10: */
	}
	return 0;

	/*        code for both increments equal to 1 */

L20:
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		i__3 = i__;
		i__4 = i__;
		z__2.r = za->r * zx[i__4].r - za->i * zx[i__4].i, z__2.i = za->r * zx[
			i__4].i + za->i * zx[i__4].r;
			z__1.r = zy[i__3].r + z__2.r, z__1.i = zy[i__3].i + z__2.i;
			zy[i__2].r = z__1.r, zy[i__2].i = z__1.i;
			/* L30: */
	}
	return 0;
} /* zaxpy_ */


static void zdotc_(doublecomplex * ret_val, integer *n, doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy)
{
	/* System generated locals */
	integer i__1, i__2;
	doublecomplex z__1, z__2, z__3;

	/* Local variables */
	integer i__, ix, iy;
	doublecomplex ztemp;


	/*     forms the dot product of a vector. */
	/*     jack dongarra, 3/11/78. */

	/* Parameter adjustments */
	--zy;
	--zx;

	/* Function Body */
	ztemp.r = 0., ztemp.i = 0.;
	ret_val->r = 0.,  ret_val->i = 0.;
	if (*n <= 0) {
		return ;
	}
	if (*incx == 1 && *incy == 1) {
		goto L20;
	}

	/*        code for unequal increments or equal increments */
	/*          not equal to 1 */

	ix = 1;
	iy = 1;
	if (*incx < 0) {
		ix = (-(*n) + 1) * *incx + 1;
	}
	if (*incy < 0) {
		iy = (-(*n) + 1) * *incy + 1;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		d_cnjg(&z__3, &zx[ix]);
		i__2 = iy;
		z__2.r = z__3.r * zy[i__2].r - z__3.i * zy[i__2].i, z__2.i = z__3.r * 
			zy[i__2].i + z__3.i * zy[i__2].r;
		z__1.r = ztemp.r + z__2.r, z__1.i = ztemp.i + z__2.i;
		ztemp.r = z__1.r, ztemp.i = z__1.i;
		ix += *incx;
		iy += *incy;
		/* L10: */
	}
	ret_val->r = ztemp.r,  ret_val->i = ztemp.i;
	return ;

	/*        code for both increments equal to 1 */

L20:
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		d_cnjg(&z__3, &zx[i__]);
		i__2 = i__;
		z__2.r = z__3.r * zy[i__2].r - z__3.i * zy[i__2].i, z__2.i = z__3.r * 
			zy[i__2].i + z__3.i * zy[i__2].r;
		z__1.r = ztemp.r + z__2.r, z__1.i = ztemp.i + z__2.i;
		ztemp.r = z__1.r, ztemp.i = z__1.i;
		/* L30: */
	}
	ret_val->r = ztemp.r,  ret_val->i = ztemp.i;
	return ;
} /* zdotc_ */


static int zdscal2_(integer *n, doublereal *da, doublecomplex *zx, integer *incx)
{
	/* System generated locals */
	integer i__1, i__2, i__3;
	doublecomplex z__1, z__2;

	/* Local variables */
	integer i__, ix;


	/*     scales a vector by a constant. */
	/*     jack dongarra, 3/11/78. */
	/*     modified 3/93 to return if incx .le. 0. */


	/* Parameter adjustments */
	--zx;

	/* Function Body */
	if (*n <= 0 || *incx <= 0) {
		return 0;
	}
	if (*incx == 1) {
		goto L20;
	}

	/*        code for increment not equal to 1 */

	ix = 1;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = ix;
		z__2.r = *da, z__2.i = 0.;
		i__3 = ix;
		z__1.r = z__2.r * zx[i__3].r - z__2.i * zx[i__3].i, z__1.i = z__2.r * 
			zx[i__3].i + z__2.i * zx[i__3].r;
		zx[i__2].r = z__1.r, zx[i__2].i = z__1.i;
		ix += *incx;
		/* L10: */
	}
	return 0;

	/*        code for increment equal to 1 */

L20:
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		z__2.r = *da, z__2.i = 0.;
		i__3 = i__;
		z__1.r = z__2.r * zx[i__3].r - z__2.i * zx[i__3].i, z__1.i = z__2.r * 
			zx[i__3].i + z__2.i * zx[i__3].r;
		zx[i__2].r = z__1.r, zx[i__2].i = z__1.i;
		/* L30: */
	}
	return 0;
} /* zdscal2_ */

///////////////////////////////////////////
//ZGPADM routine
///////////////////////////////////////////

 doublereal c_b3 = 1.;
 doublereal c_b8 = 10.;
 integer c__9 = 9;
 integer c__3 = 3;
 integer c__5 = 5;
 integer c__6 = 6;
 doublereal c_b65 = 0.;
 integer c__2 = 2;
 doublereal c_b127 = -1.;
 doublereal c_b131 = 2.;
 doublecomplex c_b172 = {0.,0.};
 doublecomplex c_b173 = {1.,0.};

static int zgpadm_(integer *ideg, integer *m, doublereal *t,
	doublecomplex *h__, integer *ldh, doublecomplex *wsp, integer *lwsp, 
	integer *ipiv, integer *iexph, integer *ns, integer *iflag)
{
	integer c__1 = 1;

	/* System generated locals */
	integer h_dim1, h_offset, i__1, i__2, i__3, i__4;
	doublereal d__1, d__2;
	doublecomplex z__1, z__2;

	doublereal hnorm;
	doublecomplex scale2;
	/* Local variables */
	integer i__, j, k;
	doublecomplex cp, cq;
	integer ip, mm, iq, ih2, iodd, iget, iput, icoef;
	doublecomplex scale;
	integer ifree, iused;

	/* -----Purpose----------------------------------------------------------| */

	/*     Computes exp(t*H), the matrix exponential of a general complex */
	/*     matrix in full, using the irreducible rational Pade approximation */
	/*     to the exponential exp(z) = r(z) = (+/-)( I + 2*(q(z)/p(z)) ), */
	/*     combined with scaling-and-squaring. */

	/* -----Arguments--------------------------------------------------------| */

	/*     ideg      : (input) the degre of the diagonal Pade to be used. */
	/*                 a value of 6 is generally satisfactory. */

	/*     m         : (input) order of H. */

	/*     H(ldh,m)  : (input) argument matrix. */

	/*     t         : (input) time-scale (can be < 0). */

	/*     wsp(lwsp) : (workspace/output) lwsp .ge. 4*m*m+ideg+1. */

	/*     ipiv(m)   : (workspace) */

	/* >>>> iexph     : (output) number such that wsp(iexph) points to exp(tH) */
	/*                 i.e., exp(tH) is located at wsp(iexph ... iexph+m*m-1) */
	/*                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */
	/*                 NOTE: if the routine was called with wsp(iptr), */
	/*                       then exp(tH) will start at wsp(iptr+iexph-1). */

	/*     ns        : (output) number of scaling-squaring used. */

	/*     iflag     : (output) exit flag. */
	/*                       0 - no problem */
	/*                      <0 - problem */

	/* ----------------------------------------------------------------------| */
	/*     Roger B. Sidje (rbs@maths.uq.edu.au) */
	/*     EXPOKIT: Software Package for Computing Matrix Exponentials. */
	/*     ACM - Transactions On Mathematical Software, 24(1):130-156, 1998 */
	/* ----------------------------------------------------------------------| */

	/* ---  check restrictions on input parameters ... */
	/* Parameter adjustments */
	--ipiv;
	h_dim1 = *ldh;
	h_offset = 1 + h_dim1;
	h__ -= h_offset;
	--wsp;

	/* Function Body */
	mm = *m * *m;
	*iflag = 0;
	if (*ldh < *m) {
		*iflag = -1;
	}
	if (*lwsp < (mm << 2) + *ideg + 1) {
		*iflag = -2;
	}
	if (*iflag != 0) {
        //s_stop("bad sizes (in input of ZGPADM)", (ftnlen)30);
        return 1;
	}

	/* ---  initialise pointers ... */

	icoef = 1;
	ih2 = icoef + (*ideg + 1);
	ip = ih2 + mm;
	iq = ip + mm;
	ifree = iq + mm;

	/* ---  scaling: seek ns such that ||t*H/2^ns|| < 1/2; */
	/*     and set scale = t/2^ns ... */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		wsp[i__2].r = 0., wsp[i__2].i = 0.;
	}
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__;
			i__4 = i__;
			d__1 = z_abs(&h__[i__ + j * h_dim1]);
			z__1.r = wsp[i__4].r + d__1, z__1.i = wsp[i__4].i;
			wsp[i__3].r = z__1.r, wsp[i__3].i = z__1.i;
		}
	}
	hnorm = 0.;
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
		/* Computing MAX */
		i__2 = i__;
		d__1 = hnorm, d__2 = wsp[i__2].r;
		hnorm = max(d__1,d__2);
	}
	hnorm = (d__1 = *t * hnorm, fabs(d__1));
	if (hnorm == 0.) {
//		printf("d__1: %3.1f, *t == %f",d__1,*t,hnorm,abs(d__1));
        //s_stop("Error - null H in input of ZGPADM.", (ftnlen)34);
        *iflag = -3;
        return 0;
	}
	/* Computing MAX */
	i__1 = 0, i__2 = (integer) (log(hnorm) / log(2.)) + 2;
	*ns = max(i__1,i__2);
	d__1 = *t / (doublereal) pow_ii(&c__2, ns);
	z__1.r = d__1, z__1.i = 0.;
	scale.r = z__1.r, scale.i = z__1.i;
	z__1.r = scale.r * scale.r - scale.i * scale.i, z__1.i = scale.r * 
		scale.i + scale.i * scale.r;
	scale2.r = z__1.r, scale2.i = z__1.i;

	/* ---  compute Pade coefficients ... */

	i__ = *ideg + 1;
	j = (*ideg << 1) + 1;
	i__1 = icoef;
	wsp[i__1].r = 1., wsp[i__1].i = 0.;
	i__1 = *ideg;
	for (k = 1; k <= i__1; ++k) {
		i__2 = icoef + k;
		i__3 = icoef + k - 1;
		d__1 = (doublereal) (i__ - k);
		z__2.r = d__1 * wsp[i__3].r, z__2.i = d__1 * wsp[i__3].i;
		d__2 = (doublereal) (k * (j - k));
		z__1.r = z__2.r / d__2, z__1.i = z__2.i / d__2;
		wsp[i__2].r = z__1.r, wsp[i__2].i = z__1.i;
	}

	/* ---  H2 = scale2*H*H ... */

	zgemm_("n", "n", m, m, m, &scale2, &h__[h_offset], ldh, &h__[h_offset], 
		ldh, &c_b172, &wsp[ih2], m, (ftnlen)1, (ftnlen)1);

	/* ---  initialise p (numerator) and q (denominator) ... */

	i__1 = icoef + *ideg - 1;
	cp.r = wsp[i__1].r, cp.i = wsp[i__1].i;
	i__1 = icoef + *ideg;
	cq.r = wsp[i__1].r, cq.i = wsp[i__1].i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = ip + (j - 1) * *m + i__ - 1;
			wsp[i__3].r = 0., wsp[i__3].i = 0.;
			i__3 = iq + (j - 1) * *m + i__ - 1;
			wsp[i__3].r = 0., wsp[i__3].i = 0.;
		}
		i__2 = ip + (j - 1) * (*m + 1);
		wsp[i__2].r = cp.r, wsp[i__2].i = cp.i;
		i__2 = iq + (j - 1) * (*m + 1);
		wsp[i__2].r = cq.r, wsp[i__2].i = cq.i;
	}

	/* ---  Apply Horner rule ... */

	iodd = 1;
	k = *ideg - 1;
L100:
	iused = iodd * iq + (1 - iodd) * ip;
	zgemm_("n", "n", m, m, m, &c_b173, &wsp[iused], m, &wsp[ih2], m, &c_b172, 
		&wsp[ifree], m, (ftnlen)1, (ftnlen)1);
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
		i__2 = ifree + (j - 1) * (*m + 1);
		i__3 = ifree + (j - 1) * (*m + 1);
		i__4 = icoef + k - 1;
		z__1.r = wsp[i__3].r + wsp[i__4].r, z__1.i = wsp[i__3].i + wsp[i__4]
		.i;
		wsp[i__2].r = z__1.r, wsp[i__2].i = z__1.i;
	}
	ip = (1 - iodd) * ifree + iodd * ip;
	iq = iodd * ifree + (1 - iodd) * iq;
	ifree = iused;
	iodd = 1 - iodd;
	--k;
	if (k > 0) {
		goto L100;
	}

	/* ---  Obtain (+/-)(I + 2*(p\q)) ... */

	if (iodd != 0) {
		zgemm_("n", "n", m, m, m, &scale, &wsp[iq], m, &h__[h_offset], ldh, &
			c_b172, &wsp[ifree], m, (ftnlen)1, (ftnlen)1);
		iq = ifree;
	} else {
		zgemm_("n", "n", m, m, m, &scale, &wsp[ip], m, &h__[h_offset], ldh, &
			c_b172, &wsp[ifree], m, (ftnlen)1, (ftnlen)1);
		ip = ifree;
	}
	z__1.r = -1., z__1.i = -0.;
	zaxpy_(&mm, &z__1, &wsp[ip], &c__1, &wsp[iq], &c__1);
	zgesv_(m, m, &wsp[iq], m, &ipiv[1], &wsp[ip], m, iflag);
	if (*iflag != 0) {
        //s_stop("Problem in ZGESV (within ZGPADM)", (ftnlen)32);
        return 1;
	}
	zdscal2_(&mm, &c_b131, &wsp[ip], &c__1);
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
		i__2 = ip + (j - 1) * (*m + 1);
		i__3 = ip + (j - 1) * (*m + 1);
		z__1.r = wsp[i__3].r + 1., z__1.i = wsp[i__3].i + 0.;
		wsp[i__2].r = z__1.r, wsp[i__2].i = z__1.i;
	}
	iput = ip;
	if (*ns == 0 && iodd != 0) {
		zdscal2_(&mm, &c_b127, &wsp[ip], &c__1);
		goto L200;
	}

	/* --   squaring : exp(t*H) = (exp(t*H))^(2^ns) ... */

	iodd = 1;
	i__1 = *ns;
	for (k = 1; k <= i__1; ++k) {
		iget = iodd * ip + (1 - iodd) * iq;
		iput = (1 - iodd) * ip + iodd * iq;
		zgemm_("n", "n", m, m, m, &c_b173, &wsp[iget], m, &wsp[iget], m, &
			c_b172, &wsp[iput], m, (ftnlen)1, (ftnlen)1);
		iodd = 1 - iodd;
	}
L200:
	*iexph = iput;
	return 0;
} /* zgpadm_ */


static void matrixMult4Real(doublereal *M1, doublereal *M2, doublereal *R)
{
        doublereal Rtmp[16];
        int i=0;
        Rtmp[0] = M1[0]*M2[0]+M1[1]*M2[4]+M1[2]*M2[8]+M1[3]*M2[12];
        Rtmp[1] = M1[0]*M2[1]+M1[1]*M2[5]+M1[2]*M2[9]+M1[3]*M2[13];
        Rtmp[2] = M1[0]*M2[2]+M1[1]*M2[6]+M1[2]*M2[10]+M1[3]*M2[14];
        Rtmp[3] = M1[0]*M2[3]+M1[1]*M2[7]+M1[2]*M2[11]+M1[3]*M2[15];

        Rtmp[4] = M1[4]*M2[0]+M1[5]*M2[4]+M1[6]*M2[8]+M1[7]*M2[12];
        Rtmp[5] = M1[4]*M2[1]+M1[5]*M2[5]+M1[6]*M2[9]+M1[7]*M2[13];
        Rtmp[6] = M1[4]*M2[2]+M1[5]*M2[6]+M1[6]*M2[10]+M1[7]*M2[14];
        Rtmp[7] = M1[4]*M2[3]+M1[5]*M2[7]+M1[6]*M2[11]+M1[7]*M2[15];

        Rtmp[8]  = M1[8]*M2[0]+M1[9]*M2[4]+M1[10]*M2[8]+M1[11]*M2[12];
        Rtmp[9]  = M1[8]*M2[1]+M1[9]*M2[5]+M1[10]*M2[9]+M1[11]*M2[13];
        Rtmp[10] = M1[8]*M2[2]+M1[9]*M2[6]+M1[10]*M2[10]+M1[11]*M2[14];
        Rtmp[11] = M1[8]*M2[3]+M1[9]*M2[7]+M1[10]*M2[11]+M1[11]*M2[15];

        Rtmp[12] = M1[12]*M2[0]+M1[13]*M2[4]+M1[14]*M2[8]+M1[15]*M2[12];
        Rtmp[13] = M1[12]*M2[1]+M1[13]*M2[5]+M1[14]*M2[9]+M1[15]*M2[13];
        Rtmp[14] = M1[12]*M2[2]+M1[13]*M2[6]+M1[14]*M2[10]+M1[15]*M2[14];
        Rtmp[15] = M1[12]*M2[3]+M1[13]*M2[7]+M1[14]*M2[11]+M1[15]*M2[15];
        for (i = 0; i < 16; i++) R[i] = Rtmp[i];
}
/*
 void normalizeMat4Real(doublereal *T) {
   doublereal len = 0.0f;
   len = sqrtf(T[0]*T[0]+T[4]*T[4]+T[8]*T[8]);
   T[0] /= len;  T[4] /= len; T[8] /= len;
   len = sqrtf(T[1]*T[1]+T[5]*T[5]+T[9]*T[9]);
   T[1] /= len;  T[5] /= len; T[9] /= len;
   len = sqrtf(T[2]*T[2]+T[6]*T[6]+T[10]*T[10]);
   T[2] /= len;  T[6] /= len; T[10] /= len;
}*/

static void normalizeVec3Real(doublereal *v) {
    doublereal len = sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    v[0] /= len;  v[1] /= len; v[2] /= len;
}

static void crossproductReal(doublereal *u, doublereal *v, doublereal *w) {
    w[0] = u[1]*v[2]-u[2]*v[1];
    w[1] = -(u[0]*v[2]-u[2]*v[0]);
    w[2] = u[0]*v[1]-u[1]*v[0];
}

static  void normalizeMat4Real(doublereal *T) {
    // extract column vectors SO3
    doublereal u[3],v[3],w[3];
    u[0] = T[0]; u[1] = T[4]; u[2] = T[8];
    v[0] = T[1]; v[1] = T[5]; v[2] = T[9];
    w[0] = T[2]; w[1] = T[6]; w[2] = T[10];

    normalizeVec3Real(w);
    crossproductReal(v,w,u); normalizeVec3Real(u);
    crossproductReal(w,u,v);

    // store column vectors SO3
    T[0] = u[0]; T[4] = u[1]; T[8]  = u[2];
    T[1] = v[0]; T[5] = v[1]; T[9]  = v[2];
    T[2] = w[0]; T[6] = w[1]; T[10] = w[2];
 }

static void normalizeMat4RealSafe(doublereal *T) {
    // extract column vectors SO3
    doublereal u[3],v[3],w[3];
    u[0] = T[0]; u[1] = T[4]; u[2] = T[8];
    v[0] = T[1]; v[1] = T[5]; v[2] = T[9];
    w[0] = T[2]; w[1] = T[6]; w[2] = T[10];

    doublereal lenU = sqrtf(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    doublereal lenV = sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    doublereal lenW = sqrtf(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);


    if (lenW > 1e-8f) { w[0] /= lenW; w[1] /= lenW; w[2] /= lenW; }
    else { w[0] = 0; w[1] = 0; w[2] = 1; }
    crossproductReal(v,w,u);
    if (lenU > 1e-8f) { u[0] /= lenU; u[1] /= lenU; u[2] /= lenU; }
    else { u[0] = 1; u[1] = 0; u[2] = 0; }
    crossproductReal(w,u,v);

    // store column vectors SO3
    T[0] = u[0]; T[4] = u[1]; T[8]  = u[2];
    T[1] = v[0]; T[5] = v[1]; T[9]  = v[2];
    T[2] = w[0]; T[6] = w[1]; T[10] = w[2];
 }

///////////////////////////////////////////
// MAIN CODE
///////////////////////////////////////////
#define ELT(numRows,i,j) (((long)j)*((long)numRows)+((long)i))
#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)>(y) ? (x) : (y))

static void expmDoubleReal(doublereal *mtx, float *TF, doublereal scale)
{
    integer ideg = 6; // pad√© order number
    integer m = 4;
    integer M = 4;
    integer ldh = M;
    integer lwsp = 4*16+6+1;
    doublecomplex wsp[4*16+6+1];// = new doublecomplex[lwsp];
    integer ipiv[4];// = new integer[M];
    integer iexph = 0;
    integer ns = 0;
    integer flag = 0;
    doublereal t = 1.0;//doublereal(tt);
    doublereal Tinc[16];
    doublereal T[4*4];
    int i=0;

    int mm = m*m;
    doublereal maxEntry = 0;
    for(i = 0; i < mm; i++) {
        T[i] = TF[i];
        maxEntry = MAX(maxEntry, fabs(mtx[i]));
    }

    doublecomplex h[16];
    for (i = 0; i < mm; i++) {
        h[i].r = mtx[i];
        h[i].i = 0;
    }
    if (maxEntry != 0) {
        zgpadm_(&ideg, &M, &t, h, &ldh, &wsp[0], &lwsp, ipiv, &iexph, &ns, &flag);
        if (flag != 0) {
            // problem: set identity motion to output
            for (i = 0; i < m*m; i++) Tinc[i] = 0;
            for (i = 0; i < m; i++) Tinc[i+i*m] = 1;
        } else {
            // no problems: set estimated motion to output
            for (i = 0; i < m*m; i++) {
                Tinc[i] = wsp[iexph-1+i].r;
            }
        }

        normalizeMat4RealSafe(Tinc); Tinc[3] *= scale; Tinc[7] *= scale; Tinc[11] *= scale;
        //memcpy(output, &wsp[iexph-1], sizeof(double) * m * m);
    } else {
        // set result to identity matrix
        for (i = 0; i < mm; i++) Tinc[i] = 0.0;
        for(i = 0; i < m; i++) Tinc[ELT(m,i,i)] = 1.0;
    }
    // transformation is concatenated to right due to inverse compositional trick!
    // thus reference warp increment is known and should be inverted and composited in the beginning of total point transform
    // ref -> warped current -> current (point transformations: inv(T(x)) -> baseT )
    matrixMult4Real(T,Tinc,T); normalizeMat4RealSafe(T);

    for(i = 0; i < mm; i++) {
        TF[i] = (float)T[i];
    }
}

static void generateTSpace(float *x, doublereal *A) {
    A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
    A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
    A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
    A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;
}


void expm(float *x, float *motionT, float scaleOut)
{
    assert(x != NULL && motionT != NULL);
    doublereal A[16];
    generateTSpace(x,A);
    expmDoubleReal(A,motionT,(doublereal)scaleOut);
}
