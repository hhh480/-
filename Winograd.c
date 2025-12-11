 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <math.h>
 #include <string.h>
 
 
 static double* alloc_mat(int n) {
     double *p = (double*) malloc((size_t)n * n * sizeof(double));
     if (!p) { fprintf(stderr, "alloc fail\n"); exit(1); }
     return p;
 }
 
 static void free_mat(double *p) { free(p); }
 
 static inline double get(const double *A, int n, int i, int j) {
     return A[(size_t)i * n + j];
 }
 static inline void set(double *A, int n, int i, int j, double v) {
     A[(size_t)i * n + j] = v;
 }
 
 /* 随机填充矩阵 */
 static void rand_fill(double *A, int n) {
     for (size_t i = 0; i < (size_t)n * n; ++i) A[i] = ((double)rand() / RAND_MAX) - 0.5;
 }
 
 /* 置零 */
 static void zero_mat(double *A, int n) {
     memset(A, 0, (size_t)n * n * sizeof(double));
 }
 
 /* C = A (copy) */
 static void copy_mat(const double *A, double *C, int n) {
     memcpy(C, A, (size_t)n * n * sizeof(double));
 }
 
 /* 比较是否近似相等 */
 static int approx_equal(const double *A, const double *B, int n, double tol) {
     for (size_t i = 0; i < (size_t)n * n; ++i) {
         double d = A[i] - B[i];
         if (fabs(d) > tol) return 0;
     }
     return 1;
 }
 
 /* 基础矩阵乘法：C += A * B */
 static void matmul_basic_add(double *C, const double *A, const double *B, int n) {
     for (int i = 0; i < n; ++i) {
         for (int k = 0; k < n; ++k) {
             double a = A[(size_t)i * n + k];
             const double *brow = B + (size_t)k * n;
             double *crow = C + (size_t)i * n;
             for (int j = 0; j < n; ++j) {
                 crow[j] += a * brow[j];
             }
         }
     }
 }
 
 /* 基础矩阵乘法：C = A * B */
 static void matmul_basic(double *C, const double *A, const double *B, int n) {
     zero_mat(C, n);
     matmul_basic_add(C, A, B, n);
 }
 
 /* 复制子块到连续 buffer: buf (p x q) = A[r0:r0+p, c0:c0+q] */
 static void copy_subblock(const double *A, int n, int r0, int c0, double *buf, int p, int q) {
     for (int i = 0; i < p; ++i) {
         memcpy(buf + (size_t)i * q, A + (size_t)(r0 + i) * n + c0, q * sizeof(double));
     }
 }
 
 /* 将连续 buffer 存回到大矩阵子块位置 */
 static void store_subblock(double *C, int n, int r0, int c0, const double *buf, int p, int q) {
     for (int i = 0; i < p; ++i) {
         memcpy(C + (size_t)(r0 + i) * n + c0, buf + (size_t)i * q, q * sizeof(double));
     }
 }
 
 /* 子块加法 bufC (p x q) += bufA (p x q) * alpha (in-place) */
 static void add_block_inplace(double *Cbuf, const double *Mbuf, int p, int q, double alpha) {
     size_t elems = (size_t)p * q;
     for (size_t i = 0; i < elems; ++i) Cbuf[i] += alpha * Mbuf[i];
 }
 
 /* bufC = bufA +/- bufB (p x q) */
 static void block_add(const double *A, const double *B, double *C, int p, int q, int sign) {
     size_t elems = (size_t)p * q;
     if (sign >= 0) {
         for (size_t i = 0; i < elems; ++i) C[i] = A[i] + B[i];
     } else {
         for (size_t i = 0; i < elems; ++i) C[i] = A[i] - B[i];
     }
 }
 
 /* pad square matrix up to new_n (new_n >= n), produced contiguous buffer */
 static double* pad_square(const double *A, int n, int new_n) {
     double *P = alloc_mat(new_n);
     zero_mat(P, new_n);
     for (int i = 0; i < n; ++i) {
         memcpy(P + (size_t)i * new_n, A + (size_t)i * n, n * sizeof(double));
     }
     return P;
 }
 
 
 /* forward declaration */
 static void winograd_rec_buf(double *C, const double *A, const double *B, int n, int leaf);
 
 /* winograd wrapper: handles odd by padding to n+1 */
 static double* winograd_rec_alloc(const double *A, const double *B, int n, int leaf) {
     if (n <= leaf) {
         double *C = alloc_mat(n);
         matmul_basic(C, A, B, n);
         return C;
     }
 
     if (n % 2 == 1) {
         int n2 = n + 1;
         double *Ap = pad_square(A, n, n2);
         double *Bp = pad_square(B, n, n2);
         double *Cp = winograd_rec_alloc(Ap, Bp, n2, leaf);
         double *C = alloc_mat(n);
         // copy top-left n x n
         for (int i = 0; i < n; ++i) memcpy(C + (size_t)i * n, Cp + (size_t)i * n2, n * sizeof(double));
         free_mat(Ap); free_mat(Bp); free_mat(Cp);
         return C;
     } else {
         // even n
         int m = n / 2;
         // allocate subblocks and M buffers
         double *A11 = alloc_mat(m), *A12 = alloc_mat(m), *A21 = alloc_mat(m), *A22 = alloc_mat(m);
         double *B11 = alloc_mat(m), *B12 = alloc_mat(m), *B21 = alloc_mat(m), *B22 = alloc_mat(m);
         copy_subblock(A, n, 0, 0, A11, m, m);
         copy_subblock(A, n, 0, m, A12, m, m);
         copy_subblock(A, n, m, 0, A21, m, m);
         copy_subblock(A, n, m, m, A22, m, m);
         copy_subblock(B, n, 0, 0, B11, m, m);
         copy_subblock(B, n, 0, m, B12, m, m);
         copy_subblock(B, n, m, 0, B21, m, m);
         copy_subblock(B, n, m, m, B22, m, m);
 
         // allocate M1..M7
         double *M1 = alloc_mat(m), *M2 = alloc_mat(m), *M3 = alloc_mat(m), *M4 = alloc_mat(m),
                *M5 = alloc_mat(m), *M6 = alloc_mat(m), *M7 = alloc_mat(m);
         double *T1 = alloc_mat(m), *T2 = alloc_mat(m); // temps for linear combos
 
         // M1 = (A11 + A22) * (B11 + B22)
         block_add(A11, A22, T1, m, m, +1);
         block_add(B11, B22, T2, m, m, +1);
         double *tmp = winograd_rec_alloc(T1, T2, m, leaf);
         copy_mat(tmp, M1, m); free_mat(tmp);
 
         // M2 = (A21 + A22) * B11
         block_add(A21, A22, T1, m, m, +1);
         tmp = winograd_rec_alloc(T1, B11, m, leaf);
         copy_mat(tmp, M2, m); free_mat(tmp);
 
         // M3 = A11 * (B12 - B22)
         block_add(B12, B22, T2, m, m, -1);
         tmp = winograd_rec_alloc(A11, T2, m, leaf);
         copy_mat(tmp, M3, m); free_mat(tmp);
 
         // M4 = A22 * (B21 - B11)
         block_add(B21, B11, T2, m, m, -1);
         tmp = winograd_rec_alloc(A22, T2, m, leaf);
         copy_mat(tmp, M4, m); free_mat(tmp);
 
         // M5 = (A11 + A12) * B22
         block_add(A11, A12, T1, m, m, +1);
         tmp = winograd_rec_alloc(T1, B22, m, leaf);
         copy_mat(tmp, M5, m); free_mat(tmp);
 
         // M6 = (A21 - A11) * (B11 + B12)
         block_add(A21, A11, T1, m, m, -1);
         block_add(B11, B12, T2, m, m, +1);
         tmp = winograd_rec_alloc(T1, T2, m, leaf);
         copy_mat(tmp, M6, m); free_mat(tmp);
 
         // M7 = (A12 - A22) * (B21 + B22)
         block_add(A12, A22, T1, m, m, -1);
         block_add(B21, B22, T2, m, m, +1);
         tmp = winograd_rec_alloc(T1, T2, m, leaf);
         copy_mat(tmp, M7, m); free_mat(tmp);
 
         // assemble C
         double *C = alloc_mat(n);
         zero_mat(C, n);
         // C11 = M1 + M4 - M5 + M7
         add_block_inplace(C + 0 * n + 0, M1, m, m, 1.0); // careful: not correct pointer arithmetic; use store_subblock instead
         // since easier/clearer: build C11..C22 buffers then store
         double *C11 = alloc_mat(m), *C12 = alloc_mat(m), *C21 = alloc_mat(m), *C22 = alloc_mat(m);
         zero_mat(C11, m); zero_mat(C12, m); zero_mat(C21, m); zero_mat(C22, m);
 
         add_block_inplace(C11, M1, m, m, 1.0);
         add_block_inplace(C11, M4, m, m, 1.0);
         add_block_inplace(C11, M5, m, m, -1.0);
         add_block_inplace(C11, M7, m, m, 1.0);
 
         add_block_inplace(C12, M3, m, m, 1.0);
         add_block_inplace(C12, M5, m, m, 1.0);
 
         add_block_inplace(C21, M2, m, m, 1.0);
         add_block_inplace(C21, M4, m, m, 1.0);
 
         add_block_inplace(C22, M1, m, m, 1.0);
         add_block_inplace(C22, M2, m, m, -1.0);
         add_block_inplace(C22, M3, m, m, 1.0);
         add_block_inplace(C22, M6, m, m, 1.0);
 
         // store subblocks into C
         store_subblock(C, n, 0, 0, C11, m, m);
         store_subblock(C, n, 0, m, C12, m, m);
         store_subblock(C, n, m, 0, C21, m, m);
         store_subblock(C, n, m, m, C22, m, m);
 
         // free temporaries
         free_mat(A11); free_mat(A12); free_mat(A21); free_mat(A22);
         free_mat(B11); free_mat(B12); free_mat(B21); free_mat(B22);
         free_mat(M1); free_mat(M2); free_mat(M3); free_mat(M4);
         free_mat(M5); free_mat(M6); free_mat(M7);
         free_mat(T1); free_mat(T2);
         free_mat(C11); free_mat(C12); free_mat(C21); free_mat(C22);
 
         return C;
     }
 }
 
 /* Helper wrapper to produce an alloc'd C for A*B using recursive Winograd */
 static double* winograd_mul(const double *A, const double *B, int n, int leaf) {
     return winograd_rec_alloc(A, B, n, leaf);
 }
 
 static double* cross_mul_alloc(const double *A, const double *B, int n, int leaf) {
     if (n % 2 == 0) {
         return winograd_mul(A, B, n, leaf);
     }
     if (n == 1) {
         double *C = alloc_mat(1);
         C[0] = A[0] * B[0];
         return C;
     }
     int m = n / 2;
     int c = m; // middle index
 
     // allocate result C
     double *C = alloc_mat(n);
     zero_mat(C, n);
    
     // allocate buffers
     double *A00 = alloc_mat(m), *A01 = (double*)malloc((size_t)m * 1 * sizeof(double)), *A02 = alloc_mat(m);
     double *A10 = (double*)malloc((size_t)1 * m * sizeof(double)), *A11 = (double*)malloc(sizeof(double)), *A12 = (double*)malloc((size_t)1 * m * sizeof(double));
     double *A20 = alloc_mat(m), *A21 = (double*)malloc((size_t)m * 1 * sizeof(double)), *A22 = alloc_mat(m);
 
     double *B00 = alloc_mat(m), *B01 = (double*)malloc((size_t)m * 1 * sizeof(double)), *B02 = alloc_mat(m);
     double *B10 = (double*)malloc((size_t)1 * m * sizeof(double)), *B11 = (double*)malloc(sizeof(double)), *B12 = (double*)malloc((size_t)1 * m * sizeof(double));
     double *B20 = alloc_mat(m), *B21 = (double*)malloc((size_t)m * 1 * sizeof(double)), *B22 = alloc_mat(m);
 
     // copy subblocks
     copy_subblock(A, n, 0, 0, A00, m, m);
     copy_subblock(A, n, 0, c, A01, m, 1);
     copy_subblock(A, n, 0, c+1, A02, m, m);
 
     copy_subblock(A, n, c, 0, A10, 1, m);
     A11[0] = get(A, n, c, c);
     copy_subblock(A, n, c, c+1, A12, 1, m);
 
     copy_subblock(A, n, c+1, 0, A20, m, m);
     copy_subblock(A, n, c+1, c, A21, m, 1);
     copy_subblock(A, n, c+1, c+1, A22, m, m);
 
     copy_subblock(B, n, 0, 0, B00, m, m);
     copy_subblock(B, n, 0, c, B01, m, 1);
     copy_subblock(B, n, 0, c+1, B02, m, m);
 
     copy_subblock(B, n, c, 0, B10, 1, m);
     B11[0] = get(B, n, c, c);
     copy_subblock(B, n, c, c+1, B12, 1, m);
 
     copy_subblock(B, n, c+1, 0, B20, m, m);
     copy_subblock(B, n, c+1, c, B21, m, 1);
     copy_subblock(B, n, c+1, c+1, B22, m, m);
 
 
     double *T = alloc_mat(m); zero_mat(T, m);
     double *tmp;
     // A00*B00
     tmp = winograd_mul(A00, B00, m, leaf);
     add_block_inplace(T, tmp, m, m, 1.0);
     free_mat(tmp);
     // A01 (m x1) * B10 (1 x m) => m x m
     {
         double *M = alloc_mat(m);
         zero_mat(M, m);
         // M = A01 * B10
         for (int i = 0; i < m; ++i) {
             double a = A01[i];
             for (int j = 0; j < m; ++j) M[i * m + j] += a * B10[j];
         }
         add_block_inplace(T, M, m, m, 1.0);
         free_mat(M);
     }
     // A02*B20
     tmp = winograd_mul(A02, B20, m, leaf);
     add_block_inplace(T, tmp, m, m, 1.0);
     free_mat(tmp);
     store_subblock(C, n, 0, 0, T, m, m);
     free_mat(T);
 
     // C01 = A00*B01 (m x 1) + A01*B11 (m x1) + A02*B21 (m x1)
     {
         double *C01 = (double*)calloc((size_t)m * 1, sizeof(double));
         // A00*B01 : A00(mxm) * B01(mx1) -> mx1
         {
             for (int i = 0; i < m; ++i) {
                 double sum = 0.0;
                 for (int k = 0; k < m; ++k) sum += A00[i * m + k] * B01[k];
                 C01[i] += sum;
             }
         }
         // A01*B11
         for (int i = 0; i < m; ++i) C01[i] += A01[i] * B11[0];
         // A02*B21 : A02(mxm) * B21(mx1)
         {
             for (int i = 0; i < m; ++i) {
                 double sum = 0.0;
                 for (int k = 0; k < m; ++k) sum += A02[i * m + k] * B21[k];
                 C01[i] += sum;
             }
         }
         // store
         for (int i = 0; i < m; ++i) set(C, n, i, c, C01[i]);
         free(C01);
     }
 
     // C02 = A00*B02 + A01*B12 + A02*B22  (mxm)
     {
         double *T2 = alloc_mat(m); zero_mat(T2, m);
         tmp = winograd_mul(A00, B02, m, leaf);
         add_block_inplace(T2, tmp, m, m, 1.0); free_mat(tmp);
         // A01( m x 1 ) * B12 (1 x m) => m x m
         {
             double *M = alloc_mat(m); zero_mat(M, m);
             for (int i = 0; i < m; ++i) {
                 double a = A01[i];
                 for (int j = 0; j < m; ++j) M[i*m + j] += a * B12[j];
             }
             add_block_inplace(T2, M, m, m, 1.0); free_mat(M);
         }
         tmp = winograd_mul(A02, B22, m, leaf);
         add_block_inplace(T2, tmp, m, m, 1.0); free_mat(tmp);
         store_subblock(C, n, 0, c+1, T2, m, m);
         free_mat(T2);
     }
 
     // C10 = A10*B00 + A11*B10 + A12*B20  (1 x m)
     {
         double *C10 = (double*)calloc((size_t)1 * m, sizeof(double));
         // A10(1 x m) * B00(m x m)
         for (int j = 0; j < m; ++j) {
             double sum = 0.0;
             for (int k = 0; k < m; ++k) sum += A10[k] * B00[k * m + j];
             C10[j] += sum;
         }
         // A11 * B10 (1x1 * 1xm) => 1xm
         for (int j = 0; j < m; ++j) C10[j] += A11[0] * B10[j];
         // A12(1xm) * B20(mxm) -> 1xm
         for (int j = 0; j < m; ++j) {
             double sum = 0.0;
             for (int k = 0; k < m; ++k) sum += A12[k] * B20[k * m + j];
             C10[j] += sum;
         }
         for (int j = 0; j < m; ++j) set(C, n, c, j, C10[j]);
         free(C10);
     }
 
     // C11 = A10*B01 + A11*B11 + A12*B21  (1x1)
     {
         double v = 0.0;
         // A10(1xm)*B01(mx1)
         for (int k = 0; k < m; ++k) v += A10[k] * B01[k];
         v += A11[0] * B11[0];
         for (int k = 0; k < m; ++k) v += A12[k] * B21[k];
         set(C, n, c, c, v);
     }
 
     // C12 = A10*B02 + A11*B12 + A12*B22 (1 x m)
     {
         double *C12 = (double*)calloc((size_t)1 * m, sizeof(double));
         // A10 * B02
         for (int j = 0; j < m; ++j) {
             double sum = 0.0;
             for (int k = 0; k < m; ++k) sum += A10[k] * B02[k * m + j];
             C12[j] += sum;
         }
         // A11 * B12 (1x1 * 1xm)
         for (int j = 0; j < m; ++j) C12[j] += A11[0] * B12[j];
         // A12 * B22 (1xm * mxm)
         for (int j = 0; j < m; ++j) {
             double sum = 0.0;
             for (int k = 0; k < m; ++k) sum += A12[k] * B22[k * m + j];
             C12[j] += sum;
         }
         for (int j = 0; j < m; ++j) set(C, n, c, c+1+j, C12[j]);
         free(C12);
     }
 
     // C20 = A20*B00 + A21*B10 + A22*B20  (mxm)
     {
         double *T3 = alloc_mat(m); zero_mat(T3, m);
         tmp = winograd_mul(A20, B00, m, leaf);
         add_block_inplace(T3, tmp, m, m, 1.0); free_mat(tmp);
         // A21 * B10 -> m x m (A21 mx1, B10 1xm)
         {
             double *M = alloc_mat(m); zero_mat(M, m);
             for (int i = 0; i < m; ++i) {
                 double a = A21[i];
                 for (int j = 0; j < m; ++j) M[i*m + j] += a * B10[j];
             }
             add_block_inplace(T3, M, m, m, 1.0); free_mat(M);
         }
         tmp = winograd_mul(A22, B20, m, leaf); add_block_inplace(T3, tmp, m, m, 1.0); free_mat(tmp);
         store_subblock(C, n, c+1, 0, T3, m, m); free_mat(T3);
     }
 
     // C21 = A20*B01 + A21*B11 + A22*B21  (mx1)
     {
         double *C21 = (double*)calloc((size_t)m * 1, sizeof(double));
         // A20 * B01
         for (int i = 0; i < m; ++i) {
             double sum = 0.0;
             for (int k = 0; k < m; ++k) sum += A20[i*m + k] * B01[k];
             C21[i] += sum;
         }
         // A21 * B11
         for (int i = 0; i < m; ++i) C21[i] += A21[i] * B11[0];
         // A22 * B21
         for (int i = 0; i < m; ++i) {
             double sum = 0.0;
             for (int k = 0; k < m; ++k) sum += A22[i*m + k] * B21[k];
             C21[i] += sum;
         }
         for (int i = 0; i < m; ++i) set(C, n, c+1 + i, c, C21[i]);
         free(C21);
     }
 
     // C22 = A20*B02 + A21*B12 + A22*B22  (mxm)
     {
         double *T4 = alloc_mat(m); zero_mat(T4, m);
         tmp = winograd_mul(A20, B02, m, leaf); add_block_inplace(T4, tmp, m, m, 1.0); free_mat(tmp);
         // A21 * B12 -> m x m
         {
             double *M = alloc_mat(m); zero_mat(M, m);
             for (int i = 0; i < m; ++i) {
                 double a = A21[i];
                 for (int j = 0; j < m; ++j) M[i*m + j] += a * B12[j];
             }
             add_block_inplace(T4, M, m, m, 1.0); free_mat(M);
         }
         tmp = winograd_mul(A22, B22, m, leaf); add_block_inplace(T4, tmp, m, m, 1.0); free_mat(tmp);
         store_subblock(C, n, c+1, c+1, T4, m, m); free_mat(T4);
     }
 
     // free all small buffers
     free_mat(A00); free(A01); free_mat(A02);
     free(A10); free(A11); free(A12);
     free_mat(A20); free(A21); free_mat(A22);
     free_mat(B00); free(B01); free_mat(B02);
     free(B10); free(B11); free(B12);
     free_mat(B20); free(B21); free_mat(B22);
 
     return C;
 }
 
 /* ---------- 测试 / 测时工具 ---------- */
 
 static double now_sec() {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return ts.tv_sec + ts.tv_nsec * 1e-9;
 }
 
 /* 普通乘法平均时间（重复 repeats 次） */
 static double avg_time_basic(int n, int repeats) {
     double *A = alloc_mat(n), *B = alloc_mat(n), *C = alloc_mat(n);
     rand_fill(A, n); rand_fill(B, n);
     double t0 = now_sec();
     for (int r = 0; r < repeats; ++r) {
         zero_mat(C, n);
         matmul_basic_add(C, A, B, n);
     }
     double t1 = now_sec();
     free_mat(A); free_mat(B); free_mat(C);
     return (t1 - t0) / repeats;
 }
 
 /* W 方法平均时间（重复 repeats 次） */
 static double avg_time_W_method(int n, int repeats, int leaf) {
     double *A = alloc_mat(n), *B = alloc_mat(n);
     rand_fill(A, n); rand_fill(B, n);
     double t0 = now_sec();
     for (int r = 0; r < repeats; ++r) {
         double *C = cross_mul_alloc(A, B, n, leaf);
         free_mat(C);
     }
     double t1 = now_sec();
     free_mat(A); free_mat(B);
     return (t1 - t0) / repeats;
 }
 
 /* correctness check single pair */
 static int check_correctness(int n, int leaf) {
     double *A = alloc_mat(n), *B = alloc_mat(n), *C1 = alloc_mat(n), *C2;
     rand_fill(A, n); rand_fill(B, n);
     matmul_basic(C1, A, B, n);
     C2 = cross_mul_alloc(A, B, n, leaf);
     int ok = approx_equal(C1, C2, n, 1e-6);
     free_mat(A); free_mat(B); free_mat(C1); free_mat(C2);
     return ok;
 }
 
 /* ---------- main ---------- */
 
 int main() {
     srand((unsigned)time(NULL));
     int n, repeats, leaf;
     printf("输入奇数 n (例如 63, 127, 1029): ");
     if (scanf("%d", &n) != 1) return 0;
     if (n <= 0) return 0;
     if (n % 2 == 0) { printf("必须输入奇数！\n"); return 0; }
     printf("重复次数 repeats (例如 1~5 建议): ");
     if (scanf("%d", &repeats) != 1) return 0;
     printf("叶子大小 leaf (比如 32 或 64，影响递归停止): ");
     if (scanf("%d", &leaf) != 1) return 0;
 
     printf("\n正在进行 %d 次平均时间测量（请稍等）...\n", repeats);
     double t_basic = avg_time_basic(n, repeats);
     double t_W = avg_time_W_method(n, repeats, leaf);
 
     printf("\n=== 平均时间（秒） ===\n");
     printf("普通 O(n^3) 矩阵乘法平均时间: %.6f s\n", t_basic);
     printf("W (Winograd/Strassen + cross) 平均时间:  %.6f s\n", t_W);
 

     int ok = check_correctness(n, leaf);
     printf("结果一致性检验: %s\n", ok ? "通过" : "失败");
 
     return 0;
 }
 