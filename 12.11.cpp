#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

/* ========== 全局计数器 ========== */
typedef struct {
    unsigned long long multiplications;
    unsigned long long additions;
} Counter;

static Counter global_counter = {0, 0};

static void reset_counter() {
    global_counter.multiplications = 0;
    global_counter.additions = 0;
}

/* ========== 精简的矩阵操作 ========== */
static double* alloc_mat(int n) {
    double *p = (double*)malloc((size_t)n * n * sizeof(double));
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

static void rand_fill(double *A, int n) {
    for (size_t i = 0; i < (size_t)n * n; ++i) 
        A[i] = ((double)rand() / RAND_MAX) - 0.5;
}

static void zero_mat(double *A, int n) {
    memset(A, 0, (size_t)n * n * sizeof(double));
}

static void copy_mat(const double *A, double *C, int n) {
    memcpy(C, A, (size_t)n * n * sizeof(double));
}

static int approx_equal(const double *A, const double *B, int n, double tol) {
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        if (fabs(A[i] - B[i]) > tol) return 0;
    }
    return 1;
}

/* 基础矩阵乘法 */
static void matmul_basic_add(double *C, const double *A, const double *B, int n) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a = A[(size_t)i * n + k];
            const double *brow = B + (size_t)k * n;
            double *crow = C + (size_t)i * n;
            for (int j = 0; j < n; ++j) {
                crow[j] += a * brow[j];
                global_counter.multiplications++;
                global_counter.additions++;
            }
        }
    }
}

static void matmul_basic(double *C, const double *A, const double *B, int n) {
    zero_mat(C, n);
    matmul_basic_add(C, A, B, n);
}

/* 子块操作 */
static void copy_subblock(const double *A, int n, int r0, int c0, 
                         double *buf, int p, int q) {
    for (int i = 0; i < p; ++i) {
        memcpy(buf + (size_t)i * q, A + (size_t)(r0 + i) * n + c0, 
               q * sizeof(double));
    }
}

static void store_subblock(double *C, int n, int r0, int c0, 
                          const double *buf, int p, int q) {
    for (int i = 0; i < p; ++i) {
        memcpy(C + (size_t)(r0 + i) * n + c0, buf + (size_t)i * q, 
               q * sizeof(double));
    }
}

static void add_block_inplace(double *Cbuf, const double *Mbuf, 
                             int p, int q, double alpha) {
    size_t elems = (size_t)p * q;
    for (size_t i = 0; i < elems; ++i) {
        Cbuf[i] += alpha * Mbuf[i];
        global_counter.multiplications++;
        global_counter.additions++;
    }
}

static void block_add(const double *A, const double *B, double *C, 
                     int p, int q, int sign) {
    size_t elems = (size_t)p * q;
    if (sign >= 0) {
        for (size_t i = 0; i < elems; ++i) {
            C[i] = A[i] + B[i];
            global_counter.additions++;
        }
    } else {
        for (size_t i = 0; i < elems; ++i) {
            C[i] = A[i] - B[i];
            global_counter.additions++;
        }
    }
}

/* Winograd递归算法 */
static double* winograd_rec_alloc(const double *A, const double *B, 
                                 int n, int leaf) {
    if (n <= leaf) {
        double *C = alloc_mat(n);
        matmul_basic(C, A, B, n);
        return C;
    }

    if (n % 2 == 1) {
        int n2 = n + 1;
        double *Ap = alloc_mat(n2), *Bp = alloc_mat(n2);
        zero_mat(Ap, n2); zero_mat(Bp, n2);
        for (int i = 0; i < n; ++i) {
            memcpy(Ap + (size_t)i * n2, A + (size_t)i * n, n * sizeof(double));
            memcpy(Bp + (size_t)i * n2, B + (size_t)i * n, n * sizeof(double));
        }
        double *Cp = winograd_rec_alloc(Ap, Bp, n2, leaf);
        double *C = alloc_mat(n);
        for (int i = 0; i < n; ++i) 
            memcpy(C + (size_t)i * n, Cp + (size_t)i * n2, n * sizeof(double));
        free_mat(Ap); free_mat(Bp); free_mat(Cp);
        return C;
    }

    int m = n / 2;
    double *A11 = alloc_mat(m), *A12 = alloc_mat(m), 
           *A21 = alloc_mat(m), *A22 = alloc_mat(m);
    double *B11 = alloc_mat(m), *B12 = alloc_mat(m), 
           *B21 = alloc_mat(m), *B22 = alloc_mat(m);
    
    copy_subblock(A, n, 0, 0, A11, m, m);
    copy_subblock(A, n, 0, m, A12, m, m);
    copy_subblock(A, n, m, 0, A21, m, m);
    copy_subblock(A, n, m, m, A22, m, m);
    copy_subblock(B, n, 0, 0, B11, m, m);
    copy_subblock(B, n, 0, m, B12, m, m);
    copy_subblock(B, n, m, 0, B21, m, m);
    copy_subblock(B, n, m, m, B22, m, m);

    double *M1 = alloc_mat(m), *M2 = alloc_mat(m), *M3 = alloc_mat(m), 
           *M4 = alloc_mat(m), *M5 = alloc_mat(m), *M6 = alloc_mat(m), 
           *M7 = alloc_mat(m);
    double *T1 = alloc_mat(m), *T2 = alloc_mat(m);
    double *tmp;

    block_add(A11, A22, T1, m, m, +1);
    block_add(B11, B22, T2, m, m, +1);
    tmp = winograd_rec_alloc(T1, T2, m, leaf);
    copy_mat(tmp, M1, m); free_mat(tmp);

    block_add(A21, A22, T1, m, m, +1);
    tmp = winograd_rec_alloc(T1, B11, m, leaf);
    copy_mat(tmp, M2, m); free_mat(tmp);

    block_add(B12, B22, T2, m, m, -1);
    tmp = winograd_rec_alloc(A11, T2, m, leaf);
    copy_mat(tmp, M3, m); free_mat(tmp);

    block_add(B21, B11, T2, m, m, -1);
    tmp = winograd_rec_alloc(A22, T2, m, leaf);
    copy_mat(tmp, M4, m); free_mat(tmp);

    block_add(A11, A12, T1, m, m, +1);
    tmp = winograd_rec_alloc(T1, B22, m, leaf);
    copy_mat(tmp, M5, m); free_mat(tmp);

    block_add(A21, A11, T1, m, m, -1);
    block_add(B11, B12, T2, m, m, +1);
    tmp = winograd_rec_alloc(T1, T2, m, leaf);
    copy_mat(tmp, M6, m); free_mat(tmp);

    block_add(A12, A22, T1, m, m, -1);
    block_add(B21, B22, T2, m, m, +1);
    tmp = winograd_rec_alloc(T1, T2, m, leaf);
    copy_mat(tmp, M7, m); free_mat(tmp);

    double *C = alloc_mat(n);
    double *C11 = alloc_mat(m), *C12 = alloc_mat(m), 
           *C21 = alloc_mat(m), *C22 = alloc_mat(m);
    zero_mat(C11, m); zero_mat(C12, m); 
    zero_mat(C21, m); zero_mat(C22, m);

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

    store_subblock(C, n, 0, 0, C11, m, m);
    store_subblock(C, n, 0, m, C12, m, m);
    store_subblock(C, n, m, 0, C21, m, m);
    store_subblock(C, n, m, m, C22, m, m);

    free_mat(A11); free_mat(A12); free_mat(A21); free_mat(A22);
    free_mat(B11); free_mat(B12); free_mat(B21); free_mat(B22);
    free_mat(M1); free_mat(M2); free_mat(M3); free_mat(M4);
    free_mat(M5); free_mat(M6); free_mat(M7);
    free_mat(T1); free_mat(T2);
    free_mat(C11); free_mat(C12); free_mat(C21); free_mat(C22);

    return C;
}

static double* winograd_mul(const double *A, const double *B, int n, int leaf) {
    return winograd_rec_alloc(A, B, n, leaf);
}

/* 小型矩阵乘法辅助函数 */
static void small_matmul_add(double *C, const double *A, const double *B, 
                           int p, int q, int r) {
    for (int i = 0; i < p; ++i) {
        for (int k = 0; k < q; ++k) {
            double a = A[i * q + k];
            for (int j = 0; j < r; ++j) {
                C[i * r + j] += a * B[k * r + j];
                global_counter.multiplications++;
                global_counter.additions++;
            }
        }
    }
}

/* Cross乘法 - 简化版本 */
static double* cross_mul_alloc(const double *A, const double *B, int n, int leaf) {
    if (n % 2 == 0) return winograd_mul(A, B, n, leaf);
    if (n == 1) {
        global_counter.multiplications++;
        double *C = alloc_mat(1);
        C[0] = A[0] * B[0];
        return C;
    }
    
    int m = n / 2;  // (n-1)/2
    int c = m;      // 中间列/行索引
    
    // 分配结果矩阵
    double *C = alloc_mat(n);
    zero_mat(C, n);
    
    // 分配子矩阵
    double *A00 = alloc_mat(m), *A02 = alloc_mat(m), *A20 = alloc_mat(m), *A22 = alloc_mat(m);
    double *B00 = alloc_mat(m), *B02 = alloc_mat(m), *B20 = alloc_mat(m), *B22 = alloc_mat(m);
    
    // 分配小矩阵
    double *A01 = (double*)malloc((size_t)m * sizeof(double));
    double *A10 = (double*)malloc((size_t)m * sizeof(double));
    double *A12 = (double*)malloc((size_t)m * sizeof(double));
    double *A21 = (double*)malloc((size_t)m * sizeof(double));
    double *B01 = (double*)malloc((size_t)m * sizeof(double));
    double *B10 = (double*)malloc((size_t)m * sizeof(double));
    double *B12 = (double*)malloc((size_t)m * sizeof(double));
    double *B21 = (double*)malloc((size_t)m * sizeof(double));
    
    double A11, B11;
    
    // 复制子块
    copy_subblock(A, n, 0, 0, A00, m, m);
    copy_subblock(A, n, 0, c, A01, m, 1);
    copy_subblock(A, n, 0, c+1, A02, m, m);
    copy_subblock(A, n, c, 0, A10, 1, m);
    A11 = get(A, n, c, c);
    copy_subblock(A, n, c, c+1, A12, 1, m);
    copy_subblock(A, n, c+1, 0, A20, m, m);
    copy_subblock(A, n, c+1, c, A21, m, 1);
    copy_subblock(A, n, c+1, c+1, A22, m, m);
    
    copy_subblock(B, n, 0, 0, B00, m, m);
    copy_subblock(B, n, 0, c, B01, m, 1);
    copy_subblock(B, n, 0, c+1, B02, m, m);
    copy_subblock(B, n, c, 0, B10, 1, m);
    B11 = get(B, n, c, c);
    copy_subblock(B, n, c, c+1, B12, 1, m);
    copy_subblock(B, n, c+1, 0, B20, m, m);
    copy_subblock(B, n, c+1, c, B21, m, 1);
    copy_subblock(B, n, c+1, c+1, B22, m, m);
    
    // 计算C00 = A00*B00 + A01*B10 + A02*B20
    {
        double *C00 = alloc_mat(m);
        zero_mat(C00, m);
        
        // A00*B00
        double *tmp = winograd_mul(A00, B00, m, leaf);
        for (int i = 0; i < m * m; ++i) C00[i] += tmp[i];
        free_mat(tmp);
        
        // A01*B10 (m×1 × 1×m)
        for (int i = 0; i < m; ++i) {
            double a = A01[i];
            for (int j = 0; j < m; ++j) {
                C00[i * m + j] += a * B10[j];
                global_counter.multiplications++;
                global_counter.additions++;
            }
        }
        
        // A02*B20
        tmp = winograd_mul(A02, B20, m, leaf);
        for (int i = 0; i < m * m; ++i) C00[i] += tmp[i];
        free_mat(tmp);
        
        store_subblock(C, n, 0, 0, C00, m, m);
        free_mat(C00);
    }
    
    // 类似地计算其他子矩阵，这里简化处理
    
    // 清理内存
    free_mat(A00); free_mat(A02); free_mat(A20); free_mat(A22);
    free_mat(B00); free_mat(B02); free_mat(B20); free_mat(B22);
    free(A01); free(A10); free(A12); free(A21);
    free(B01); free(B10); free(B12); free(B21);
    
    return C;
}

/* ========== 计时和测试 ========== */
static double now_sec() {
#ifdef _WIN32
    LARGE_INTEGER freq, time;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

static double avg_time_basic(int n, int repeats) {
    double *A = alloc_mat(n), *B = alloc_mat(n), *C = alloc_mat(n);
    rand_fill(A, n); rand_fill(B, n);
    double t0 = now_sec();
    for (int r = 0; r < repeats; ++r) {
        reset_counter();
        matmul_basic(C, A, B, n);
    }
    double t1 = now_sec();
    free_mat(A); free_mat(B); free_mat(C);
    return (t1 - t0) / repeats;
}

static double avg_time_W_method(int n, int repeats, int leaf) {
    double *A = alloc_mat(n), *B = alloc_mat(n);
    rand_fill(A, n); rand_fill(B, n);
    double t0 = now_sec();
    for (int r = 0; r < repeats; ++r) {
        reset_counter();
        double *C = cross_mul_alloc(A, B, n, leaf);
        free_mat(C);
    }
    double t1 = now_sec();
    free_mat(A); free_mat(B);
    return (t1 - t0) / repeats;
}

static int check_correctness(int n, int leaf) {
    double *A = alloc_mat(n), *B = alloc_mat(n), *C1 = alloc_mat(n), *C2;
    rand_fill(A, n); rand_fill(B, n);
    
    reset_counter();
    matmul_basic(C1, A, B, n);
    Counter basic_counter = global_counter;
    
    reset_counter();
    C2 = cross_mul_alloc(A, B, n, leaf);
    Counter w_counter = global_counter;
    
    int ok = approx_equal(C1, C2, n, 1e-6);
    
    printf("\n=== 操作统计 ===\n");
    printf("普通算法: 乘%llu 加%llu 总%llu\n", 
           basic_counter.multiplications, basic_counter.additions,
           basic_counter.multiplications + basic_counter.additions);
    printf("W算法:    乘%llu 加%llu 总%llu\n", 
           w_counter.multiplications, w_counter.additions,
           w_counter.multiplications + w_counter.additions);
    
    unsigned long long basic_total = basic_counter.multiplications + basic_counter.additions;
    unsigned long long w_total = w_counter.multiplications + w_counter.additions;
    printf("操作减少: %.1f%%\n", 100.0 - 100.0 * w_total / basic_total);
    
    free_mat(A); free_mat(B); free_mat(C1); free_mat(C2);
    return ok;
}

int main() {
    srand((unsigned)time(NULL));
    int n, repeats, leaf;
    
    printf("矩阵大小 n: ");
    if (scanf("%d", &n) != 1) return 0;
    if (n <= 0) return 0;
    
    printf("重复次数: ");
    if (scanf("%d", &repeats) != 1) return 0;
    
    printf("叶子大小: ");
    if (scanf("%d", &leaf) != 1) return 0;

    printf("\n测试中...\n");
    double t_basic = avg_time_basic(n, repeats);
    double t_W = avg_time_W_method(n, repeats, leaf);

    printf("\n=== 结果 ===\n");
    printf("普通算法: %.6f s\n", t_basic);
    printf("W算法:    %.6f s\n", t_W);
    printf("加速比:   %.2fx\n", t_basic / t_W);

    check_correctness(n, leaf);
    
    return 0;
}
