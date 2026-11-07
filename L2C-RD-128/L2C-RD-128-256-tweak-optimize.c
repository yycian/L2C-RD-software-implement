#define _GNU_SOURCE
#define L2C_RD128_NO_MAIN
#include "L2C-RD-128-256-tweak.c"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <x86intrin.h>

typedef uint32_t u32;
typedef uint64_t u64;

#define BLOCK_SIZE      16u
#define DATA_BYTES      (2u * 1024u * 1024u)
#define NUM_BLOCKS      (DATA_BYTES / BLOCK_SIZE)
#define REPEATS         100
#define PARALLEL_BLOCKS 16
#ifndef BATCH
#define BATCH           32
#endif

#if BATCH <= 0
#error "BATCH must be positive"
#elif (BATCH % PARALLEL_BLOCKS) != 0 || (NUM_BLOCKS % BATCH) != 0
#error "BATCH must be a multiple of the 16-way group and divide NUM_BLOCKS"
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define BENCH_NOINLINE __attribute__((noinline, noclone))
#define BENCH_SCALAR   __attribute__((optimize("no-tree-loop-vectorize", "no-tree-slp-vectorize", "no-tree-loop-distribute-patterns")))
#define L2C_INLINE     static inline __attribute__((always_inline))
#else
#error "This benchmark requires GCC function-specific vectorization controls"
#endif

typedef struct {
    u32 wh[4];                
    u32 rk[ROUNDS][4];        
} fast_keys;

static void fast_keys_init(fast_keys *fk, const u128 KSEQ[], int rounds)
{
    int i, j;
    for (j = 0; j < 4; j++)
        fk->wh[j] = (u32)((KSEQ[0] >> (96 - 32 * j)) & MASK32);
    for (i = 1; i < rounds; i++)
        for (j = 0; j < 4; j++)
            fk->rk[i][j] = (u32)((KSEQ[i] >> (96 - 32 * j)) & MASK32);
}

static inline u32 rl32(u32 x, unsigned r) { return (u32)((x << r) | (x >> (32 - r))); }
static inline u32 rr32(u32 x, unsigned r) { return (u32)((x >> r) | (x << (32 - r))); }

L2C_INLINE void l2c128_even(u32 a[4])
{
    u32 f1 = a[1] ^ (rl32(a[3], 19) & a[2]);
    u32 f0 = (u32)(~(rl32(a[3], 4) & a[2])) ^ a[0];
    u32 f2 = a[2] ^ (rl32(f0, 31) & f1);
    u32 f3 = a[3] ^ (u32)(~(rl32(f0, 10) & f1));
    u32 Rhi = (f1 << 22) | (f2 >> 10);
    u32 Rlo = (f2 << 22) | (f1 >> 10);
    a[0] = f1;
    a[1] = rr32(Rlo ^ f0, 12);
    a[2] = rr32(Rhi ^ f3, 9);
    a[3] = f2;
}

L2C_INLINE void l2c128_odd(u32 a[4], const u32 K[4], int swap)
{
    u32 Rhi = (a[1] << 22) | (a[2] >> 10);
    u32 Rlo = (a[2] << 22) | (a[1] >> 10);
    u32 w0 = rl32(a[0], 12) ^ Rlo;
    u32 w3 = rl32(a[3], 9) ^ Rhi;
    u32 b3 = (u32)(~(rl32(w0, 10) & a[1])) ^ w3;
    u32 b2 = (rl32(w0, 31) & a[1]) ^ a[2];
    u32 b0 = w0 ^ (u32)(~(rl32(b3, 4) & b2));
    u32 b1 = a[1] ^ (rl32(b3, 19) & b2);
    if (swap) { a[0] = b1 ^ K[1]; a[1] = b0 ^ K[0]; a[2] = b3 ^ K[3]; a[3] = b2 ^ K[2]; }
    else      { a[0] = b0 ^ K[0]; a[1] = b1 ^ K[1]; a[2] = b2 ^ K[2]; a[3] = b3 ^ K[3]; }
}

L2C_INLINE void l2c128_key(u32 a[4], const u32 K[4])
{
    a[0] ^= K[0]; a[1] ^= K[1]; a[2] ^= K[2]; a[3] ^= K[3];
}

_Static_assert(ROUNDS % 2 == 0 && ROUNDS >= 4, "l2c128_ks() needs an even ROUNDS >= 4");

static inline void l2c128_ks(u32 c0, u32 c1, u32 c2, u32 c3,
                             const fast_keys *fk,
                             u32 *O0, u32 *O1, u32 *O2, u32 *O3)
{
    u32 a[4] = { c0 ^ fk->wh[0], c1 ^ fk->wh[1], c2 ^ fk->wh[2], c3 ^ fk->wh[3] };
    int i;

    l2c128_even(a);                             
    l2c128_odd(a, fk->rk[1], 1);                
    for (i = 2; i <= ROUNDS - 4; i += 2) {       
        l2c128_key(a, fk->rk[i]);
        l2c128_even(a);
        l2c128_odd(a, fk->rk[i + 1], 1);
    }
    l2c128_key(a, fk->rk[ROUNDS - 2]);
    l2c128_even(a);                            
    l2c128_odd(a, fk->rk[ROUNDS - 1], 0);        
    *O0 = a[0]; *O1 = a[1]; *O2 = a[2]; *O3 = a[3];
}

L2C_INLINE void ctr_group(u32 *restrict buf, u64 base,
                          const fast_keys *restrict fk, int lanes)
{
    u32 C0[PARALLEL_BLOCKS], C1[PARALLEL_BLOCKS];
    u32 C2[PARALLEL_BLOCKS], C3[PARALLEL_BLOCKS];
    u32 O0[PARALLEL_BLOCKS], O1[PARALLEL_BLOCKS];
    u32 O2[PARALLEL_BLOCKS], O3[PARALLEL_BLOCKS];

    __asm__ volatile ("" : "+r" (lanes));
    for (int j = 0; j < lanes; j++) {
        u64 c = base + (u64)j;
        C0[j] = 0; C1[j] = 0;
        C2[j] = (u32)(c >> 32); C3[j] = (u32)c;
    }
#pragma GCC ivdep
    for (int j = 0; j < lanes; j++)
        l2c128_ks(C0[j], C1[j], C2[j], C3[j], fk,
                  &O0[j], &O1[j], &O2[j], &O3[j]);
    for (int j = 0; j < lanes; j++) {
        u32 *p = buf + (size_t)j * 4;
        p[0] ^= O0[j]; p[1] ^= O1[j];
        p[2] ^= O2[j]; p[3] ^= O3[j];
    }
}

static BENCH_NOINLINE
void ctr_group_auto(u32 *restrict buf, u64 base, const fast_keys *restrict fk, int lanes)
{
    ctr_group(buf, base, fk, lanes);
}

static BENCH_NOINLINE BENCH_SCALAR
void ctr_group_scalar(u32 *restrict buf, u64 base, const fast_keys *restrict fk, int lanes)
{
    ctr_group(buf, base, fk, lanes);
}

typedef void group_fn(u32 *restrict, u64, const fast_keys *restrict, int);

L2C_INLINE void ctr_batched(u32 *buf, size_t nblocks, u64 base,
                            const fast_keys *fk, group_fn *group)
{
    size_t b = 0;
    for (; nblocks - b >= BATCH; b += BATCH)
        for (size_t g = b; g < b + BATCH; g += PARALLEL_BLOCKS)
            group(buf + g * 4, base + g, fk, PARALLEL_BLOCKS);

    while (b < nblocks) {
        size_t remaining = nblocks - b;
        int lanes = (int)(remaining < PARALLEL_BLOCKS ? remaining : PARALLEL_BLOCKS);
        group(buf + b * 4, base + b, fk, lanes);
        b += (size_t)lanes;
    }
}

static BENCH_NOINLINE
void ctr_batched_auto(u32 *buf, size_t nblocks, u64 base, const fast_keys *fk)
{
    ctr_batched(buf, nblocks, base, fk, ctr_group_auto);
}

static BENCH_NOINLINE
void ctr_batched_scalar(u32 *buf, size_t nblocks, u64 base, const fast_keys *fk)
{
    ctr_batched(buf, nblocks, base, fk, ctr_group_scalar);
}

static inline u64 tsc0(void) { _mm_lfence(); u64 t = __rdtsc(); _mm_lfence(); return t; }
static inline u64 tsc1(void) { unsigned a; u64 t = __rdtscp(&a); _mm_lfence(); return t; }
static double secs(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}
static int perf_open(void) {
    struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_HARDWARE; pe.size = sizeof pe;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
    return (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
}

int main(void)
{
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    int pinned = (sched_setaffinity(0, sizeof set, &set) == 0);

    u128 key[2] = { 0, 0 };
    u128 tweakey = mk128(0x1111111111111111ULL, 0x1111111111111111ULL);
    u128 KSEQ[ROUNDS];
    fast_keys fk;

    l2crd128_setup(key, tweakey, ROUNDS, KSEQ);
    fast_keys_init(&fk, KSEQ, ROUNDS);

    {
        u128 pt = mk128(0x0022332233223322ULL, 0x0022332233223322ULL);
        u128 ct = l2crd128_encrypt_with_kseq(pt, KSEQ, ROUNDS);
        u128 ct_full = encrypt_block(pt, key, tweakey, ROUNDS, 0);
        u128 exp = mk128(0x237C92CE368F810BULL, 0x7183CAB4E71E4AF4ULL);
        int ok = (ct == exp && ct_full == exp);
        printf("Test vector: %s  (", ok ? "PASS" : "FAIL");
        print_u128_hex(ct);
        printf(", expected 237C92CE368F810B7183CAB4E71E4AF4)\n\n");
        if (!ok) return 1;
    }

    {
        int ok = 1;
        u64 s = 0x123456789ABCDEFULL;
        for (long t = 0; t < 500000; t++) {
            u32 w[4];
            u128 pt, ct;
            u32 o0, o1, o2, o3;
            for (int j = 0; j < 4; j++) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                w[j] = (u32)s;
            }
            pt = words_to_state(w[0], w[1], w[2], w[3]);
            ct = l2crd128_encrypt_with_kseq(pt, KSEQ, ROUNDS);
            l2c128_ks(w[0], w[1], w[2], w[3], &fk, &o0, &o1, &o2, &o3);
            if (words_to_state(o0, o1, o2, o3) != ct) {
                printf("SCALAR MISMATCH at t=%ld\n", t); ok = 0; break;
            }
        }
        printf("scalar-fast vs reference : %s\n",
               ok ? "bit-exact (500,000 random inputs)" : "FAILED");
        if (!ok) return 1;
    }

    {
        int ok = 1;
        const size_t N = 4096;
        u32 *scalar = calloc(N * 4, sizeof(u32));
        u32 *auto_vec = calloc(N * 4, sizeof(u32));
        if (!scalar || !auto_vec) {
            free(scalar); free(auto_vec); return 1;
        }
        ctr_batched_scalar(scalar, N, 1, &fk);
        ctr_batched_auto(auto_vec, N, 1, &fk);
        for (size_t i = 0; i < N && ok; i++) {
            u128 ct = l2crd128_encrypt_with_kseq((u128)(1 + i), KSEQ, ROUNDS);
            u32 e[4]; state_to_words(ct, e);
            for (int j = 0; j < 4; j++)
                if (scalar[i * 4 + j] != e[j] ||
                    auto_vec[i * 4 + j] != e[j]) {
                    printf("CTR MISMATCH blk=%zu w%d\n", i, j); ok = 0;
                }
        }
        free(scalar); free(auto_vec);
        printf("scalar/auto vs reference : %s\n\n",
               ok ? "bit-exact (4,096 counters each)" : "FAILED");
        if (!ok) return 1;
    }

    const size_t NW = DATA_BYTES / sizeof(u32);
    u32 *buf = malloc(DATA_BYTES), *ref = malloc(DATA_BYTES);
    if (!buf || !ref) { free(buf); free(ref); return 1; }
    for (size_t i = 0; i < NW; i++) ref[i] = (u32)i ^ 0x12345678u;

    int pfd = perf_open(); int have_perf = (pfd != -1);

    for (int w = 0; w < 4; w++) {
        memcpy(buf, ref, DATA_BYTES);
        ctr_batched_scalar(buf, NUM_BLOCKS, 1, &fk);
        memcpy(buf, ref, DATA_BYTES);
        ctr_batched_auto(buf, NUM_BLOCKS, 1, &fk);
    }

    printf("cipher      : L2C-RD-128/256 (%d rounds, 128-bit block, 256-bit key)\n", ROUNDS);
    printf("mode        : single-thread, CTR, LLC-resident (2 MiB)\n");
    printf("comparison  : identical u32/native-endian CTR; vectorization is the only toggle\n");
    printf("parallelism : scalar x1; auto AVX2 2 x 8 lanes = 16-way\n");
    printf("repeats     : %d (interleaved, best-of-N)   core pinned: %s   batch: %d\n",
           REPEATS, pinned ? "yes" : "NO", BATCH);
    printf("cycle source: %s\n\n",
           have_perf ? "RDTSC reference ticks + perf core cycles" : "RDTSC reference ticks");
    printf("%-22s %14s %13s %11s %10s\n",
           "impl", "ticks/block", "ticks/byte", "MiB/s", "speedup");
    printf("--------------------------------------------------------------------------\n");

    u64 best_tsc[2] = {~0ull, ~0ull};
    u64 best_core[2] = {~0ull, ~0ull};
    double best_sec[2] = {1e300, 1e300};
    volatile u64 sink = 0;

    for (int rep = 0; rep < REPEATS; rep++) {
        for (int pass = 0; pass < 2; pass++) {
            int which = (rep + pass) & 1;
            memcpy(buf, ref, DATA_BYTES);
            if (have_perf && (ioctl(pfd, PERF_EVENT_IOC_RESET, 0) < 0 ||
                              ioctl(pfd, PERF_EVENT_IOC_ENABLE, 0) < 0)) {
                perror("perf reset/enable");
                close(pfd); free(buf); free(ref); return 1;
            }
            struct timespec a, b;
            clock_gettime(CLOCK_MONOTONIC_RAW, &a); u64 c0 = tsc0();
            if (which == 0)
                ctr_batched_scalar(buf, NUM_BLOCKS, 1, &fk);
            else
                ctr_batched_auto(buf, NUM_BLOCKS, 1, &fk);
            u64 c1 = tsc1(); clock_gettime(CLOCK_MONOTONIC_RAW, &b);
            u64 core = 0;
            if (have_perf) {
                if (ioctl(pfd, PERF_EVENT_IOC_DISABLE, 0) < 0) {
                    perror("perf disable");
                    close(pfd); free(buf); free(ref); return 1;
                }
                ssize_t bytes = read(pfd, &core, sizeof core);
                if (bytes != (ssize_t)sizeof core) {
                    if (bytes < 0) perror("perf read");
                    else fprintf(stderr, "perf read: expected %zu bytes, got %zd\n",
                                 sizeof core, bytes);
                    close(pfd); free(buf); free(ref); return 1;
                }
            }
            u32 s = 0; for (size_t i = 0; i < NW; i++) s ^= buf[i]; sink += s ^ (u32)rep;
            double sc = secs(a, b);
            if (c1 - c0 < best_tsc[which]) {
                best_tsc[which] = c1 - c0;
                best_sec[which] = sc;
                best_core[which] = core;
            }
        }
    }

    double scalar_cpb = (double)best_tsc[0] / NUM_BLOCKS;
    for (int which = 0; which < 2; which++) {
        double cpb = (double)best_tsc[which] / NUM_BLOCKS;
        double cpby = (double)best_tsc[which] / DATA_BYTES;
        double mib = ((double)DATA_BYTES / 1048576.0) / best_sec[which];
        const char *name = which == 0 ? "scalar u32 (no SIMD)"
                                      : "auto AVX2 u32 (x16)";
        printf("%-22s %14.2f %13.2f %11.2f %9.2fx\n",
               name, cpb, cpby, mib, scalar_cpb / cpb);
        if (have_perf) {
            printf("%-22s %14.2f %13.2f  (true core cycles)\n", "",
                   (double)best_core[which] / NUM_BLOCKS,
                   (double)best_core[which] / DATA_BYTES);
        }
    }
    printf("\nsink=%016llX\n", (unsigned long long)sink);
    if (have_perf) close(pfd);
    free(buf); free(ref); return 0;
}
