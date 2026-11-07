#define _GNU_SOURCE
#define L2C_RD_NO_MAIN
#include "L2C-RD-64-tweak.c"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <x86intrin.h>
#define DATA_BYTES  (2u * 1024u * 1024u)
#define BLOCK_BYTES 8u
#define NUM_BLOCKS  (DATA_BYTES / BLOCK_BYTES)
#define REPEATS     100
#ifndef BATCH
#define BATCH 32
#endif
_Static_assert(BATCH > 0, "BATCH must be positive");

#if defined(__GNUC__) && !defined(__clang__)
#define BENCH_NOINLINE __attribute__((noinline, noclone))
#define BENCH_SCALAR   __attribute__((optimize("no-tree-loop-vectorize", "no-tree-slp-vectorize")))
#define L2C_INLINE     static inline __attribute__((always_inline))
#else
#error "This benchmark requires GCC function-specific vectorization controls"
#endif

typedef struct {
    uint16_t wh[4];         
    uint16_t rk[ROUNDS][4]; 
} fast_keys;

static void split_words(uint64_t k, uint16_t w[4])
{
    w[0] = (uint16_t)(k >> 48); w[1] = (uint16_t)(k >> 32);
    w[2] = (uint16_t)(k >> 16); w[3] = (uint16_t)(k);
}

static void fast_keys_from_kseq(const uint64_t KSEQ[], fast_keys *fk)
{
    int i;
    split_words(KSEQ[0], fk->wh);
    for (i = 1; i < ROUNDS; i++) split_words(KSEQ[i], fk->rk[i]);
}

static inline uint16_t rl16(uint16_t x, unsigned r){ return (uint16_t)((x << r) | (x >> (16 - r))); }
static inline uint16_t rr16(uint16_t x, unsigned r){ return (uint16_t)((x >> r) | (x << (16 - r))); }

L2C_INLINE void l2c_even(uint16_t a[4])
{
    uint16_t f1 = a[1] ^ (uint16_t)(rl16(a[3],4) & a[2]);
    uint16_t f0 = (uint16_t)(~(rl16(a[3],3) & a[2])) ^ a[0];
    uint16_t f2 = a[2] ^ (uint16_t)(rl16(f0,10) & f1);
    uint16_t f3 = a[3] ^ (uint16_t)(~(rl16(f0,3) & f1));
    uint16_t Rhi = (uint16_t)((f1 << 11) | (f2 >> 5));
    uint16_t Rlo = (uint16_t)((f2 << 11) | (f1 >> 5));
    a[0] = f1;
    a[1] = rr16((uint16_t)(Rlo ^ f0), 5);
    a[2] = rr16((uint16_t)(Rhi ^ f3), 9);
    a[3] = f2;
}

L2C_INLINE void l2c_odd(uint16_t a[4], const uint16_t K[4], int swap)
{
    uint16_t Rhi = (uint16_t)((a[1] << 11) | (a[2] >> 5));
    uint16_t Rlo = (uint16_t)((a[2] << 11) | (a[1] >> 5));
    uint16_t w0 = rl16(a[0],5) ^ Rlo;
    uint16_t w3 = rl16(a[3],9) ^ Rhi;
    uint16_t b3 = (uint16_t)(~(rl16(w0,3) & a[1])) ^ w3;
    uint16_t b2 = (uint16_t)(rl16(w0,10) & a[1]) ^ a[2];
    uint16_t b0 = w0 ^ (uint16_t)(~(rl16(b3,3) & b2));
    uint16_t b1 = a[1] ^ (uint16_t)(rl16(b3,4) & b2);
    if (swap) { a[0] = b1 ^ K[1]; a[1] = b0 ^ K[0]; a[2] = b3 ^ K[3]; a[3] = b2 ^ K[2]; }
    else      { a[0] = b0 ^ K[0]; a[1] = b1 ^ K[1]; a[2] = b2 ^ K[2]; a[3] = b3 ^ K[3]; }
}

L2C_INLINE void l2c_key(uint16_t a[4], const uint16_t K[4])
{
    a[0] ^= K[0]; a[1] ^= K[1]; a[2] ^= K[2]; a[3] ^= K[3];
}

_Static_assert(ROUNDS % 2 == 0 && ROUNDS >= 4, "l2c_ks() needs an even ROUNDS >= 4");

static inline void l2c_ks(uint16_t c0, uint16_t c1, uint16_t c2, uint16_t c3,
                          const fast_keys *fk,
                          uint16_t *O0, uint16_t *O1, uint16_t *O2, uint16_t *O3)
{
    uint16_t a[4] = { (uint16_t)(c0 ^ fk->wh[0]), (uint16_t)(c1 ^ fk->wh[1]),
                      (uint16_t)(c2 ^ fk->wh[2]), (uint16_t)(c3 ^ fk->wh[3]) };
    int i;

    l2c_even(a);                                
    l2c_odd(a, fk->rk[1], 1);                   
    for (i = 2; i <= ROUNDS - 4; i += 2) {       
        l2c_key(a, fk->rk[i]);
        l2c_even(a);
        l2c_odd(a, fk->rk[i+1], 1);
    }
    l2c_key(a, fk->rk[ROUNDS - 2]);
    l2c_even(a);                                
    l2c_odd(a, fk->rk[ROUNDS - 1], 0);          
    *O0 = a[0]; *O1 = a[1]; *O2 = a[2]; *O3 = a[3];
}

static uint64_t l2c_enc64(uint64_t pt, const fast_keys *fk)
{
    uint16_t o0,o1,o2,o3;
    l2c_ks((uint16_t)(pt>>48),(uint16_t)(pt>>32),(uint16_t)(pt>>16),(uint16_t)pt,
           fk,&o0,&o1,&o2,&o3);
    return ((uint64_t)o0<<48)|((uint64_t)o1<<32)|((uint64_t)o2<<16)|(uint64_t)o3;
}

L2C_INLINE void ctr_batched(uint16_t *buf, size_t nblocks, uint64_t base, const fast_keys *fk)
{
    size_t b = 0;
    if (base <= UINT32_MAX && nblocks <= (uint64_t)UINT32_MAX - base + 1) {
        for (; nblocks - b >= BATCH; b += BATCH) {
            uint16_t C2[BATCH], C3[BATCH];
            uint16_t O0[BATCH], O1[BATCH], O2[BATCH], O3[BATCH];
            int j;
            for (j = 0; j < BATCH; j++) {
                uint32_t c = (uint32_t)(base + b + j);
                C2[j] = (uint16_t)(c >> 16); C3[j] = (uint16_t)c;
            }
            for (j = 0; j < BATCH; j++)               /* <-- vectorized loop */
                l2c_ks(0, 0, C2[j], C3[j], fk, &O0[j], &O1[j], &O2[j], &O3[j]);
            for (j = 0; j < BATCH; j++) {
                uint16_t *p = buf + (b + j) * 4;
                p[0] ^= O0[j]; p[1] ^= O1[j]; p[2] ^= O2[j]; p[3] ^= O3[j];
            }
        }
    }
   
    for (; b < nblocks; b++) {
        uint64_t stream = l2c_enc64(base + (uint64_t)b, fk);
        uint16_t *p = buf + b * 4;
        p[0] ^= (uint16_t)(stream >> 48); p[1] ^= (uint16_t)(stream >> 32);
        p[2] ^= (uint16_t)(stream >> 16); p[3] ^= (uint16_t)stream;
    }
}

static BENCH_NOINLINE
void ctr_batched_auto(uint16_t *buf, size_t nblocks, uint64_t base, const fast_keys *fk)
{
    ctr_batched(buf, nblocks, base, fk);
}

static BENCH_NOINLINE BENCH_SCALAR
void ctr_batched_scalar(uint16_t *buf, size_t nblocks, uint64_t base, const fast_keys *fk)
{
    ctr_batched(buf, nblocks, base, fk);
}

static inline uint64_t tsc0(void){ _mm_lfence(); uint64_t t=__rdtsc(); _mm_lfence(); return t; }
static inline uint64_t tsc1(void){ unsigned a; uint64_t t=__rdtscp(&a); _mm_lfence(); return t; }
static double secs(struct timespec a, struct timespec b){ return (b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9; }
static int perf_open(void){ struct perf_event_attr pe; memset(&pe,0,sizeof pe);
    pe.type=PERF_TYPE_HARDWARE; pe.size=sizeof pe; pe.config=PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled=1; pe.exclude_kernel=1; pe.exclude_hv=1;
    return (int)syscall(__NR_perf_event_open,&pe,0,-1,-1,0); }
static uint64_t words64(const uint16_t*p){ return ((uint64_t)p[0]<<48)|((uint64_t)p[1]<<32)|((uint64_t)p[2]<<16)|(uint64_t)p[3]; }

int main(void)
{
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0,&set);
    int pinned = (sched_setaffinity(0, sizeof set, &set) == 0);

    uint64_t key[2]  = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
    uint64_t tweakey = 0x1111111111111111ULL;
    uint64_t KSEQ[ROUNDS];
    l2crd64_setup(key, tweakey, ROUNDS, KSEQ);
    fast_keys fk; fast_keys_from_kseq(KSEQ, &fk);

    {
        uint64_t z[2] = {0, 0}, K[ROUNDS];
        l2crd64_setup(z, 0x1111111111111111ULL, ROUNDS, K);
        uint64_t ct = l2crd64_encrypt_with_kseq(0x0022332233223322ULL, K, ROUNDS);
        printf("Test vector: %s  (%016llX, expected B39EBB620A76CABE)\n\n",
               ct == 0xB39EBB620A76CABEULL ? "PASS" : "FAIL", (unsigned long long)ct);
        if (ct != 0xB39EBB620A76CABEULL) return 1;
    }

    {
        uint64_t seed = 0x9E3779B97F4A7C15ULL; long i; int ok = 1;
        for (i = 0; i < 2000000; i++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            uint64_t ref = l2crd64_encrypt_with_kseq(seed, KSEQ, ROUNDS);
            if (l2c_enc64(seed, &fk) != ref) {
                printf("U16 CORE MISMATCH at %ld\n", i); ok = 0; break; }
        }
        printf("u16 core     vs reference : %s\n", ok ? "bit-exact (2,000,000 vectors)" : "FAILED");
        if (!ok) return 1;
    }

    {
        int ok = 1; size_t n = 4096;
        uint16_t *scalar = calloc(n, 8), *auto_vec = calloc(n, 8);
        if (!scalar || !auto_vec) { free(scalar); free(auto_vec); return 1; }
        ctr_batched_scalar(scalar, n, 1, &fk);
        ctr_batched_auto(auto_vec, n, 1, &fk);
        for (size_t j = 0; j < n; j++) {
            uint64_t ref = l2crd64_encrypt_with_kseq(1 + j, KSEQ, ROUNDS);
            uint64_t got_scalar = words64(scalar + j*4), got_auto = words64(auto_vec + j*4);
            if (ref != got_scalar || ref != got_auto) {
                printf("CTR MISMATCH blk=%zu ref=%016llX scalar=%016llX auto=%016llX\n",
                       j, (unsigned long long)ref,
                       (unsigned long long)got_scalar, (unsigned long long)got_auto);
                ok = 0; break;
            }
        }
        free(scalar); free(auto_vec);
        printf("scalar/auto  vs reference : %s\n",
               ok ? "bit-exact (4,096 counters each)" : "FAILED");
        if (!ok) return 1;
    }

    uint8_t *buf = malloc(DATA_BYTES), *ref = malloc(DATA_BYTES);
    if (!buf || !ref) { free(buf); free(ref); return 1; }
    for (size_t i = 0; i < DATA_BYTES; i++) ref[i] = (uint8_t)(i*167+13);

    int pfd = perf_open(); int have_perf = (pfd != -1);

    for (int w = 0; w < 4; w++) {
        memcpy(buf, ref, DATA_BYTES);
        ctr_batched_scalar((uint16_t*)buf, NUM_BLOCKS, 1, &fk);
        memcpy(buf, ref, DATA_BYTES);
        ctr_batched_auto((uint16_t*)buf, NUM_BLOCKS, 1, &fk);
    }

    printf("mode        : single-thread, CTR, LLC-resident (%u MiB)\n",
           DATA_BYTES/(1024u*1024u));
    printf("comparison  : identical u16/native-endian CTR; vectorization is the only toggle\n");
    printf("repeats     : %d (interleaved, best-of-N)   core pinned: %s   batch: %d\n",
           REPEATS, pinned ? "yes" : "NO", BATCH);
    printf("cycle source: %s\n\n", have_perf ? "RDTSC reference ticks + perf core cycles" : "RDTSC reference ticks");
    printf("%-22s %14s %13s %11s %10s\n","impl","ticks/block","ticks/byte","MiB/s","speedup");
    printf("--------------------------------------------------------------------------\n");

    uint64_t best_tsc[2] = {~0ull, ~0ull}, best_core[2] = {~0ull, ~0ull};
    double best_sec[2] = {1e300, 1e300};
    volatile uint64_t sink = 0;

    for (int rep = 0; rep < REPEATS; rep++) {
        for (int pass = 0; pass < 2; pass++) {
            int which = (rep + pass) & 1;
            memcpy(buf, ref, DATA_BYTES);
            if (have_perf && (ioctl(pfd,PERF_EVENT_IOC_RESET,0)<0 ||
                              ioctl(pfd,PERF_EVENT_IOC_ENABLE,0)<0)) {
                perror("perf reset/enable");
                close(pfd); free(buf); free(ref); return 1;
            }
            struct timespec a,b; clock_gettime(CLOCK_MONOTONIC_RAW,&a); uint64_t c0 = tsc0();
            if (which == 0)
                ctr_batched_scalar((uint16_t*)buf, NUM_BLOCKS, 1, &fk);
            else
                ctr_batched_auto((uint16_t*)buf, NUM_BLOCKS, 1, &fk);
            uint64_t c1 = tsc1(); clock_gettime(CLOCK_MONOTONIC_RAW,&b);
            uint64_t core = 0;
            if (have_perf) {
                if (ioctl(pfd,PERF_EVENT_IOC_DISABLE,0)<0) {
                    perror("perf disable");
                    close(pfd); free(buf); free(ref); return 1;
                }
                ssize_t bytes = read(pfd,&core,sizeof core);
                if (bytes != (ssize_t)sizeof core) {
                    if (bytes<0) perror("perf read");
                    else fprintf(stderr,"perf read: expected %zu bytes, got %zd\n",sizeof core,bytes);
                    close(pfd); free(buf); free(ref); return 1;
                }
            }
            uint32_t s=0; for(size_t i=0;i<DATA_BYTES;i++) s^=buf[i]; sink += s^(uint32_t)rep;
            double sc = secs(a,b);
            if (c1-c0 < best_tsc[which]) {
                best_tsc[which] = c1-c0; best_sec[which] = sc; best_core[which] = core;
            }
        }
    }

    double scalar_cpb = (double)best_tsc[0]/NUM_BLOCKS;
    for (int which = 0; which < 2; which++) {
        double cpb=(double)best_tsc[which]/NUM_BLOCKS;
        double cpby=(double)best_tsc[which]/DATA_BYTES;
        double mib=((double)DATA_BYTES/1048576.0)/best_sec[which];
        const char *n = which == 0 ? "scalar u16 (no SIMD)"
                                   : "auto AVX2 u16 (x16)";
        printf("%-22s %14.2f %13.2f %11.2f %9.2fx\n", n, cpb, cpby, mib, scalar_cpb/cpb);
        if (have_perf) printf("%-22s %14.2f %13.2f  (true core cycles)\n","",
                              (double)best_core[which]/NUM_BLOCKS,
                              (double)best_core[which]/DATA_BYTES);
    }
    printf("\nsink=%016llX\n",(unsigned long long)sink);
    if (have_perf) close(pfd);
    free(buf); free(ref);
    return 0;
}
