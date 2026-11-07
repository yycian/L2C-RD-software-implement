#define _GNU_SOURCE
#define L2C_RD32_NO_MAIN
#include "L2C-RD-32-tweak.c"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <x86intrin.h>

typedef uint32_t u32; typedef uint64_t u64;
#define BLOCK_SIZE 4
#ifndef BATCH
#define BATCH 32
#endif
_Static_assert(BATCH > 0, "BATCH must be positive");

#if defined(__GNUC__) && !defined(__clang__)
#define BENCH_NOINLINE __attribute__((noinline, noclone))
#define BENCH_SCALAR   __attribute__((optimize("no-tree-loop-vectorize", "no-tree-slp-vectorize", "no-tree-loop-distribute-patterns")))
#else
#error "This benchmark requires GCC function-specific vectorization controls"
#endif
typedef struct {
    uint8_t wh[4];          
    uint8_t rk[ROUNDS][4]; 
} fast_keys;

static void split_words(uint32_t k, uint8_t w[4])
{
    w[0] = (uint8_t)(k >> 24); w[1] = (uint8_t)(k >> 16);
    w[2] = (uint8_t)(k >> 8);  w[3] = (uint8_t)(k);
}

static void fast_keys_from_kseq(const uint32_t KSEQ[], fast_keys *fk)
{
    int i;
    split_words(KSEQ[0], fk->wh);
    for (i = 1; i < ROUNDS; i++) split_words(KSEQ[i], fk->rk[i]);
}

static inline uint16_t rl8(uint16_t x, unsigned r)
{
    return (uint16_t)(((x << r) | (x >> (8 - r))) & 0xFFu);
}
static inline uint16_t rr8(uint16_t x, unsigned r){ return rl8(x, 8-r); }

#define L2C32_INLINE static inline __attribute__((always_inline))

L2C32_INLINE void l2c32_even(uint16_t a[4])
{
    uint16_t f1 = a[1] ^ (uint16_t)(a[3] & a[2]);               /* rotate by 0 */
    uint16_t f0 = (uint16_t)((~(rl8(a[3],3) & a[2])) & 0xFFu) ^ a[0];
    uint16_t f2 = a[2] ^ (uint16_t)(rl8(f0,2) & f1);
    uint16_t f3 = a[3] ^ (uint16_t)((~(rl8(f0,3) & f1)) & 0xFFu);
    uint16_t Rhi = (uint16_t)(((f1 << 1) | (f2 >> 7)) & 0xFFu);
    uint16_t Rlo = (uint16_t)(((f2 << 1) | (f1 >> 7)) & 0xFFu);
    a[0] = f1;
    a[1] = rr8((uint16_t)(Rlo ^ f0), 2);
    a[2] = rr8((uint16_t)(Rhi ^ f3), 2);
    a[3] = f2;
}

L2C32_INLINE void l2c32_odd(uint16_t a[4], const uint8_t K[4], int swap)
{
    uint16_t Rhi = (uint16_t)(((a[1] << 1) | (a[2] >> 7)) & 0xFFu);
    uint16_t Rlo = (uint16_t)(((a[2] << 1) | (a[1] >> 7)) & 0xFFu);
    uint16_t w0 = rl8(a[0],2) ^ Rlo;
    uint16_t w3 = rl8(a[3],2) ^ Rhi;
    uint16_t b3 = (uint16_t)((~(rl8(w0,3) & a[1])) & 0xFFu) ^ w3;
    uint16_t b2 = (uint16_t)(rl8(w0,2) & a[1]) ^ a[2];
    uint16_t b0 = w0 ^ (uint16_t)((~(rl8(b3,3) & b2)) & 0xFFu);
    uint16_t b1 = a[1] ^ (uint16_t)(b3 & b2);                   /* rotate by 0 */
    if (swap) { a[0] = b1 ^ K[1]; a[1] = b0 ^ K[0]; a[2] = b3 ^ K[3]; a[3] = b2 ^ K[2]; }
    else      { a[0] = b0 ^ K[0]; a[1] = b1 ^ K[1]; a[2] = b2 ^ K[2]; a[3] = b3 ^ K[3]; }
}

L2C32_INLINE void l2c32_key(uint16_t a[4], const uint8_t K[4])
{
    a[0] ^= K[0]; a[1] ^= K[1]; a[2] ^= K[2]; a[3] ^= K[3];
}

_Static_assert(ROUNDS % 2 == 0 && ROUNDS >= 4, "l2c32_ks() needs an even ROUNDS >= 4");

L2C32_INLINE
void l2c32_ks(uint16_t c0, uint16_t c1, uint16_t c2, uint16_t c3,
                            const fast_keys *fk,
                            uint16_t *O0, uint16_t *O1,
                            uint16_t *O2, uint16_t *O3)
{
    uint16_t a[4] = { (uint16_t)(c0 ^ fk->wh[0]), (uint16_t)(c1 ^ fk->wh[1]),
                      (uint16_t)(c2 ^ fk->wh[2]), (uint16_t)(c3 ^ fk->wh[3]) };
    int i;

    l2c32_even(a);                               
    l2c32_odd(a, fk->rk[1], 1);                  
    for (i = 2; i <= ROUNDS - 4; i += 2) {      
        l2c32_key(a, fk->rk[i]);
        l2c32_even(a);
        l2c32_odd(a, fk->rk[i+1], 1);
    }
    l2c32_key(a, fk->rk[ROUNDS - 2]);
    l2c32_even(a);                               
    l2c32_odd(a, fk->rk[ROUNDS - 1], 0);         
    *O0 = a[0]; *O1 = a[1]; *O2 = a[2]; *O3 = a[3];
}

static uint32_t l2c32_enc32(uint32_t pt, const fast_keys *fk)
{
    uint16_t o0,o1,o2,o3;
    l2c32_ks((uint16_t)(uint8_t)(pt>>24),(uint16_t)(uint8_t)(pt>>16),
             (uint16_t)(uint8_t)(pt>>8),(uint16_t)(uint8_t)pt,
             fk,&o0,&o1,&o2,&o3);
    return ((uint32_t)o0<<24)|((uint32_t)o1<<16)|((uint32_t)o2<<8)|(uint32_t)o3;
}

L2C32_INLINE
void ctr_batched(uint8_t *buf, size_t nblocks, u32 base, const fast_keys *fk)
{
    size_t b;
    for (b = 0; nblocks - b >= BATCH; b += BATCH) {
        uint16_t C0[BATCH],C1[BATCH],C2[BATCH],C3[BATCH];
        uint16_t O0[BATCH],O1[BATCH],O2[BATCH],O3[BATCH];
        int j;
        for (j = 0; j < BATCH; j++) {
            u32 c = base + (u32)b + (u32)j;
            C0[j]=(uint16_t)(uint8_t)(c>>24); C1[j]=(uint16_t)(uint8_t)(c>>16);
            C2[j]=(uint16_t)(uint8_t)(c>>8);  C3[j]=(uint16_t)(uint8_t)c;
        }
        for (j = 0; j < BATCH; j++)               /* <-- vectorized loop */
            l2c32_ks(C0[j],C1[j],C2[j],C3[j],fk,&O0[j],&O1[j],&O2[j],&O3[j]);
        for (j = 0; j < BATCH; j++) {
            uint8_t *p = buf + (b + j) * 4;
            p[0]^=(uint8_t)O0[j]; p[1]^=(uint8_t)O1[j];
            p[2]^=(uint8_t)O2[j]; p[3]^=(uint8_t)O3[j];
        }
    }
    for (; b < nblocks; b++) {
        u32 stream = l2c32_enc32(base + (u32)b, fk);
        uint8_t *p = buf + b * 4;
        p[0] ^= (uint8_t)(stream >> 24); p[1] ^= (uint8_t)(stream >> 16);
        p[2] ^= (uint8_t)(stream >> 8);  p[3] ^= (uint8_t)stream;
    }
}

static BENCH_NOINLINE
void ctr_batched_auto(uint8_t *buf, size_t nblocks, u32 base, const fast_keys *fk)
{
    ctr_batched(buf, nblocks, base, fk);
}

static BENCH_NOINLINE BENCH_SCALAR
void ctr_batched_scalar(uint8_t *buf, size_t nblocks, u32 base, const fast_keys *fk)
{
    ctr_batched(buf, nblocks, base, fk);
}

static inline u64 tsc0(void){ _mm_lfence(); u64 t=__rdtsc(); _mm_lfence(); return t; }
static inline u64 tsc1(void){ unsigned a; u64 t=__rdtscp(&a); _mm_lfence(); return t; }
static double secs(struct timespec a, struct timespec b){ return (b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9; }
static int perf_open(void){ struct perf_event_attr pe; memset(&pe,0,sizeof pe);
    pe.type=PERF_TYPE_HARDWARE; pe.size=sizeof pe; pe.config=PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled=1; pe.exclude_kernel=1; pe.exclude_hv=1;
    return (int)syscall(__NR_perf_event_open,&pe,0,-1,-1,0); }
static u32 load32(const uint8_t*p){ return ((u32)p[0]<<24)|((u32)p[1]<<16)|((u32)p[2]<<8)|p[3]; }

int main(void){
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0,&set);
    int pinned=(sched_setaffinity(0,sizeof set,&set)==0);

    u32 key[2]  = {0x01234567U, 0x89ABCDEFU};
    u32 tweakey = 0x11111111U;
    u32 KSEQ[ROUNDS];
    l2crd32_setup(key, tweakey, ROUNDS, KSEQ);
    fast_keys fk; fast_keys_from_kseq(KSEQ, &fk);

    { u32 z[2]={0,0}, K[ROUNDS];
      l2crd32_setup(z,0x00033333U,ROUNDS,K);
      u32 ct = l2crd32_encrypt_with_kseq(0x33300000U, K, ROUNDS);
      printf("Test vector: %s  (%08X, expected 1D797660)\n\n",
             ct==0x1D797660U?"PASS":"FAIL", ct);
      if(ct!=0x1D797660U) return 1; }

    { u32 seed=0x9E3779B9U; int ok=1;
      for(long i=0;i<2000000;i++){
        seed^=seed<<13; seed^=seed>>17; seed^=seed<<5;
        u32 ref=l2crd32_encrypt_with_kseq(seed,KSEQ,ROUNDS);
        u32 got=l2c32_enc32(seed,&fk);
        if(ref!=got){ printf("SCALAR MISMATCH at %ld: ref=%08X got=%08X\n",i,ref,got); ok=0; break; }
      }
      printf("scalar-fast vs reference : %s\n", ok?"bit-exact (2,000,000 vectors)":"FAILED");
      if(!ok) return 1; }

    { int ok=1; size_t n=4096;
      uint8_t *scalar=calloc(n,4),*auto_vec=calloc(n,4);
      if(!scalar||!auto_vec){ free(scalar); free(auto_vec); return 1; }
      ctr_batched_scalar(scalar,n,1,&fk);
      ctr_batched_auto(auto_vec,n,1,&fk);
      for(size_t j=0;j<n;j++){
        u32 expected=l2crd32_encrypt_with_kseq(1+(u32)j,KSEQ,ROUNDS);
        u32 got_scalar=load32(scalar+j*4),got_auto=load32(auto_vec+j*4);
        if(got_scalar!=expected||got_auto!=expected){
            printf("CTR MISMATCH blk=%zu ref=%08X scalar=%08X auto=%08X\n",
                   j,expected,got_scalar,got_auto); ok=0; break;
        }
      }
      free(scalar); free(auto_vec);
      printf("scalar/auto vs reference : %s\n\n",
             ok?"bit-exact (4,096 counters each)":"FAILED");
      if(!ok) return 1; }

    const size_t DATA=2ULL*1024*1024, NB=DATA/BLOCK_SIZE;
    const int REPEATS=100;
    uint8_t *buf=malloc(DATA), *ref=malloc(DATA);
    if(!buf||!ref){ free(buf); free(ref); return 1; }
    for(size_t i=0;i<DATA;i++) ref[i]=(uint8_t)(i*167+13);

    int pfd=perf_open(); int have_perf=(pfd!=-1);

    for(int w=0;w<4;w++){
        memcpy(buf,ref,DATA);
        ctr_batched_scalar(buf,NB,1,&fk);
        memcpy(buf,ref,DATA);
        ctr_batched_auto(buf,NB,1,&fk);
    }

    printf("cipher      : L2C-RD-32 (%d rounds, 32-bit block, four 8-bit words)\n", ROUNDS);
    printf("mode        : single-thread, CTR, LLC-resident (2 MiB)\n");
    printf("comparison  : identical u16-lane/big-endian CTR; vectorization is the only toggle\n");
    printf("parallelism : scalar x1; auto AVX2 16 x uint16 lanes = 16-way\n");
    printf("repeats     : %d (interleaved, best-of-N)   core pinned: %s   batch: %d\n",
           REPEATS,pinned?"yes":"NO",BATCH);
    printf("cycle source: %s\n\n",have_perf?"RDTSC reference ticks + perf core cycles":"RDTSC reference ticks");
    printf("%-22s %14s %13s %11s %10s\n","impl","ticks/block","ticks/byte","MiB/s","speedup");
    printf("--------------------------------------------------------------------------\n");

    u64 best_tsc[2]={~0ull,~0ull},best_core[2]={~0ull,~0ull};
    double best_sec[2]={1e300,1e300};
    volatile u64 sink=0;
    for(int rep=0;rep<REPEATS;rep++){
      for(int pass=0;pass<2;pass++){
            int which=(rep+pass)&1;
            memcpy(buf,ref,DATA);
            if(have_perf && (ioctl(pfd,PERF_EVENT_IOC_RESET,0)<0 ||
                             ioctl(pfd,PERF_EVENT_IOC_ENABLE,0)<0)) {
                perror("perf reset/enable");
                close(pfd); free(buf); free(ref); return 1;
            }
            struct timespec a,b; clock_gettime(CLOCK_MONOTONIC_RAW,&a); u64 c0=tsc0();
            if(which==0) ctr_batched_scalar(buf,NB,1,&fk);
            else         ctr_batched_auto(buf,NB,1,&fk);
            u64 c1=tsc1(); clock_gettime(CLOCK_MONOTONIC_RAW,&b);
            u64 core=0;
            if(have_perf) {
                if(ioctl(pfd,PERF_EVENT_IOC_DISABLE,0)<0) {
                    perror("perf disable");
                    close(pfd); free(buf); free(ref); return 1;
                }
                ssize_t bytes=read(pfd,&core,sizeof core);
                if(bytes!=(ssize_t)sizeof core) {
                    if(bytes<0) perror("perf read");
                    else fprintf(stderr,"perf read: expected %zu bytes, got %zd\n",sizeof core,bytes);
                    close(pfd); free(buf); free(ref); return 1;
                }
            }
            u32 s=0; for(size_t i=0;i<DATA;i++) s^=buf[i]; sink+=s^(u32)rep;
            double sc=secs(a,b);
            if(c1-c0<best_tsc[which]){
                best_tsc[which]=c1-c0; best_sec[which]=sc; best_core[which]=core;
            }
        }
    }

    double scalar_cpb=(double)best_tsc[0]/NB;
    for(int which=0;which<2;which++){
        double cpb=(double)best_tsc[which]/NB,cpby=(double)best_tsc[which]/DATA;
        double mib=((double)DATA/1048576.0)/best_sec[which];
        const char*n=which==0?"scalar u16 (no SIMD)":"auto AVX2 u16 (x16)";
        printf("%-22s %14.2f %13.2f %11.2f %9.2fx\n",n,cpb,cpby,mib,scalar_cpb/cpb);
        if(have_perf) printf("%-22s %14.2f %13.2f  (true core cycles)\n","",
                            (double)best_core[which]/NB,(double)best_core[which]/DATA);
    }
    printf("\nsink=%016llX\n",(unsigned long long)sink);
    if(have_perf) close(pfd);
    free(buf); free(ref); return 0;
}
