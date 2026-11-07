#include <stdio.h>
#include <stdint.h>

typedef unsigned __int128 u128;

#define MASK4   0xFU
#define MASK32  0xFFFFFFFFULL
#define MASK64  0xFFFFFFFFFFFFFFFFULL

#define ROUNDS  24
static u128 mask128(void)
{
    return (((u128)MASK64) << 64) | (u128)MASK64;
}

#define MASK128 (mask128())
static const int P_box[16]    = {1, 10, 13, 12, 2, 15, 6, 11, 4, 9, 7, 8, 5, 14, 0, 3};
static const int BYTE_ROT[16] = {3, 2, 0, 3, 0, 3, 5, 4, 4, 6, 3, 4, 5, 6, 4, 6};

static uint8_t rotl8(uint8_t x, int n)
{
    n &= 7;
    if (n == 0) return x;
    return (uint8_t)((x << n) | (x >> (8 - n)));
}

static uint32_t rotl32(uint32_t x, int n)
{
    n &= 31;
    if (n == 0) return x;
    return (uint32_t)(((x << n) & MASK32) | (x >> (32 - n)));
}

static uint32_t rotr32(uint32_t x, int n)
{
    n &= 31;
    if (n == 0) return x;
    return (uint32_t)(((x >> n) | (x << (32 - n))) & MASK32);
}

static u128 rotl128(u128 x, int n)
{
    n &= 127;
    if (n == 0) return x & MASK128;
    return (((x << n) | (x >> (128 - n)))) & MASK128;
}

static uint64_t rol64(uint64_t val, int rbits)
{
    rbits %= 64;
    if (rbits == 0) return val;
    return ((val << rbits) | (val >> (64 - rbits)));
}

static u128 flip_bits128(u128 x, int start, int length)
{
    u128 mask = (((u128)1 << length) - 1) << start;
    return x ^ mask;
}

static void state_to_words(u128 state, uint32_t w[4])
{
    w[0] = (uint32_t)((state >> 96) & MASK32);
    w[1] = (uint32_t)((state >> 64) & MASK32);
    w[2] = (uint32_t)((state >> 32) & MASK32);
    w[3] = (uint32_t)(state & MASK32);
}

static u128 words_to_state(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3)
{
    return ((((u128)(w0 & MASK32)) << 96) |
            (((u128)(w1 & MASK32)) << 64) |
            (((u128)(w2 & MASK32)) << 32) |
            ((u128)(w3 & MASK32))) & MASK128;
}

static u128 Permutation(u128 state)
{
    u128 result = 0;
    int i;
    for (i = 0; i < 16; i++) {
        uint8_t byte = (uint8_t)((state >> (8 * i)) & 0xFF);
        result |= (u128)rotl8(byte, BYTE_ROT[i]) << (8 * P_box[i]);
    }
    return result & MASK128;
}

static const int _TAPS[14] = {0, 1, 4, 8, 10, 12, 13, 15, 16, 17, 18, 20, 23, 29};

static u128 TweakLFSR(u128 tweak_int)
{
    uint8_t nibbles[32];
    u128 result = 0;
    int i, k;

    for (i = 0; i < 32; i++)
        nibbles[i] = (uint8_t)((tweak_int >> (4 * i)) & MASK4);

    for (k = 0; k < 14; k++) {
        int b = _TAPS[k];
        uint8_t t = nibbles[b];
        int b3 = (t >> 3) & 1;
        int b2 = (t >> 2) & 1;
        int b1 = (t >> 1) & 1;
        int b0 = t & 1;
        nibbles[b] = (uint8_t)(((b0 ^ b1) << 3) | (b3 << 2) | (b2 << 1) | b1);
    }

    for (i = 0; i < 32; i++)
        result |= (u128)nibbles[i] << (4 * i);

    return result & MASK128;
}

static void CalcTweak(u128 tweak, int num_rounds, u128 TK[])
{
    u128 tweak_r = tweak;
    int i;
    for (i = 0; i < num_rounds; i++) {
        TK[i] = tweak_r;
        tweak_r = Permutation(tweak_r);
        tweak_r = TweakLFSR(tweak_r);
    }
}

static uint64_t alpa2_half(uint64_t x)
{
    int i;
    for (i = 0; i < 2; i++) {
        uint64_t t;
        x = (x << 28) | (x >> 36);
        t = (x >> 56) & 0xFF;
        x ^= ((((t << 1) | (t >> 7)) & 0xFF) << 24) ^ (((t << 3) & 0xFF) << 56);
    }
    return x;
}

static u128 alpa2(u128 x)
{
    uint64_t hi = (uint64_t)((x >> 64) & MASK64);
    uint64_t lo = (uint64_t)(x & MASK64);
    return (((u128)alpa2_half(hi) << 64) | (u128)alpa2_half(lo)) & MASK128;
}

static u128 F(u128 state)
{
    uint32_t x[4];
    uint32_t and1, b4, b3, and2, b1, b2;
    state_to_words(state, x);
    and1 = ~(rotl32(x[0], 10) & x[1]) & (uint32_t)MASK32;
    b4   = (and1 ^ x[3]) & (uint32_t)MASK32;
    b3   = ((rotl32(x[0], 31) & x[1]) ^ x[2]) & (uint32_t)MASK32;
    and2 = ~(rotl32(b4, 4) & b3) & (uint32_t)MASK32;
    b1   = (x[0] ^ and2) & (uint32_t)MASK32;
    b2   = (x[1] ^ (rotl32(b4, 19) & b3)) & (uint32_t)MASK32;
    return words_to_state(b1, b2, b3, b4);
}

static u128 F_inverse(u128 state)
{
    uint32_t b[4];
    uint32_t a2, and1, a1, a3, and2, a4;
    state_to_words(state, b);
    a2   = (b[1] ^ (rotl32(b[3], 19) & b[2])) & (uint32_t)MASK32;
    and1 = ~(rotl32(b[3], 4) & b[2]) & (uint32_t)MASK32;
    a1   = (and1 ^ b[0]) & (uint32_t)MASK32;
    a3   = (b[2] ^ (rotl32(a1, 31) & a2)) & (uint32_t)MASK32;
    and2 = ~(rotl32(a1, 10) & a2) & (uint32_t)MASK32;
    a4   = (b[3] ^ and2) & (uint32_t)MASK32;
    return words_to_state(a1, a2, a3, a4);
}

static void mixword1(const uint32_t in[4], uint32_t out[4])
{
    uint32_t w0 = in[0], w1 = in[1], w2 = in[2], w3 = in[3];
    uint32_t w0_rot = rotl32(w0, 12);
    uint32_t w3_rot = rotl32(w3, 9);
    uint64_t tr1 = ((uint64_t)w1 << 32) | w2;
    uint64_t mixed = rol64(tr1, 22);
    out[0] = (uint32_t)((w0_rot ^ (mixed & MASK32)) & MASK32);
    out[1] = w1 & (uint32_t)MASK32;
    out[2] = w2 & (uint32_t)MASK32;
    out[3] = (uint32_t)((w3_rot ^ ((mixed >> 32) & MASK32)) & MASK32);
}

static void mixword2(const uint32_t in[4], uint32_t out[4])
{
    uint32_t w0 = in[0], w1 = in[1], w2 = in[2], w3 = in[3];
    uint64_t tr2 = ((uint64_t)w1 << 32) | w2;
    uint64_t mixed = rol64(tr2, 22);
    uint32_t w0_new = (uint32_t)((mixed & MASK32) ^ w0);
    uint32_t w3_new = (uint32_t)(((mixed >> 32) & MASK32) ^ w3);
    out[0] = rotr32(w0_new, 12);
    out[1] = w1 & (uint32_t)MASK32;
    out[2] = w2 & (uint32_t)MASK32;
    out[3] = rotr32(w3_new, 9);
}

static void keySchedule3(const u128 MK[2], u128 Rk[], int num_rounds)
{
    u128 seed0 = MK[0] & MASK128;
    u128 seed1 = MK[1] & MASK128;
    int i;

    Rk[0] = flip_bits128(seed0, 0, 27);

    Rk[1] = flip_bits128(seed1, 0, 27);
    Rk[1] = rotl128(Rk[1], 13);
    Rk[1] = alpa2(Rk[1]) ^ (u128)1 ^ Rk[0];

    for (i = 2; i < num_rounds; i++) {
        Rk[i] = alpa2(Rk[i - 1]) ^ (u128)i ^ Rk[i - 2];
        Rk[i] = rotl128(Rk[i], 13);
    }
}

/* ---- round function ---- */

static u128 round_function(u128 state, const u128 key_new[], int round_num,
                           int last_round)
{
    uint32_t p[4];
    uint32_t tmp[4];
    u128 state_new;

    state_to_words(state, p);
    state = words_to_state(p[0], p[1], p[2], p[3]);

    if (round_num < last_round) {
        if (round_num % 2 == 0) {
            u128 f_inv;
            if (round_num != 0) {
                state = state ^ key_new[round_num];
            }
            f_inv = F_inverse(state);
            state_to_words(f_inv, p);
            mixword2(p, tmp);
            p[0] = tmp[0]; p[1] = tmp[1]; p[2] = tmp[2]; p[3] = tmp[3];
        } else {
            mixword1(p, tmp);
            p[0] = tmp[0]; p[1] = tmp[1]; p[2] = tmp[2]; p[3] = tmp[3];
            state = words_to_state(p[0], p[1], p[2], p[3]);
            state_new = F(state);
            state_new = state_new ^ key_new[round_num];
            state_to_words(state_new, p);
        }
        state_new = words_to_state(p[1], p[0], p[3], p[2]);
    } else {
        mixword1(p, tmp);
        p[0] = tmp[0]; p[1] = tmp[1]; p[2] = tmp[2]; p[3] = tmp[3];
        state = words_to_state(p[0], p[1], p[2], p[3]);
        state_new = F(state);
        state_new = state_new ^ key_new[round_num];
    }
    return state_new;
}

static void print_u128_hex(u128 v)
{
    uint64_t hi = (uint64_t)((v >> 64) & MASK64);
    uint64_t lo = (uint64_t)(v & MASK64);
    printf("%016llX%016llX", (unsigned long long)hi, (unsigned long long)lo);
}

static u128 encrypt_block(u128 plaintext, const u128 key[2], u128 tweakey,
                          int rounds, int verbose)
{
    u128 state = plaintext & MASK128;
    u128 RK[ROUNDS];
    u128 TK[ROUNDS];
    u128 RTK[ROUNDS];
    u128 KSEQ[ROUNDS];
    u128 MK_full, RK0;
    int i, last_round;

    keySchedule3(key, RK, rounds);
    CalcTweak(tweakey, rounds, TK);

    for (i = 0; i < rounds; i++)
        RTK[i] = (RK[i] ^ TK[i]) & MASK128;
    MK_full = (key[0] ^ key[1]) & MASK128;
    RK0 = (MK_full ^ tweakey) & MASK128;

    KSEQ[0] = RK0;
    for (i = 1; i < rounds; i++)
        KSEQ[i] = RTK[i];

    state ^= KSEQ[0];
    state &= MASK128;

    if (verbose) {
        printf("After initial whitening: ");
        print_u128_hex(state);
        printf("\n");
    }

    last_round = rounds - 1;
    for (i = 0; i < rounds; i++) {
        state = round_function(state, KSEQ, i, last_round);
        if (verbose) {
            printf("Round %2d state: ", i);
            print_u128_hex(state);
            printf("\n");
        }
    }
    return state & MASK128;
}

static u128 mk128(uint64_t hi, uint64_t lo)
{
    return (((u128)hi) << 64) | (u128)lo;
}

void l2crd128_setup(const u128 key[2], u128 tweakey, int rounds, u128 KSEQ[])
{
    u128 RK[ROUNDS];
    u128 TK[ROUNDS];
    u128 MK_full;
    int i;

    if (rounds > ROUNDS)
        rounds = ROUNDS;

    keySchedule3(key, RK, rounds);
    CalcTweak(tweakey, rounds, TK);

    MK_full = (key[0] ^ key[1]) & MASK128;
    KSEQ[0] = (MK_full ^ tweakey) & MASK128;

    for (i = 1; i < rounds; i++)
        KSEQ[i] = (RK[i] ^ TK[i]) & MASK128;
}

u128 l2crd128_encrypt_with_kseq(u128 plaintext, const u128 KSEQ[], int rounds)
{
    u128 state = plaintext & MASK128;
    int i, last_round;

    state ^= KSEQ[0];
    state &= MASK128;

    last_round = rounds - 1;
    for (i = 0; i < rounds; i++)
        state = round_function(state, KSEQ, i, last_round);

    return state & MASK128;
}

#ifndef L2C_RD128_NO_MAIN
int main(void)
{
    u128 plaintext = mk128(0x0022332233223322ULL, 0x0022332233223322ULL);
    u128 key[2] = {
        mk128(0x0000000000000000ULL, 0x0000000000000000ULL),  /* K0 */
        mk128(0x0000000000000000ULL, 0x0000000000000000ULL)   /* K1 */
    };
    u128 tweakey = mk128(0x1111111111111111ULL, 0x1111111111111111ULL);
    u128 ct;

    printf("l2crd-128-256  (128-bit block / 256-bit key)\n");
    printf("Plaintext:  ");
    print_u128_hex(plaintext);
    printf("\n");
    printf("Key K0:     ");
    print_u128_hex(key[0]);
    printf("\n");
    printf("Key K1:     ");
    print_u128_hex(key[1]);
    printf("\n");
    printf("------------------------------------------------\n");

    ct = encrypt_block(plaintext, key, tweakey, ROUNDS, 1);

    printf("------------------------------------------------\n");
    printf("Ciphertext: ");
    print_u128_hex(ct);
    printf("\n");
    return 0;
}
#endif
