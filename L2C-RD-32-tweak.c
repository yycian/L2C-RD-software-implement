#include <stdio.h>
#include <stdint.h>
#define MASK8   0xFFU
#define MASK16  0xFFFFU
#define MASK32  0xFFFFFFFFU
#define ROUNDS  14

static const int P_box[16] __attribute__((unused)) = {
    0xA, 0x2, 0x1, 0x9, 0x6, 0xC, 0x4, 0x8,
    0x7, 0x3, 0x0, 0xF, 0x5, 0xE, 0xD, 0xB
};

static uint8_t rotl8(uint8_t x, int n)
{
    n &= 7;
    if (n == 0) return x;
    return (uint8_t)(((x << n) & MASK8) | (x >> (8 - n)));
}

static uint8_t rotr8(uint8_t x, int n)
{
    n &= 7;
    if (n == 0) return x;
    return (uint8_t)(((x >> n) & MASK8) | ((x << (8 - n)) & MASK8));
}

static uint32_t rotl32(uint32_t x, int n)
{
    n &= 31;
    if (n == 0) return x;
    return (uint32_t)(((x << n) | (x >> (32 - n))) & MASK32);
}

static uint16_t rol16(uint16_t val, int rbits)
{
    rbits %= 16;
    if (rbits == 0) return val;
    return (uint16_t)(((val << rbits) | (val >> (16 - rbits))));
}

static uint32_t flip_bits(uint32_t x, int start, int length)
{
    uint32_t mask = (((1U << length) - 1U) << start);
    return x ^ mask;
}

static void state_to_words(uint32_t state, uint8_t w[4])
{
    w[0] = (uint8_t)((state >> 24) & MASK8);
    w[1] = (uint8_t)((state >> 16) & MASK8);
    w[2] = (uint8_t)((state >> 8) & MASK8);
    w[3] = (uint8_t)(state & MASK8);
}

static uint32_t words_to_state(uint8_t w0, uint8_t w1, uint8_t w2, uint8_t w3)
{
    return (((uint32_t)w0 << 24) |
            ((uint32_t)w1 << 16) |
            ((uint32_t)w2 << 8)  |
            ((uint32_t)w3)) & MASK32;
}

static const int BYTE_P[4]   = {3, 2, 1, 0};
static const int BYTE_ROT[4] = {3, 2, 2, 5};
static const int TWEAK_EDGES[6][2] = {
    {2, 13}, {5, 1}, {10, 29}, {12, 21}, {26, 22}, {27, 20}
};

static uint32_t Permutation(uint32_t state)
{
    uint32_t result = 0;
    int i;
    for (i = 0; i < 4; i++) {
        uint8_t byte = (uint8_t)((state >> (8 * i)) & MASK8);
        result |= (uint32_t)rotl8(byte, BYTE_ROT[i]) << (8 * BYTE_P[i]);
    }
    return result & MASK32;
}

static uint32_t TweakLFSR(uint32_t tweak_int)
{
    uint32_t result = tweak_int;
    int k;
    for (k = 0; k < 6; k++) {
        int dest = TWEAK_EDGES[k][0], source = TWEAK_EDGES[k][1];
        result ^= ((tweak_int >> source) & 1U) << dest;
    }
    return result & MASK32;
}

static void CalcTweak(uint32_t tweak, int num_rounds, uint32_t TK[])
{
    uint32_t tweak_r = tweak;
    int i;
    for (i = 0; i < num_rounds; i++)
        TK[i] = 0;
    for (i = 1; i < num_rounds; i++) {
        tweak_r = Permutation(tweak_r);
        tweak_r = TweakLFSR(tweak_r);
        TK[i] = tweak_r;
    }
}

static uint32_t F(uint32_t state)
{
    uint8_t x[4];
    uint8_t and1, b4, b3, and2, b1, b2;
    state_to_words(state, x);
    and1 = (uint8_t)(~(rotl8(x[0], 3) & x[1]) & MASK8);
    b4   = (uint8_t)((and1 ^ x[3]) & MASK8);
    b3   = (uint8_t)(((rotl8(x[0], 2) & x[1]) ^ x[2]) & MASK8);
    and2 = (uint8_t)(~(rotl8(b4, 3) & b3) & MASK8);
    b1   = (uint8_t)((x[0] ^ and2) & MASK8);
    b2   = (uint8_t)((x[1] ^ (rotl8(b4, 0) & b3)) & MASK8);
    return words_to_state(b1, b2, b3, b4);
}

static uint32_t F_inverse(uint32_t state)
{
    uint8_t b[4];
    uint8_t a2, and1, a1, a3, and2, a4;
    state_to_words(state, b);
    a2   = (uint8_t)((b[1] ^ (rotl8(b[3], 0) & b[2])) & MASK8);
    and1 = (uint8_t)(~(rotl8(b[3], 3) & b[2]) & MASK8);
    a1   = (uint8_t)((and1 ^ b[0]) & MASK8);
    a3   = (uint8_t)((b[2] ^ (rotl8(a1, 2) & a2)) & MASK8);
    and2 = (uint8_t)(~(rotl8(a1, 3) & a2) & MASK8);
    a4   = (uint8_t)((b[3] ^ and2) & MASK8);
    return words_to_state(a1, a2, a3, a4);
}


static uint32_t alpa2(uint32_t x)
{
   
    int i;
    for (i = 0; i < 2; i++) {
        uint32_t t;
        x = rotl32(x, 14);
        t = (x >> 24) & 0xFF;
        x ^= ((uint32_t)rotl8((uint8_t)t, 1) << 8) ^ (((t << 3) & 0xFF) << 24);
    }
    return x;
}


static void keySchedule3(const uint32_t MK[2], uint32_t Rk[], int num_rounds)
{
    uint32_t rk_prev2, rk_prev1;
    int i;
    Rk[0] = 0;
    rk_prev2 = flip_bits(MK[0], 0, 7);
    rk_prev2 = rotl32(rk_prev2, 3);           
    Rk[1] = flip_bits(MK[1], 0, 7);
    Rk[1] = rotl32(Rk[1], 3);
    Rk[1] = alpa2(Rk[1]) ^ 1U ^ rk_prev2;
    rk_prev1 = Rk[1];

    for (i = 2; i < num_rounds; i++) {
        uint32_t cur = alpa2(rk_prev1) ^ (uint32_t)i ^ rk_prev2;
        Rk[i] = cur;
        rk_prev2 = rk_prev1;
        rk_prev1 = cur;
    }
}

static void mixword1(const uint8_t in[4], uint8_t out[4])
{
    uint8_t w0 = in[0], w1 = in[1], w2 = in[2], w3 = in[3];
    uint8_t w0_rot = rotl8(w0, 2);
    uint8_t w3_rot = rotl8(w3, 2);
    uint16_t tr1 = (uint16_t)(((uint16_t)w1 << 8) | w2);
    uint16_t mixed = rol16(tr1, 1);
    out[0] = (uint8_t)((w0_rot ^ (mixed & MASK8)) & MASK8);
    out[1] = (uint8_t)(w1 & MASK8);
    out[2] = (uint8_t)(w2 & MASK8);
    out[3] = (uint8_t)((w3_rot ^ ((mixed >> 8) & MASK8)) & MASK8);
}

static void mixword2(const uint8_t in[4], uint8_t out[4])
{
    uint8_t w0 = in[0], w1 = in[1], w2 = in[2], w3 = in[3];
    uint16_t tr2 = (uint16_t)(((uint16_t)w1 << 8) | w2);
    uint16_t mixed = rol16(tr2, 1);
    uint8_t w0_new = (uint8_t)((mixed & MASK8) ^ w0);
    uint8_t w3_new = (uint8_t)(((mixed >> 8) & MASK8) ^ w3);
    out[0] = rotr8(w0_new, 2);
    out[1] = (uint8_t)(w1 & MASK8);
    out[2] = (uint8_t)(w2 & MASK8);
    out[3] = rotr8(w3_new, 2);
}

static uint32_t round_function(uint32_t state, const uint32_t key_new[], int round_num, int last_round)
{
    uint8_t p[4];
    uint8_t tmp[4];
    uint32_t state_new;

    state_to_words(state, p);
    state = words_to_state(p[0], p[1], p[2], p[3]);

    if (round_num < last_round) {
        if (round_num % 2 == 0) {
            uint32_t f_inv;
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

static __attribute__((unused)) uint32_t encrypt_block(uint32_t plaintext, const uint32_t key[2],
                              uint32_t tweakey, int rounds, int verbose,
                              int decrypt)
{
    uint32_t state = plaintext & MASK32;
    uint32_t RK[ROUNDS];
    uint32_t TK[ROUNDS];
    uint32_t RTK[ROUNDS];
    uint32_t KSEQ[ROUNDS];
    uint32_t RK0;
    int i;

    keySchedule3(key, RK, rounds);
    CalcTweak(tweakey, rounds, TK);

    for (i = 0; i < rounds; i++)
        RTK[i] = (RK[i] ^ TK[i]) & MASK32;

    RK0 = key[0] & MASK32;

    KSEQ[0] = RK0;
    for (i = 1; i < rounds; i++)
        KSEQ[i] = RTK[i];

    if (decrypt) {
        for (i = 0; i < rounds / 2; i++) {
            uint32_t t = KSEQ[i];
            KSEQ[i] = KSEQ[rounds - 1 - i];
            KSEQ[rounds - 1 - i] = t;
        }
    }

    state ^= KSEQ[0];
    state &= MASK32;

    if (verbose) {
        if (decrypt)
            printf("After first xor in decrypt: %08X\n", state);
        else
            printf("After initial whitening:   %08X\n", state);
    }

    for (i = 0; i < rounds; i++) {
        state = round_function(state, KSEQ, i, rounds - 1);
        if (verbose) {
            if (decrypt)
                printf("Decrypt Round %2d state: %08X\n", i, state);
            else
                printf("Encrypt Round %2d state: %08X\n", i, state);
        }
    }
    return state & MASK32;
}

void l2crd32_setup(const uint32_t key[2], uint32_t tweakey,
                   int rounds, uint32_t KSEQ[])
{
    uint32_t RK[ROUNDS];
    uint32_t TK[ROUNDS];
    int i;

    if (rounds > ROUNDS)
        rounds = ROUNDS;

    keySchedule3(key, RK, rounds);
    CalcTweak(tweakey, rounds, TK);

    KSEQ[0] = key[0] & MASK32;
    for (i = 1; i < rounds; i++)
        KSEQ[i] = (RK[i] ^ TK[i]) & MASK32;
}

uint32_t l2crd32_encrypt_with_kseq(uint32_t plaintext, const uint32_t KSEQ[],
                                   int rounds)
{
    uint32_t state = plaintext & MASK32;
    int i;

    state ^= KSEQ[0];
    state &= MASK32;
    for (i = 0; i < rounds; i++)
        state = round_function(state, KSEQ, i, rounds - 1);
    return state & MASK32;
}

#ifndef L2C_RD32_NO_MAIN
int main(void)
{
    uint32_t plaintext = 0x33300000U;
    uint32_t key[2] = {0x00000000U, 0x00000000U};
    uint32_t tweakey = 0x00033333U;
    uint32_t ct, pt;

    printf("32-bit Block Cipher\n");
    printf("Plaintext: %08X\n", plaintext);
    printf("----------------------------------------\n");

    ct = encrypt_block(plaintext, key, tweakey, ROUNDS, 1, 0);

    printf("----------------------------------------\n");
    printf("Ciphertext: %08X\n", ct);

    pt = encrypt_block(ct, key, tweakey, ROUNDS, 1, 1);

    printf("----------------------------------------\n");
    printf("Plaintext: %08X\n", pt);
    return 0;
}
#endif
