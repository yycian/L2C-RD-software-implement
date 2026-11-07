#include <stdio.h>
#include <stdint.h>
#define MASK4   0xFULL
#define MASK16  0xFFFFULL
#define MASK64  0xFFFFFFFFFFFFFFFFULL
#define ROUNDS  18

static const int P_box[16] = {
    0xA, 0x2, 0x7, 0x9, 0x6, 0xC, 0x4, 0x8,
    0x1, 0x3, 0x0, 0xD, 0x5, 0xE, 0xF, 0xB
};

static uint16_t rotl16(uint16_t x, int n)
{
    n &= 15;
    if (n == 0) return x;
    return (uint16_t)(((x << n) & MASK16) | (x >> (16 - n)));
}

static uint16_t rotr16(uint16_t x, int n)
{
    n &= 15;
    if (n == 0) return x;
    return (uint16_t)(((x >> n) & MASK16) | ((x << (16 - n)) & MASK16));
}

static uint64_t rotl64(uint64_t x, int n)
{
    n &= 63;
    if (n == 0) return x;
    return ((x << n) | (x >> (64 - n))) & MASK64;
}

static uint32_t rol32(uint32_t val, int rbits)
{
    rbits %= 32;
    if (rbits == 0) return val;
    return (uint32_t)(((val << rbits) | (val >> (32 - rbits))));
}


static uint64_t flip_bits(uint64_t x, int start, int length)
{
    uint64_t mask = (((1ULL << length) - 1ULL) << start);
    return x ^ mask;
}

static void state_to_words(uint64_t state, uint16_t w[4])
{
    w[0] = (uint16_t)((state >> 48) & MASK16);
    w[1] = (uint16_t)((state >> 32) & MASK16);
    w[2] = (uint16_t)((state >> 16) & MASK16);
    w[3] = (uint16_t)(state & MASK16);
}

static uint64_t words_to_state(uint16_t w0, uint16_t w1, uint16_t w2, uint16_t w3)
{
    return (((uint64_t)w0 << 48) |
            ((uint64_t)w1 << 32) |
            ((uint64_t)w2 << 16) |
            ((uint64_t)w3)) & MASK64;
}

static uint64_t Permutation(uint64_t state)
{
    uint64_t result = 0;
    int i;
    for (i = 0; i < 16; i++) {
        uint64_t nibble = (state >> (4 * i)) & MASK4;
        result |= nibble << (4 * P_box[i]);
    }
    return result & MASK64;
}

static uint64_t TweakLFSR(uint64_t tweak_int)
{
    uint8_t nibbles[16];
    static const int taps[7] = {0, 1, 3, 4, 7, 8, 11};
    int i, k;
    uint64_t result = 0;

    for (i = 0; i < 16; i++)
        nibbles[i] = (uint8_t)((tweak_int >> (4 * i)) & MASK4);

    for (k = 0; k < 7; k++) {
        int b = taps[k];
        uint8_t t = nibbles[b];
        int b3 = (t >> 3) & 1;
        int b2 = (t >> 2) & 1;
        int b1 = (t >> 1) & 1;
        int b0 = t & 1;
        nibbles[b] = (uint8_t)(((b0 ^ b1) << 3) | (b3 << 2) | (b2 << 1) | b1);
    }

    for (i = 0; i < 16; i++)
        result |= (uint64_t)nibbles[i] << (4 * i);

    return result & MASK64;
}

static void CalcTweak(uint64_t tweak, int num_rounds, uint64_t TK[])
{
    uint64_t tweak_r = tweak;
    int i;
    for (i = 0; i < num_rounds; i++) {
        TK[i] = tweak_r;
        tweak_r = Permutation(tweak_r);
        tweak_r = TweakLFSR(tweak_r);
    }
}

static uint64_t alpa2(uint64_t x)
{
    int i;
    for (i = 0; i < 2; i++) {
        uint64_t t;
        x = rotl64(x, 28);
        t = (x >> 56) & 0xFF;
        x ^= ((((t << 1) | (t >> 7)) & 0xFF) << 56) ^ (((t << 3) & 0xFF) << 24);
    }
    return x;
}

static void keySchedule3(const uint64_t MK[2], uint64_t Rk[], int num_rounds)
{
    int i;
    Rk[0] = flip_bits(MK[0], 0, 14);
    Rk[0] = rotl64(Rk[0], 7);
    Rk[1] = flip_bits(MK[1], 0, 14);
    Rk[1] = rotl64(Rk[1], 7);
    Rk[1] = alpa2(Rk[1]) ^ 1ULL ^ Rk[0];
    for (i = 2; i < num_rounds; i++) {
        Rk[i] = alpa2(Rk[i - 1]) ^ (uint64_t)i ^ Rk[i - 2];
        Rk[i] = rotl64(Rk[i], 7);
    }
}

static uint64_t F(uint64_t state)
{
    uint16_t x[4];
    uint16_t and1, b4, b3, and2, b1, b2;
    state_to_words(state, x);
    and1 = (uint16_t)(~(rotl16(x[0], 3) & x[1]) & MASK16);
    b4   = (uint16_t)((and1 ^ x[3]) & MASK16);
    b3   = (uint16_t)(((rotl16(x[0], 10) & x[1]) ^ x[2]) & MASK16);
    and2 = (uint16_t)(~(rotl16(b4, 3) & b3) & MASK16);
    b1   = (uint16_t)((x[0] ^ and2) & MASK16);
    b2   = (uint16_t)((x[1] ^ (rotl16(b4, 4) & b3)) & MASK16);
    return words_to_state(b1, b2, b3, b4);
}

static uint64_t F_inverse(uint64_t state)
{
    uint16_t b[4];
    uint16_t a2, and1, a1, a3, and2, a4;
    state_to_words(state, b);
    a2   = (uint16_t)((b[1] ^ (rotl16(b[3], 4) & b[2])) & MASK16);
    and1 = (uint16_t)(~(rotl16(b[3], 3) & b[2]) & MASK16);
    a1   = (uint16_t)((and1 ^ b[0]) & MASK16);
    a3   = (uint16_t)((b[2] ^ (rotl16(a1, 10) & a2)) & MASK16);
    and2 = (uint16_t)(~(rotl16(a1, 3) & a2) & MASK16);
    a4   = (uint16_t)((b[3] ^ and2) & MASK16);
    return words_to_state(a1, a2, a3, a4);
}

static void mixword1(uint16_t in[4], uint16_t out[4])
{
    uint16_t w0 = in[0], w1 = in[1], w2 = in[2], w3 = in[3];
    uint16_t w0_rot = rotl16(w0, 5);
    uint16_t w3_rot = rotl16(w3, 9);
    uint32_t tr1 = ((uint32_t)w1 << 16) | w2;
    uint32_t mixed = rol32(tr1, 11);
    out[0] = (uint16_t)((w0_rot ^ (mixed & MASK16)) & MASK16);
    out[1] = (uint16_t)(w1 & MASK16);
    out[2] = (uint16_t)(w2 & MASK16);
    out[3] = (uint16_t)((w3_rot ^ ((mixed >> 16) & MASK16)) & MASK16);
}

static void mixword2(uint16_t in[4], uint16_t out[4])
{
    uint16_t w0 = in[0], w1 = in[1], w2 = in[2], w3 = in[3];
    uint32_t tr2 = ((uint32_t)w1 << 16) | w2;
    uint32_t mixed = rol32(tr2, 11);
    uint16_t w0_new = (uint16_t)((mixed & MASK16) ^ w0);
    uint16_t w3_new = (uint16_t)(((mixed >> 16) & MASK16) ^ w3);
    out[0] = rotr16(w0_new, 5);
    out[1] = (uint16_t)(w1 & MASK16);
    out[2] = (uint16_t)(w2 & MASK16);
    out[3] = rotr16(w3_new, 9);
}

static uint64_t round_function(uint64_t state, const uint64_t key_new[], int round_num,
                              int last_round)
{
    uint16_t p[4];
    uint16_t tmp[4];
    uint64_t state_new;
    state_to_words(state, p);
    state = words_to_state(p[0], p[1], p[2], p[3]);
    if (round_num < last_round) {
        if (round_num % 2 == 0) {
            uint64_t f_inv;
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

void l2crd64_setup(const uint64_t key[2], uint64_t tweakey,
                   int rounds, uint64_t KSEQ[])
{
    uint64_t RK[ROUNDS];
    uint64_t TK[ROUNDS];
    int i;

    if (rounds > ROUNDS)
        rounds = ROUNDS;

    keySchedule3(key, RK, rounds);
    CalcTweak(tweakey, rounds, TK);

    KSEQ[0] = (key[0] ^ tweakey) & MASK64;
    for (i = 1; i < rounds; i++)
        KSEQ[i] = (RK[i] ^ TK[i]) & MASK64;
}

uint64_t l2crd64_encrypt_with_kseq(uint64_t plaintext, const uint64_t KSEQ[],
                                   int rounds)
{
    uint64_t state = plaintext & MASK64;
    int i;

    state ^= KSEQ[0];
    state &= MASK64;
    for (i = 0; i < rounds; i++)
        state = round_function(state, KSEQ, i, rounds - 1);
    return state & MASK64;
}

uint64_t encrypt_block(uint64_t plaintext, const uint64_t key[2],
                       uint64_t tweakey, int rounds, int verbose, int decrypt)
{
    uint64_t RK[ROUNDS];
    uint64_t TK[ROUNDS];
    uint64_t KSEQ[ROUNDS];
    uint64_t state;
    int i;

    keySchedule3(key, RK, rounds);

    if (verbose) {
        printf("RK: [");
        for (i = 0; i < rounds; i++)
            printf("%s'0x%llx'", (i ? ", " : ""), (unsigned long long)RK[i]);
        printf("]\n");
    }

    CalcTweak(tweakey, rounds, TK);

    KSEQ[0] = (key[0] ^ tweakey) & MASK64;
    for (i = 1; i < rounds; i++)
        KSEQ[i] = (RK[i] ^ TK[i]) & MASK64;

    if (decrypt) {
        for (i = 0; i < rounds / 2; i++) {
            uint64_t t = KSEQ[i];
            KSEQ[i] = KSEQ[rounds - 1 - i];
            KSEQ[rounds - 1 - i] = t;
        }
    }

    if (verbose) {
        state = (plaintext ^ KSEQ[0]) & MASK64;
        printf(decrypt ? "After first xor in decrypt: %016llX\n"
                       : "After initial whitening:   %016llX\n", (unsigned long long)state);
        for (i = 0; i < rounds; i++) {
            state = round_function(state, KSEQ, i, rounds - 1);
            printf(decrypt ? "Decrypt Round %2d state: %016llX\n"
                           : "Encrypt Round %2d state: %016llX\n", i, (unsigned long long)state);
        }
        return state & MASK64;
    }

    return l2crd64_encrypt_with_kseq(plaintext, KSEQ, rounds);
}

#ifndef L2C_RD_NO_MAIN
int main(void)
{
    uint64_t plaintext = 0x0022332233223322ULL;
    uint64_t key[2] = {0x0000000000000000ULL, 0x0000000000000000ULL};
    uint64_t tweakey = 0x1111111111111111ULL;
    uint64_t ct;

    printf("64-bit Block Cipher\n");
    printf("Plaintext: %016llX\n", (unsigned long long)plaintext);
    printf("----------------------------------------\n");

    ct = encrypt_block(plaintext, key, tweakey, ROUNDS, 1, 0);

    printf("----------------------------------------\n");
    printf("Ciphertext: %016llX\n", (unsigned long long)ct);
    return 0;
}
#endif
