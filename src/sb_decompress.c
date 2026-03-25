/*
 * Alien Breed 3D I - PC Port
 * sb_decompress.c - =SB= (LHA/LH5) decompressor
 *
 * Decompresses data in the =SB= format used by AB3D for level data,
 * wall textures, and other game assets.
 *
 * The LH5 decode algorithm is a direct port of the decode routines from
 * jca02266/lha (LHa for UNIX), adapted to work on in-memory buffers
 * instead of files. The original code is by Nobutaka Watazaki / t.okamoto.
 *
 * Original license:  Public domain / distribution allowed
 */

#include "sb_decompress.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logging.h"
#define printf ab3d_log_printf

/* =======================================================================
 * =SB= header parsing
 * ======================================================================= */

static uint32_t read_be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}

int sb_is_compressed(const uint8_t *data, size_t data_len)
{
    if (data_len < 12) return 0;
    return data[0] == '=' && data[1] == 'S' &&
           data[2] == 'B' && data[3] == '=';
}

uint32_t sb_unpacked_size(const uint8_t *data, size_t data_len)
{
    if (!sb_is_compressed(data, data_len)) return 0;
    return read_be32(data + 4);
}

/* =======================================================================
 * LH5 parameters (from lha_macro.h)
 * ======================================================================= */

/* =SB= uses LH6-compatible parameters: 32K sliding dictionary (DICBIT=15),
 * pbit=5, np=16.  The Amiga WorkSpace buffer is 32K (ds.l 8192 = 32768 bytes)
 * which fits a 32K ring buffer exactly.  This matches LH6 in jca02266/lha. */
#define SB_DICBIT     15        /* 2^15 = 32768 byte sliding dictionary */
#define DICSIZ        (1U << SB_DICBIT)
#define MAXMATCH      256
#define THRESHOLD     3
#define NC            (255 + MAXMATCH + 2 - THRESHOLD) /* 510 */
#define USHRT_BIT     16
#define NT            (USHRT_BIT + 3)                  /* 19 */
#define NP_SB         (SB_DICBIT + 1)                  /* 16 */
#define TBIT          5
#define CBIT          9
#define PBIT_SB       5         /* 5-bit position code count field */
#define NPT           0x80   /* buffer size for pt_len (>= max(NT,NP)) */

#ifndef MIN
#define MIN(a,b) ((a) <= (b) ? (a) : (b))
#endif

/* =======================================================================
 * LH5 decoder state (replaces global variables from reference impl)
 * ======================================================================= */

typedef struct {
    /* Bitstream state (from bitio.c) */
    const uint8_t *inbuf;
    size_t         insize;
    size_t         inpos;
    uint16_t       bitbuf;
    uint8_t        subbitbuf;
    uint8_t        bitcount;

    /* Huffman tables (from huf.c) */
    uint16_t left[2 * NC - 1];
    uint16_t right[2 * NC - 1];
    uint16_t c_table[4096];
    uint16_t pt_table[256];
    uint8_t  c_len[NC];
    uint8_t  pt_len[NPT];
    uint16_t blocksize;
    int      np;
    int      pbit;

    /* Output ring buffer (sliding dictionary) */
    uint8_t  dtext[DICSIZ];
    uint32_t loc;
} LH5State;

/* =======================================================================
 * Bitstream I/O (adapted from bitio.c to use memory buffer)
 * ======================================================================= */

static void fillbuf(LH5State *s, int n)
{
    while (n > s->bitcount) {
        n -= s->bitcount;
        s->bitbuf = (uint16_t)((s->bitbuf << s->bitcount) +
                    (s->subbitbuf >> (8 - s->bitcount)));
        if (s->inpos < s->insize) {
            s->subbitbuf = s->inbuf[s->inpos++];
        } else {
            s->subbitbuf = 0;
        }
        s->bitcount = 8;
    }
    s->bitcount = (uint8_t)(s->bitcount - n);
    s->bitbuf = (uint16_t)((s->bitbuf << n) +
                (s->subbitbuf >> (8 - n)));
    s->subbitbuf = (uint8_t)(s->subbitbuf << n);
}

static uint16_t getbits(LH5State *s, int n)
{
    uint16_t x = (uint16_t)(s->bitbuf >> (16 - n));
    fillbuf(s, n);
    return x;
}

#define peekbits(s, n) ((s)->bitbuf >> (16 - (n)))

static void init_getbits(LH5State *s)
{
    s->bitbuf = 0;
    s->subbitbuf = 0;
    s->bitcount = 0;
    fillbuf(s, 16);
}

/* =======================================================================
 * make_table (from maketbl.c - builds decoding lookup table)
 * ======================================================================= */

static int make_table(LH5State *s, int nchar, uint8_t bitlen[],
                      int tablebits, uint16_t table[])
{
    uint16_t count[17];
    uint16_t weight[17];
    uint16_t start[17];
    uint16_t total;
    unsigned int i;
    int j, k, m, n, avail;
    uint16_t *p;

    avail = nchar;

    for (i = 1; i <= 16; i++) {
        count[i] = 0;
        weight[i] = (uint16_t)(1 << (16 - i));
    }

    for (i = 0; i < (unsigned)nchar; i++) {
        if (bitlen[i] > 16) {
            return -1;  /* bad table */
        }
        count[bitlen[i]]++;
    }

    total = 0;
    for (i = 1; i <= 16; i++) {
        start[i] = total;
        total = (uint16_t)(total + weight[i] * count[i]);
    }
    if ((total & 0xffff) != 0) {
        return -1;  /* bad table */
    }

    m = 16 - tablebits;
    for (i = 1; i <= (unsigned)tablebits; i++) {
        start[i] >>= m;
        weight[i] >>= m;
    }

    j = (int)(start[tablebits + 1] >> m);
    k = MIN(1 << tablebits, 4096);
    if (j != 0) {
        for (i = (unsigned)j; i < (unsigned)k; i++)
            table[i] = 0;
    }

    for (j = 0; j < nchar; j++) {
        k = bitlen[j];
        if (k == 0) continue;

        unsigned int l = start[k] + weight[k];
        if (k <= tablebits) {
            l = MIN(l, 4096);
            for (i = start[k]; i < l; i++)
                table[i] = (uint16_t)j;
        } else {
            i = start[k];
            if ((i >> m) > 4096) {
                return -1;  /* bad table */
            }
            p = &table[i >> m];
            i <<= tablebits;
            n = k - tablebits;
            while (--n >= 0) {
                if (*p == 0) {
                    s->right[avail] = s->left[avail] = 0;
                    *p = (uint16_t)avail++;
                }
                if (i & 0x8000)
                    p = &s->right[*p];
                else
                    p = &s->left[*p];
                i <<= 1;
            }
            *p = (uint16_t)j;
        }
        start[k] = (uint16_t)l;
    }

    return 0;
}

/* =======================================================================
 * Huffman table reading (from huf.c)
 * ======================================================================= */

static int read_pt_len(LH5State *s, int nn, int nbit, int i_special)
{
    int i, c, n;

    n = getbits(s, nbit);
    if (n == 0) {
        c = getbits(s, nbit);
        for (i = 0; i < nn; i++)
            s->pt_len[i] = 0;
        for (i = 0; i < 256; i++)
            s->pt_table[i] = (uint16_t)c;
    } else {
        i = 0;
        while (i < MIN(n, NPT)) {
            c = peekbits(s, 3);
            if (c != 7)
                fillbuf(s, 3);
            else {
                unsigned short mask = 1 << (16 - 4);
                while (mask & s->bitbuf) {
                    mask >>= 1;
                    c++;
                }
                fillbuf(s, c - 3);
            }

            s->pt_len[i++] = (uint8_t)c;
            if (i == i_special) {
                c = getbits(s, 2);
                while (--c >= 0 && i < NPT)
                    s->pt_len[i++] = 0;
            }
        }
        while (i < nn)
            s->pt_len[i++] = 0;
        if (make_table(s, nn, s->pt_len, 8, s->pt_table) < 0)
            return -1;
    }
    return 0;
}

static int read_c_len(LH5State *s)
{
    int i, c, n;

    n = getbits(s, CBIT);
    if (n == 0) {
        c = getbits(s, CBIT);
        for (i = 0; i < NC; i++)
            s->c_len[i] = 0;
        for (i = 0; i < 4096; i++)
            s->c_table[i] = (uint16_t)c;
    } else {
        i = 0;
        while (i < MIN(n, NC)) {
            c = s->pt_table[peekbits(s, 8)];
            if (c >= NT) {
                unsigned short mask = 1 << (16 - 9);
                do {
                    if (s->bitbuf & mask)
                        c = s->right[c];
                    else
                        c = s->left[c];
                    mask >>= 1;
                } while (c >= NT && (mask || c != (int)s->left[c]));
            }
            fillbuf(s, s->pt_len[c]);
            if (c <= 2) {
                if (c == 0)
                    c = 1;
                else if (c == 1)
                    c = getbits(s, 4) + 3;
                else
                    c = getbits(s, CBIT) + 20;
                while (--c >= 0)
                    s->c_len[i++] = 0;
            } else {
                s->c_len[i++] = (uint8_t)(c - 2);
            }
        }
        while (i < NC)
            s->c_len[i++] = 0;
        if (make_table(s, NC, s->c_len, 12, s->c_table) < 0)
            return -1;
    }
    return 0;
}

/* =======================================================================
 * LH5 decode functions (from huf.c)
 * ======================================================================= */

/* Decode next character/length code */
static int decode_c(LH5State *s)
{
    if (s->blocksize == 0) {
        s->blocksize = getbits(s, 16);
        if (read_pt_len(s, NT, TBIT, 3) < 0) return -1;
        if (read_c_len(s) < 0) return -1;
        if (read_pt_len(s, s->np, s->pbit, -1) < 0) return -1;
    }
    s->blocksize--;

    unsigned int j = s->c_table[peekbits(s, 12)];
    if (j < NC)
        fillbuf(s, s->c_len[j]);
    else {
        fillbuf(s, 12);
        unsigned short mask = 1 << (16 - 1);
        do {
            if (s->bitbuf & mask)
                j = s->right[j];
            else
                j = s->left[j];
            mask >>= 1;
        } while (j >= NC && (mask || j != s->left[j]));
        fillbuf(s, s->c_len[j] - 12);
    }
    return (int)j;
}

/* Decode position/offset */
static int decode_p(LH5State *s)
{
    unsigned int j = s->pt_table[peekbits(s, 8)];
    if (j < (unsigned)s->np)
        fillbuf(s, s->pt_len[j]);
    else {
        fillbuf(s, 8);
        unsigned short mask = 1 << (16 - 1);
        do {
            if (s->bitbuf & mask)
                j = s->right[j];
            else
                j = s->left[j];
            mask >>= 1;
        } while (j >= (unsigned)s->np && (mask || j != s->left[j]));
        fillbuf(s, s->pt_len[j] - 8);
    }
    if (j != 0)
        j = (1U << (j - 1)) + getbits(s, (int)(j - 1));
    return (int)j;
}

/* =======================================================================
 * Main decompression function
 * ======================================================================= */

size_t sb_decompress(const uint8_t *src, size_t src_len,
                     uint8_t *dst, size_t dst_len)
{
    if (!sb_is_compressed(src, src_len))
        return 0;

    uint32_t unpacked = read_be32(src + 4);

    if (unpacked > dst_len)
        return 0;

    /* Allocate decoder state on heap */
    LH5State *s = (LH5State *)calloc(1, sizeof(LH5State));
    if (!s) return 0;

    /* Skip 12-byte =SB= header to get to the LH5 compressed bitstream */
    s->inbuf = src + 12;
    s->insize = src_len - 12;
    s->inpos = 0;

    /* =SB= uses LH6 parameters: DICBIT=15, np=16, pbit=5 */
    s->np = NP_SB;       /* 15 */
    s->pbit = PBIT_SB;   /* 5 */

    /* Initialize ring buffer with spaces (as per reference lha) */
    memset(s->dtext, 0x20, DICSIZ);
    s->loc = 0;

    /* Initialize bitstream */
    s->blocksize = 0;
    init_getbits(s);

    /* Decode loop */
    size_t out_pos = 0;
    while (out_pos < unpacked) {
        int c = decode_c(s);
        if (c < 0) break;

        if (c < 256) {
            /* Literal byte */
            if (out_pos < dst_len)
                dst[out_pos] = (uint8_t)c;
            out_pos++;
            s->dtext[s->loc++] = (uint8_t)c;
            if (s->loc >= DICSIZ) s->loc = 0;
        } else {
            /* Copy from history */
            int match_len = c - 256 + THRESHOLD;
            int match_pos = decode_p(s);
            if (match_pos < 0) break;

            unsigned int match_off = (s->loc - (unsigned)match_pos - 1) & (DICSIZ - 1);
            for (int i = 0; i < match_len; i++) {
                uint8_t b = s->dtext[(match_off + (unsigned)i) & (DICSIZ - 1)];
                if (out_pos < dst_len)
                    dst[out_pos] = b;
                out_pos++;
                s->dtext[s->loc++] = b;
                if (s->loc >= DICSIZ) s->loc = 0;
            }
        }
    }

    free(s);
    return out_pos;
}

/* =======================================================================
 * File loading helper
 * ======================================================================= */

int sb_load_file(const char *path, uint8_t **out_data, size_t *out_size)
{
    *out_data = NULL;
    *out_size = 0;

    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("[SB] Cannot open: %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_len = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_len < 12) {
        printf("[SB] File too small: %s (%ld bytes)\n", path, file_len);
        fclose(f);
        return -1;
    }

    uint8_t *file_buf = (uint8_t *)malloc((size_t)file_len);
    if (!file_buf) {
        fclose(f);
        return -1;
    }

    if (fread(file_buf, 1, (size_t)file_len, f) != (size_t)file_len) {
        printf("[SB] Read error: %s\n", path);
        free(file_buf);
        fclose(f);
        return -1;
    }
    fclose(f);

    if (!sb_is_compressed(file_buf, (size_t)file_len)) {
        /* Not compressed - return raw data */
        printf("[SB] Raw file (not =SB= compressed): %s (%ld bytes)\n",
               path, file_len);
        *out_data = file_buf;
        *out_size = (size_t)file_len;
        return 0;
    }

    uint32_t unpacked = sb_unpacked_size(file_buf, (size_t)file_len);
    uint8_t *dst = (uint8_t *)calloc(1, unpacked);
    if (!dst) {
        free(file_buf);
        return -1;
    }

    size_t decoded = sb_decompress(file_buf, (size_t)file_len, dst, unpacked);
    free(file_buf);

    if (decoded == 0) {
        printf("[SB] Decompression failed: %s\n", path);
        free(dst);
        return -1;
    }

    printf("[SB] Decompressed: %s -> %zu bytes (expected %u)\n",
           path, decoded, unpacked);

    *out_data = dst;
    *out_size = decoded;
    return 0;
}
