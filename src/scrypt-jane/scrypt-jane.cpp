/*
 * scrypt-jane by Andrew M, https://github.com/floodyberry/scrypt-jane
 *
 * Public Domain or MIT License, whichever is easier
 *
 * Adapted to ccminer by tpruvot@github (2015)
 */

#include "api_internal.h"

#include "scrypt-jane.h"
#include "scrypt-jane-portable.h"
#include "scrypt-jane-chacha.h"

#define scrypt_maxN 30  /* (1 << (30 + 1)) = ~2 billion */
#define scrypt_r_32kb 8 /* (1 << 8) = 256 * 2 blocks in a chunk * 64 bytes = Max of 32kb in a chunk */
#define scrypt_maxr scrypt_r_32kb /* 32kb */
#define scrypt_maxp 25  /* (1 << 25) = ~33 million */

// ---------------------------- BEGIN keccak functions ------------------------------------

#define SCRYPT_HASH "Keccak-512"
#define SCRYPT_HASH_DIGEST_SIZE 64
#define SCRYPT_KECCAK_F 1600
#define SCRYPT_KECCAK_C (SCRYPT_HASH_DIGEST_SIZE * 8 * 2) /* 1024 */
#define SCRYPT_KECCAK_R (SCRYPT_KECCAK_F - SCRYPT_KECCAK_C) /* 576 */
#define SCRYPT_HASH_BLOCK_SIZE (SCRYPT_KECCAK_R / 8)

typedef uint8_t scrypt_hash_digest[SCRYPT_HASH_DIGEST_SIZE];

typedef struct scrypt_hash_state_t {
	uint64_t state[SCRYPT_KECCAK_F / 64];
	uint32_t leftover;
	uint8_t buffer[SCRYPT_HASH_BLOCK_SIZE];
} scrypt_hash_state;

static const uint64_t keccak_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

static void keccak_block(scrypt_hash_state *S, const uint8_t *in)
{
	size_t i;
	uint64_t *s = S->state, t[5], u[5], v, w;

	/* absorb input */
	for (i = 0; i < SCRYPT_HASH_BLOCK_SIZE / 8; i++, in += 8) {
		s[i] ^= U8TO64_LE(in);
	}

	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[ 1];
		s[ 1] = ROTL64(s[ 6], 44);
		s[ 6] = ROTL64(s[ 9], 20);
		s[ 9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[ 2], 62);
		s[ 2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19],  8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[ 4], 27);
		s[ 4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21],  2);
		s[21] = ROTL64(s[ 8], 55);
		s[ 8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[ 5], 36);
		s[ 5] = ROTL64(s[ 3], 28);
		s[ 3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[ 7],  6);
		s[ 7] = ROTL64(s[10],  3);
		s[10] = ROTL64(    v,  1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
		v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[i];
	}
}

static void scrypt_hash_init(scrypt_hash_state *S) {
	memset(S, 0, sizeof(*S));
}

static void scrypt_hash_update(scrypt_hash_state *S, const uint8_t *in, size_t inlen)
{
	size_t want;

	/* handle the previous data */
	if (S->leftover) {
		want = (SCRYPT_HASH_BLOCK_SIZE - S->leftover);
		want = (want < inlen) ? want : inlen;
		memcpy(S->buffer + S->leftover, in, want);
		S->leftover += (uint32_t)want;
		if (S->leftover < SCRYPT_HASH_BLOCK_SIZE) {
			return;
		}
		in += want;
		inlen -= want;
		keccak_block(S, S->buffer);
	}

	/* handle the current data */
	while (inlen >= SCRYPT_HASH_BLOCK_SIZE) {
		keccak_block(S, in);
		in += SCRYPT_HASH_BLOCK_SIZE;
		inlen -= SCRYPT_HASH_BLOCK_SIZE;
	}

	/* handle leftover data */
	S->leftover = (uint32_t)inlen;
	if (S->leftover) {
		memcpy(S->buffer, in, S->leftover);
	}
}

static void scrypt_hash_finish(scrypt_hash_state *S, uint8_t *hash)
{
	size_t i;

	S->buffer[S->leftover] = 0x01;
	memset(S->buffer + (S->leftover + 1), 0, SCRYPT_HASH_BLOCK_SIZE - (S->leftover + 1));
	S->buffer[SCRYPT_HASH_BLOCK_SIZE - 1] |= 0x80;
	keccak_block(S, S->buffer);

	for (i = 0; i < SCRYPT_HASH_DIGEST_SIZE; i += 8) {
		U64TO8_LE(&hash[i], S->state[i / 8]);
	}
}

static uint8_t scrypt_hash_finish_1(scrypt_hash_state *S)
{
	size_t i;

	S->buffer[S->leftover] = 0x01;
	memset(S->buffer + (S->leftover + 1), 0, SCRYPT_HASH_BLOCK_SIZE - (S->leftover + 1));
	S->buffer[SCRYPT_HASH_BLOCK_SIZE - 1] |= 0x80;
	keccak_block(S, S->buffer);
	return S->state[0];
}

// ---------------------------- END keccak functions ------------------------------------

// ---------------------------- BEGIN PBKDF2 functions ------------------------------------

typedef struct scrypt_hmac_state_t {
	scrypt_hash_state inner, outer;
} scrypt_hmac_state;


static void scrypt_hash(scrypt_hash_digest hash, const uint8_t *m, size_t mlen)
{
	scrypt_hash_state st;

	scrypt_hash_init(&st);
	scrypt_hash_update(&st, m, mlen);
	scrypt_hash_finish(&st, hash);
}

/* hmac */
static void scrypt_hmac_init(scrypt_hmac_state *st, const uint8_t *key, size_t keylen)
{
	uint8_t pad[SCRYPT_HASH_BLOCK_SIZE] = {0};
	size_t i;

	scrypt_hash_init(&st->inner);
	scrypt_hash_init(&st->outer);

	if (keylen <= SCRYPT_HASH_BLOCK_SIZE) {
		/* use the key directly if it's <= blocksize bytes */
		memcpy(pad, key, keylen);
	} else {
		/* if it's > blocksize bytes, hash it */
		scrypt_hash(pad, key, keylen);
	}

	/* inner = (key ^ 0x36) */
	/* h(inner || ...) */
	for (i = 0; i < SCRYPT_HASH_BLOCK_SIZE; i++)
		pad[i] ^= 0x36;
	scrypt_hash_update(&st->inner, pad, SCRYPT_HASH_BLOCK_SIZE);

	/* outer = (key ^ 0x5c) */
	/* h(outer || ...) */
	for (i = 0; i < SCRYPT_HASH_BLOCK_SIZE; i++)
		pad[i] ^= (0x5c ^ 0x36);
	scrypt_hash_update(&st->outer, pad, SCRYPT_HASH_BLOCK_SIZE);
}

static void scrypt_hmac_update(scrypt_hmac_state *st, const uint8_t *m, size_t mlen)
{
	/* h(inner || m...) */
	scrypt_hash_update(&st->inner, m, mlen);
}

static void scrypt_hmac_finish(scrypt_hmac_state *st, scrypt_hash_digest mac)
{
	/* h(inner || m) */
	scrypt_hash_digest innerhash;
	scrypt_hash_finish(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	scrypt_hash_update(&st->outer, innerhash, sizeof(innerhash));
	scrypt_hash_finish(&st->outer, mac);
}

static uint8_t scrypt_hmac_finish_1(scrypt_hmac_state *st)
{
	/* h(inner || m) */
	scrypt_hash_digest innerhash;
	scrypt_hash_finish(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	scrypt_hash_update(&st->outer, innerhash, sizeof(innerhash));
	return scrypt_hash_finish_1(&st->outer);
}

static void scrypt_pbkdf2(const uint8_t *password, size_t password_len, const uint8_t *salt, size_t salt_len, uint8_t *out, uint64_t bytes)
{
	scrypt_hmac_state hmac_pw, hmac_pw_salt, work;
	scrypt_hash_digest ti, u;
	uint8_t be[4];
	uint32_t i, blocks;

	/* bytes must be <= (0xffffffff - (SCRYPT_HASH_DIGEST_SIZE - 1)), which they will always be under scrypt */

	/* hmac(password, ...) */
	scrypt_hmac_init(&hmac_pw, password, password_len);

	/* hmac(password, salt...) */
	hmac_pw_salt = hmac_pw;
	if (salt_len && salt) {
		scrypt_hmac_update(&hmac_pw_salt, salt, salt_len);
	}

	blocks = ((uint32_t)bytes + (SCRYPT_HASH_DIGEST_SIZE - 1)) / SCRYPT_HASH_DIGEST_SIZE;
	for (i = 1; i <= blocks; i++) {
		/* U1 = hmac(password, salt || be(i)) */
		U32TO8_BE(be, i);
		work = hmac_pw_salt;
		scrypt_hmac_update(&work, be, 4);
		scrypt_hmac_finish(&work, ti);
		memcpy(u, ti, sizeof(u));

		memcpy(out, ti, (size_t) (bytes > SCRYPT_HASH_DIGEST_SIZE ? SCRYPT_HASH_DIGEST_SIZE : bytes));
		out += SCRYPT_HASH_DIGEST_SIZE;
		bytes -= SCRYPT_HASH_DIGEST_SIZE;
	}
}

static uint8_t scrypt_pbkdf2_1(const uint8_t *password, size_t password_len, const uint8_t *salt, size_t salt_len)
{
	scrypt_hmac_state hmac_pw, hmac_pw_salt, work;
	static uint8_t be[4] = { 0, 0, 0, 1 };

	/* bytes must be <= (0xffffffff - (SCRYPT_HASH_DIGEST_SIZE - 1)), which they will always be under scrypt */

	/* hmac(password, ...) */
	scrypt_hmac_init(&hmac_pw, password, password_len);

	/* hmac(password, salt...) */
	hmac_pw_salt = hmac_pw;
	if (salt_len && salt) {
		scrypt_hmac_update(&hmac_pw_salt, salt, salt_len);
	}

	/* U1 = hmac(password, salt || be(1)) */
	work = hmac_pw_salt;
	scrypt_hmac_update(&work, be, 4);
	return scrypt_hmac_finish_1(&work);
}

// ---------------------------- END PBKDF2 functions ------------------------------------

static void scrypt_fatal_error_default(const char *msg) {
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

static scrypt_fatal_errorfn scrypt_fatal_error = scrypt_fatal_error_default;

void scrypt_set_fatal_error_default(scrypt_fatal_errorfn fn) {
	scrypt_fatal_error = fn;
}

typedef struct scrypt_aligned_alloc_t {
	uint8_t *mem, *ptr;
} scrypt_aligned_alloc;

#if defined(SCRYPT_TEST_SPEED)
static uint8_t *mem_base = (uint8_t *)0;
static size_t mem_bump = 0;

/* allocations are assumed to be multiples of 64 bytes and total allocations not to exceed ~1.01gb */
static scrypt_aligned_alloc scrypt_alloc(uint64_t size)
{
	scrypt_aligned_alloc aa;
	if (!mem_base) {
		mem_base = (uint8_t *)malloc((1024 * 1024 * 1024) + (1024 * 1024) + (SCRYPT_BLOCK_BYTES - 1));
		if (!mem_base)
			scrypt_fatal_error("scrypt: out of memory");
		mem_base = (uint8_t *)(((size_t)mem_base + (SCRYPT_BLOCK_BYTES - 1)) & ~(SCRYPT_BLOCK_BYTES - 1));
	}
	aa.mem = mem_base + mem_bump;
	aa.ptr = aa.mem;
	mem_bump += (size_t)size;
	return aa;
}

static void scrypt_free(scrypt_aligned_alloc *aa)
{
	mem_bump = 0;
}
#else
static scrypt_aligned_alloc scrypt_alloc(uint64_t size)
{
	static const size_t max_alloc = (size_t)-1;
	scrypt_aligned_alloc aa;
	size += (SCRYPT_BLOCK_BYTES - 1);
	if (size > max_alloc) {
		scrypt_fatal_error("scrypt: not enough address space on this CPU to allocate required memory");
	}
	aa.mem = (uint8_t *)malloc((size_t)size);
	aa.ptr = (uint8_t *)(((size_t)aa.mem + (SCRYPT_BLOCK_BYTES - 1)) & ~(SCRYPT_BLOCK_BYTES - 1));
	if (!aa.mem) {
		scrypt_fatal_error("scrypt: out of memory");
	}
	return aa;
}

static void scrypt_free(scrypt_aligned_alloc *aa)
{
	free(aa->mem);
}
#endif

static uint8_t scrypt_jane_hash_1_1(const uchar *password, size_t password_len, const uchar*salt, size_t salt_len, uint32_t N, uint8_t *X, uint8_t *Y, uint8_t *V)
{
	uint32_t chunk_bytes, i;
	const uint32_t p = 1;

#if !defined(SCRYPT_CHOOSE_COMPILETIME)
	scrypt_ROMixfn scrypt_ROMix = scrypt_getROMix();
#endif

	chunk_bytes = SCRYPT_BLOCK_BYTES * 1 * 2;

	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2(password, password_len, salt, salt_len, X, chunk_bytes * p);

	/* 2: X = ROMix(X) */
	for (i = 0; i < p; i++) {
		scrypt_ROMix_1((scrypt_mix_word_t *)(X + (chunk_bytes * i)), (scrypt_mix_word_t *)Y, (scrypt_mix_word_t *)V, N, 1);
	}

	/* 3: Out = PBKDF2(password, X) */
	return scrypt_pbkdf2_1(password, password_len, X, chunk_bytes * p);
}

static void cpu_shutdown(struct cgpu_info *cgpu);

typedef struct {
	uint64_t chunk_bytes;
	scrypt_aligned_alloc YX, V;
	uint8_t *X, *Y;
} _cpuState;

static int cpu_detect(struct cgpu_info *gpus, int *active)
{
	struct cgpu_info *cgpu = &gpus[*active];

	cgpu->drv = &cpu_drv;
	cgpu->driver_id = 0;

	*active += 1;

	return 0;
}

static void reinit_cpu_device(struct cgpu_info *gpu)
{
}

static _cpuState * initCpu(struct cgpu_info *cgpu, unsigned N, uint32_t r, uint32_t p)
{
	_cpuState *cpuState = (_cpuState *)calloc(1, sizeof(_cpuState));
	if (cpuState) {
		cpuState->chunk_bytes = 2ULL * SCRYPT_BLOCK_BYTES * r;
		cpuState->V = scrypt_alloc(N * cpuState->chunk_bytes);
		if (!cpuState->V.ptr) {
			free(cpuState);
			return NULL;
		}
		cpuState->YX = scrypt_alloc((p + 1) * cpuState->chunk_bytes);
		if (!cpuState->YX.ptr) {
			free(cpuState);
			scrypt_free(&cpuState->V);
			return NULL;
		}
		cpuState->Y = cpuState->YX.ptr;
		cpuState->X = cpuState->Y + cpuState->chunk_bytes;
	}

	return cpuState;
}

static bool cpu_prepare(struct cgpu_info *cgpu, unsigned N, uint32_t r, uint32_t p)
{
	if (N != cgpu->N || r != cgpu->r || p != cgpu->p) {
		if (cgpu->device_data) {
			cpu_shutdown(cgpu);
		}
		cgpu->device_data = initCpu(cgpu, N, r, p);
		if (!cgpu->device_data) {
			applog(LOG_ERR, "Failed to init CPU, disabling device %d", cgpu->id);
			cgpu->deven = DEV_DISABLED;
			cgpu->status = LIFE_NOSTART;
			return false;
		}
		cgpu->N = N;
		cgpu->r = r;
		cgpu->p = p;
		applog(LOG_INFO, "initCpu() finished.");
	}
	return true;
}

static bool cpu_init(struct cgpu_info *cgpu)
{
	cgpu->status = LIFE_WELL;
	return true;
}

static int64_t cpu_scrypt_positions(struct cgpu_info *cgpu, uint8_t *pdata, uint64_t start_position, uint64_t end_position, uint8_t hash_len_bits, uint32_t options, uint8_t *output, uint32_t N, uint32_t r, uint32_t p, struct timeval *tv_start, struct timeval *tv_end)
{
	cgpu->busy = 1;
	if (cpu_prepare(cgpu, N, r, p))
	{
		_cpuState *cpuState = (_cpuState *)cgpu->device_data;
		uint64_t n = start_position;
		uint8_t label_mask = (1 << hash_len_bits) - 1;
		uint32_t available = 8;
		uint8_t label;

		gettimeofday(tv_start, NULL);

		*output = 0;

		do {
			((uint64_t*)pdata)[4] = n;
			if (1 == r && 1 == p) {
				label = scrypt_jane_hash_1_1((uchar*)pdata, 72, NULL, 0, (uint32_t)N, cpuState->X, cpuState->Y, cpuState->V.ptr);
			}
			else {
				uint32_t i;

				/* 1: X = PBKDF2(password, salt) */
				scrypt_pbkdf2((uchar*)pdata, 72, NULL, 0, cpuState->X, cpuState->chunk_bytes * p);

				/* 2: X = ROMix(X) */
				for (i = 0; i < p; i++) {
					scrypt_ROMix_1((scrypt_mix_word_t *)(cpuState->X + (cpuState->chunk_bytes * i)), (scrypt_mix_word_t *)cpuState->Y, (scrypt_mix_word_t *)cpuState->V.ptr, N, r);
				}

				/* 3: Out = PBKDF2(password, X) */
				label = scrypt_pbkdf2_1((uchar*)pdata, 72, cpuState->X, cpuState->chunk_bytes * p);
			}
			if (8 == hash_len_bits) {
				*output++ = label;
			}
			else {
				label &= label_mask;
				if (available >= hash_len_bits) {
					label <<= 8 - available;
					*output |= label;
					available -= hash_len_bits;
					if (0 == available) {
						available = 8;
						output++;
					}
				}
				else {
					*output++ |= label << (8 - available);
					*output = label >> available;
					available = 8 - (hash_len_bits - available);
				}
			}
			n++;
		} while (n <= end_position && !abort_flag);

		gettimeofday(tv_end, NULL);
	}

	cgpu->busy = 0;
	return 0;
}

static void cpu_shutdown(struct cgpu_info *cgpu)
{
	if (cgpu->device_data) {
		_cpuState *cpuState = (_cpuState *)cgpu->device_data;

		if (!cpuState->V.ptr) {
			scrypt_free(&cpuState->V);
		}

		if (!cpuState->YX.ptr) {
			scrypt_free(&cpuState->YX);
		}

		free(cgpu->device_data);
		cgpu->device_data = NULL;
	}
}

struct device_drv cpu_drv = {
	DRIVER_CPU,
	"cpu",
	"CPU",
	cpu_detect,
	reinit_cpu_device,
	cpu_init,
	cpu_scrypt_positions,
	cpu_shutdown
};

