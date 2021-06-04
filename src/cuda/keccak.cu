//
//  =============== KECCAK part on nVidia GPU ======================
//
// The keccak512 (SHA-3) is used in the PBKDF2 for scrypt-jane coins
// in place of the SHA2 based PBKDF2 used in scrypt coins.
//
// NOTE: compile this .cu module for compute_20,sm_20 with --maxrregcount=64
//

#include <map>

#include "api_internal.h"
#include "cuda_helper.h"

#include "keccak.h"
#include "salsa_kernel.h"

// define some error checking macros
#define DELIMITER '/'
#define __FILENAME__ ( strrchr(__FILE__, DELIMITER) != NULL ? strrchr(__FILE__, DELIMITER)+1 : __FILE__ )

#undef checkCudaErrors
#define checkCudaErrors(gpuId, x) \
{ \
	cudaGetLastError(); \
	x; \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess && !g_spacemesh_api_abort_flag) \
		applog(LOG_ERR, "GPU #%d: cudaError %d (%s) (%s line %d)\n", gpuId, err, cudaGetErrorString(err), __FILENAME__, __LINE__); \
}

// from salsa_kernel.cu
extern std::map<int, uint32_t *> context_idata[2];
extern std::map<int, uint32_t *> context_odata[2];
extern std::map<int, cudaStream_t> context_streams[2];
extern std::map<int, uint32_t *> context_hash[2];

#ifndef ROTL64
#define ROTL64(a,b) (((a) << (b)) | ((a) >> (64 - b)))
#endif

// CB
#define U32TO64_LE(p) \
	(((uint64_t)(*p)) | (((uint64_t)(*(p + 1))) << 32))

#define U64TO32_LE(p, v) \
	*p = (uint32_t)((v)); *(p+1) = (uint32_t)((v) >> 32);

static __device__ void mycpy64(uint32_t *d, const uint32_t *s)
{
#pragma unroll 16
	for (int k = 0; k < 16; ++k) {
		d[k] = s[k];
	}
}

// ---------------------------- BEGIN keccak functions ------------------------------------

#define KECCAK_HASH "Keccak-512"
#define SCRYPT_HASH_DIGEST_SIZE 64
#define SCRYPT_KECCAK_F 1600
#define SCRYPT_KECCAK_C (SCRYPT_HASH_DIGEST_SIZE * 8 * 2) /* 1024 */
#define SCRYPT_KECCAK_R (SCRYPT_KECCAK_F - SCRYPT_KECCAK_C) /* 576 */
#define SCRYPT_HASH_BLOCK_SIZE (SCRYPT_KECCAK_R / 8)

typedef struct keccak_hash_state_t {
	uint64_t state[25];					// 25*2
	uint32_t buffer[72 / 4];			// 72
} keccak_hash_state;

__device__ void statecopy0(keccak_hash_state *d, keccak_hash_state *s)
{
#pragma unroll 25
	for (int i = 0; i < 25; ++i) {
		d->state[i] = s->state[i];
	}
}

static const uint64_t host_keccak_round_constants[24] = {
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

__constant__ uint64_t c_keccak_round_constants[24];
__constant__ uint32_t c_data[32];

#define U8TO32_LE(p)                                            \
	(((uint32_t)((p)[0])      ) | ((uint32_t)((p)[1]) <<  8) |  \
	 ((uint32_t)((p)[2]) << 16) | ((uint32_t)((p)[3]) << 24))

#define U32TO8_LE(p, v)                                           \
	(p)[0] = (uint8_t)((v)      ); (p)[1] = (uint8_t)((v) >>  8); \
	(p)[2] = (uint8_t)((v) >> 16); (p)[3] = (uint8_t)((v) >> 24);

#define U8TO64_LE(p)                                                  \
	(((uint64_t)U8TO32_LE(p)) | ((uint64_t)U8TO32_LE((p) + 4) << 32))

#define U64TO8_LE(p, v)                        \
	U32TO8_LE((p),     (uint32_t)((v)      )); \
	U32TO8_LE((p) + 4, (uint32_t)((v) >> 32));

__device__
void keccak_block(keccak_hash_state *S, const uint32_t *in)
{
	uint64_t *s = S->state, t[5], u[5], v, w;

	/* absorb input */
#pragma unroll 9
	for (int i = 0; i < 72 / 8; i++, in += 2) {
		s[i] ^= U32TO64_LE(in);
	}

	for (int i = 0; i < 24; i++) {
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
		v = s[1];
		s[1] = ROTL64(s[6], 44);
		s[6] = ROTL64(s[9], 20);
		s[9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[2], 62);
		s[2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19], 8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[4], 27);
		s[4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21], 2);
		s[21] = ROTL64(s[8], 55);
		s[8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[5], 36);
		s[5] = ROTL64(s[3], 28);
		s[3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[7], 6);
		s[7] = ROTL64(s[10], 3);
		s[10] = ROTL64(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= c_keccak_round_constants[i];
	}
}

__device__
void keccak_hash_init(keccak_hash_state *S)
{
#pragma unroll 25
	for (int i = 0; i < 25; ++i) {
		S->state[i] = 0ULL;
	}
}

// assuming there is no leftover data and exactly 72 bytes are incoming
// we can directly call into the block hashing function
__device__ void keccak_hash_update72(keccak_hash_state *S, const uint32_t *in)
{
	keccak_block(S, in);
}

__device__ void keccak_hash_update4(keccak_hash_state *S, const uint32_t *in)
{
	*S->buffer = *in;
}

__device__ void keccak_hash_update64(keccak_hash_state *S, const uint32_t *in)
{
	mycpy64(S->buffer, in);
}

__device__
void keccak_hash_finish4(keccak_hash_state *S, uint32_t *hash)
{
	S->buffer[4 / 4] = 0x01;
#pragma unroll
	for (int i = 4 / 4 + 1; i < 72 / 4; ++i) {
		S->buffer[i] = 0;
	}
	S->buffer[72 / 4 - 1] |= 0x80000000U;
	keccak_block(S, (const uint32_t*)S->buffer);

#pragma unroll 8
	for (int i = 0; i < 64; i += 8) {
		U64TO32_LE((&hash[i / 4]), S->state[i / 8]);
	}
}

__device__
void keccak_hash_finish64(keccak_hash_state *S, uint32_t *hash)
{
	S->buffer[64 / 4] = 0x01;
#pragma unroll
	for (int i = 64 / 4 + 1; i < 72 / 4; ++i) {
		S->buffer[i] = 0;
	}
	S->buffer[72 / 4 - 1] |= 0x80000000U;
	keccak_block(S, (const uint32_t*)S->buffer);

#pragma unroll 8
	for (int i = 0; i < 64; i += 8) {
		U64TO32_LE((&hash[i / 4]), S->state[i / 8]);
	}
}

__device__
uint8_t keccak_hash_1_finish64(keccak_hash_state *S)
{
	S->buffer[64 / 4] = 0x01;
#pragma unroll
	for (int i = 64 / 4 + 1; i < 72 / 4; ++i) {
		S->buffer[i] = 0;
	}
	S->buffer[72 / 4 - 1] |= 0x80000000U;
	keccak_block(S, (const uint32_t*)S->buffer);

	return S->state[0];
}

__device__
void keccak_hash_finish(keccak_hash_state *S, uint32_t buffered, uint32_t *hash)
{
	uint32_t i = buffered / 4;
	S->buffer[i] = 0x01;

	for (i = i + 1; i < 72 / 4; ++i) {
		S->buffer[i] = 0;
	}

	S->buffer[72 / 4 - 1] |= 0x80000000U;
	keccak_block(S, (const uint32_t*)S->buffer);

#pragma unroll 8
	for (int i = 0; i < 64; i += 8) {
		U64TO32_LE((&hash[i / 4]), S->state[i / 8]);
	}
}

// ---------------------------- END keccak functions ------------------------------------

// ---------------------------- BEGIN PBKDF2 functions ------------------------------------

typedef struct pbkdf2_hmac_state_t {
	keccak_hash_state inner, outer;
} pbkdf2_hmac_state;

/* hmac */

__device__
void pbkdf2_hmac_init72(pbkdf2_hmac_state *st, const uint32_t *key)
{
	uint32_t pad[72 / 4] = { 0 };

	keccak_hash_init(&st->inner);
	keccak_hash_init(&st->outer);

#pragma unroll 18
	for (int i = 0; i < 72 / 4; i++) {
		pad[i] = key[i];
	}

	/* inner = (key ^ 0x36) */
	/* h(inner || ...) */
#pragma unroll 18
	for (int i = 0; i < 72 / 4; i++) {
		pad[i] ^= 0x36363636U;
	}
	keccak_hash_update72(&st->inner, pad);

	/* outer = (key ^ 0x5c) */
	/* h(outer || ...) */
#pragma unroll 18
	for (int i = 0; i < 72 / 4; i++) {
		pad[i] ^= 0x6a6a6a6aU;
	}
	keccak_hash_update72(&st->outer, pad);
}

__device__ void pbkdf2_hmac_update4(pbkdf2_hmac_state *st, const uint32_t *m)
{
	/* h(inner || m...) */
	keccak_hash_update4(&st->inner, m);
}

__device__
uint32_t pbkdf2_hmac_update(pbkdf2_hmac_state *st, const uint32_t *m, uint32_t length)
{
	/* h(inner || m...) */
	while (length >= 72) {
		keccak_hash_update72(&st->inner, m);
		length -= 72;
		m += 74 / 4;
	}

	if (length > 0)  ((uint64_t*)st->inner.buffer)[0] = ((const uint64_t*)m)[0];
	if (length > 8)  ((uint64_t*)st->inner.buffer)[1] = ((const uint64_t*)m)[1];
	if (length > 16) ((uint64_t*)st->inner.buffer)[2] = ((const uint64_t*)m)[2];
	if (length > 24) ((uint64_t*)st->inner.buffer)[3] = ((const uint64_t*)m)[3];
	if (length > 32) ((uint64_t*)st->inner.buffer)[4] = ((const uint64_t*)m)[4];
	if (length > 40) ((uint64_t*)st->inner.buffer)[5] = ((const uint64_t*)m)[5];
	if (length > 48) ((uint64_t*)st->inner.buffer)[6] = ((const uint64_t*)m)[6];
	if (length > 56) ((uint64_t*)st->inner.buffer)[7] = ((const uint64_t*)m)[7];

	return length;
}

__device__
uint32_t pbkdf2_hmac_buffer_update4(pbkdf2_hmac_state *st, const uint32_t m, uint32_t buffered)
{
	st->inner.buffer[buffered / 4] = m;
	return buffered + 4;
}

__device__ void pbkdf2_hmac_finish4(pbkdf2_hmac_state *st, uint32_t *mac)
{
	/* h(inner || m) */
	uint32_t innerhash[16];
	keccak_hash_finish4(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	keccak_hash_update64(&st->outer, innerhash);
	keccak_hash_finish64(&st->outer, mac);
}

__device__ uint8_t pbkdf2_hmac_finish(pbkdf2_hmac_state *st, uint32_t buffered)
{
	/* h(inner || m) */
	uint32_t innerhash[16];
	keccak_hash_finish(&st->inner, buffered, innerhash);

	/* h(outer || h(inner || m)) */
	keccak_hash_update64(&st->outer, innerhash);
	return keccak_hash_1_finish64(&st->outer);
}

__device__ void pbkdf2_statecopy0(pbkdf2_hmac_state *d, pbkdf2_hmac_state *s)
{
	statecopy0(&d->inner, &s->inner);
	statecopy0(&d->outer, &s->outer);
}

// ---------------------------- END PBKDF2 functions ------------------------------------

#define U32TO8_BE(p, v)                                           \
	(p)[0] = (uint8_t)((v) >> 24); (p)[1] = (uint8_t)((v) >> 16); \
	(p)[2] = (uint8_t)((v) >>  8); (p)[3] = (uint8_t)((v)      );

__global__ __launch_bounds__(128)
void cuda_pre_keccak512(uint32_t *g_idata, uint64_t nonce, uint32_t r)
{
	uint32_t i, blocks;
	uint32_t data[20];

	const uint32_t thread = (blockIdx.x * blockDim.x) + threadIdx.x;
	nonce   += thread;
	g_idata += thread * 32 * r;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;
	uint32_t bytes = r * 128;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	pbkdf2_hmac_state work;
	uint32_t ti[16];

	blocks = ((uint32_t)bytes + (SCRYPT_HASH_DIGEST_SIZE - 1)) / SCRYPT_HASH_DIGEST_SIZE;
	for (i = 1; i <= blocks; i++) {
		/* U1 = hmac(password, salt || be(i)) */
		uint32_t be = cuda_swab32(i);
		pbkdf2_statecopy0(&work, &hmac_pw);
		pbkdf2_hmac_update4(&work, &be);
		pbkdf2_hmac_finish4(&work, ti);
		mycpy64(g_idata, ti);

		g_idata += SCRYPT_HASH_DIGEST_SIZE / sizeof(uint32_t);
		bytes -= SCRYPT_HASH_DIGEST_SIZE;
	}

	union {
		uint64_t u64;
		uint32_t u32[2];
	} d;
}

#define CHECK_POW() \
	int cmp = 0; \
	union { \
		uint64_t u64; \
		uint32_t u32[2]; \
	} d; \
 \
	d.u64 = hmac_pw.outer.state[0]; \
	d.u32[0] = cuda_swab32(d.u32[0]); \
	d.u32[1] = cuda_swab32(d.u32[1]); \
	if (c_data[20] != d.u32[0]) { \
		cmp = d.u32[0] < c_data[20] ? -1 : 1; \
	} \
	if (cmp == 0 && c_data[21] != d.u32[1]) { \
		cmp = d.u32[1] < c_data[21] ? -1 : 1; \
	} \
 \
	d.u64 = hmac_pw.outer.state[1]; \
	d.u32[0] = cuda_swab32(d.u32[0]); \
	d.u32[1] = cuda_swab32(d.u32[1]); \
	if (cmp == 0 && c_data[22] != d.u32[0]) { \
		cmp = d.u32[0] < c_data[22] ? -1 : 1; \
	} \
	if (cmp == 0 && c_data[23] != d.u32[1]) { \
		cmp = d.u32[1] < c_data[23] ? -1 : 1; \
	} \
 \
	d.u64 = hmac_pw.outer.state[2]; \
	d.u32[0] = cuda_swab32(d.u32[0]); \
	d.u32[1] = cuda_swab32(d.u32[1]); \
	if (cmp == 0 && c_data[24] != d.u32[0]) { \
		cmp = d.u32[0] < c_data[24] ? -1 : 1; \
	} \
	if (cmp == 0 && c_data[25] != d.u32[1]) { \
		cmp = d.u32[1] < c_data[25] ? -1 : 1; \
	} \
 \
	d.u64 = hmac_pw.outer.state[3]; \
	d.u32[0] = cuda_swab32(d.u32[0]); \
	d.u32[1] = cuda_swab32(d.u32[1]); \
	if (cmp == 0 && c_data[26] != d.u32[0]) { \
		cmp = d.u32[0] < c_data[26] ? -1 : 1; \
	} \
	if (cmp == 0 && c_data[27] != d.u32[1]) { \
		cmp = d.u32[1] < c_data[27] ? -1 : 1; \
	} \
	if (cmp < 0) { \
		solutions[0] = nonce; \
	}

__global__ __launch_bounds__(128)
void cuda_pre_keccak512_1_1(uint32_t *g_idata, uint64_t nonce)
{
	uint32_t i, blocks;
	uint32_t data[20];

	const uint32_t thread = (blockIdx.x * blockDim.x) + threadIdx.x;
	nonce += thread;
	g_idata += thread * 32;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;
	uint32_t bytes = 128;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	pbkdf2_hmac_state work;

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000; //  cuda_swab32(1);
	pbkdf2_statecopy0(&work, &hmac_pw);
	pbkdf2_hmac_update4(&work, &be);
	pbkdf2_hmac_finish4(&work, g_idata);

	g_idata += SCRYPT_HASH_DIGEST_SIZE / sizeof(uint32_t);

	be = 0x02000000; //  cuda_swab32(2);
	pbkdf2_statecopy0(&work, &hmac_pw);
	pbkdf2_hmac_update4(&work, &be);
	pbkdf2_hmac_finish4(&work, g_idata);
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_8(uint32_t *g_odata, uint8_t *labels, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce   += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	labels[thread] = pbkdf2_hmac_finish(&hmac_pw, buffered);

	CHECK_POW()
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_7(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x7f;

	CHECK_POW()

	out += thread * 28 / 32;

	uint32_t labels1, labels2, labels3, labels4, labels5, labels6, labels7;

	labels1  = __shfl_sync(0xFFFFFFFF, label, 0, 32);
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 1, 32) << 7;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 2, 32) << 14;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 3, 32) << 21;
	labels2  = __shfl_sync(0xFFFFFFFF, label, 4, 32);
	labels1 |= labels2 << 28;

	labels2 >>= 4;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 5, 32) << 3;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 6, 32) << 10;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 7, 32) << 17;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 8, 32) << 24;
	labels3  = __shfl_sync(0xFFFFFFFF, label, 9, 32);
	labels2 |= labels3 << 31;

	labels3 >>= 1;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 10, 32) << 6;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 11, 32) << 13;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 12, 32) << 20;
	labels4  = __shfl_sync(0xFFFFFFFF, label, 13, 32);
	labels3 |= labels4 << 27;

	labels4 >>= 5;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 14, 32) << 2;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 15, 32) << 9;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 16, 32) << 16;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 17, 32) << 23;
	labels5  = __shfl_sync(0xFFFFFFFF, label, 18, 32);
	labels4 |= labels5 << 30;

	labels5 >>= 2;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 19, 32) << 5;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 20, 32) << 12;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 21, 32) << 19;
	labels6  = __shfl_sync(0xFFFFFFFF, label, 22, 32);
	labels5 |= labels6 << 26;

	labels6 >>= 6;
	labels6 |= __shfl_sync(0xFFFFFFFF, label, 23, 32) << 1;
	labels6 |= __shfl_sync(0xFFFFFFFF, label, 24, 32) << 8;
	labels6 |= __shfl_sync(0xFFFFFFFF, label, 25, 32) << 15;
	labels6 |= __shfl_sync(0xFFFFFFFF, label, 26, 32) << 22;
	labels7  = __shfl_sync(0xFFFFFFFF, label, 27, 32);
	labels6 |= labels7 << 29;

	labels7 >>= 3;
	labels7 |= __shfl_sync(0xFFFFFFFF, label, 28, 32) << 4;
	labels7 |= __shfl_sync(0xFFFFFFFF, label, 29, 32) << 11;
	labels7 |= __shfl_sync(0xFFFFFFFF, label, 30, 32) << 18;
	labels7 |= __shfl_sync(0xFFFFFFFF, label, 31, 32) << 25;

	if (0 == threadIdx.x % 32) {
		((uint32_t*)out)[0] = labels1;
		((uint32_t*)out)[1] = labels2;
		((uint32_t*)out)[2] = labels3;
		((uint32_t*)out)[3] = labels4;
		((uint32_t*)out)[4] = labels5;
		((uint32_t*)out)[5] = labels6;
		((uint32_t*)out)[6] = labels7;
	}
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_6(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x3f;

	CHECK_POW()

	out += thread * 24 / 32;

	uint2 labels1, labels2, labels3;

	labels1.x  = __shfl_sync(0xFFFFFFFF, label, 0, 32);
	labels1.x |= __shfl_sync(0xFFFFFFFF, label, 1, 32) << 6;
	labels1.x |= __shfl_sync(0xFFFFFFFF, label, 2, 32) << 12;
	labels1.x |= __shfl_sync(0xFFFFFFFF, label, 3, 32) << 18;
	labels1.x |= __shfl_sync(0xFFFFFFFF, label, 4, 32) << 24;
	labels1.y  = __shfl_sync(0xFFFFFFFF, label, 5, 32);
	labels1.x |= labels1.y << 30;

	labels1.y >>= 2;
	labels1.y |= __shfl_sync(0xFFFFFFFF, label, 6, 32) << 4;
	labels1.y |= __shfl_sync(0xFFFFFFFF, label, 7, 32) << 10;
	labels1.y |= __shfl_sync(0xFFFFFFFF, label, 8, 32) << 16;
	labels1.y |= __shfl_sync(0xFFFFFFFF, label, 9, 32) << 22;
	labels2.x  = __shfl_sync(0xFFFFFFFF, label, 10, 32);
	labels1.y |= labels2.x << 28;

	labels2.x >>= 4;
	labels2.x |= __shfl_sync(0xFFFFFFFF, label, 11, 32) << 2;
	labels2.x |= __shfl_sync(0xFFFFFFFF, label, 12, 32) << 8;
	labels2.x |= __shfl_sync(0xFFFFFFFF, label, 13, 32) << 14;
	labels2.x |= __shfl_sync(0xFFFFFFFF, label, 14, 32) << 20;
	labels2.x |= __shfl_sync(0xFFFFFFFF, label, 15, 32) << 26;

	labels2.y  = __shfl_sync(0xFFFFFFFF, label, 16, 32);
	labels2.y |= __shfl_sync(0xFFFFFFFF, label, 17, 32) << 6;
	labels2.y |= __shfl_sync(0xFFFFFFFF, label, 18, 32) << 12;
	labels2.y |= __shfl_sync(0xFFFFFFFF, label, 19, 32) << 18;
	labels2.y |= __shfl_sync(0xFFFFFFFF, label, 20, 32) << 24;
	labels3.x  = __shfl_sync(0xFFFFFFFF, label, 21, 32);
	labels2.y |= labels3.x << 30;

	labels3.x >>= 2;
	labels3.x |= __shfl_sync(0xFFFFFFFF, label, 22, 32) << 4;
	labels3.x |= __shfl_sync(0xFFFFFFFF, label, 23, 32) << 10;
	labels3.x |= __shfl_sync(0xFFFFFFFF, label, 24, 32) << 16;
	labels3.x |= __shfl_sync(0xFFFFFFFF, label, 25, 32) << 22;
	labels3.y  = __shfl_sync(0xFFFFFFFF, label, 26, 32);
	labels3.x |= labels3.y << 28;

	labels3.y >>= 4;
	labels3.y |= __shfl_sync(0xFFFFFFFF, label, 27, 32) << 2;
	labels3.y |= __shfl_sync(0xFFFFFFFF, label, 28, 32) << 8;
	labels3.y |= __shfl_sync(0xFFFFFFFF, label, 29, 32) << 14;
	labels3.y |= __shfl_sync(0xFFFFFFFF, label, 30, 32) << 20;
	labels3.y |= __shfl_sync(0xFFFFFFFF, label, 31, 32) << 26;

	if (0 == threadIdx.x % 32) {
		((uint2*)out)[0] = labels1;
		((uint2*)out)[1] = labels2;
		((uint2*)out)[2] = labels3;
	}
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_5(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x1f;

	CHECK_POW()

	out += thread * 20 / 32;

	uint32_t labels1, labels2, labels3, labels4, labels5;

	labels1  = __shfl_sync(0xFFFFFFFF, label, 0, 32);
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 1, 32) << 5;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 2, 32) << 10;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 3, 32) << 15;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 4, 32) << 20;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 5, 32) << 25;
	labels2  = __shfl_sync(0xFFFFFFFF, label, 6, 32);
	labels1 |= labels2 << 30;

	labels2 >>= 2;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 7, 32) << 3;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 8, 32) << 8;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 9, 32) << 13;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 10, 32) << 18;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 11, 32) << 23;
	labels3  = __shfl_sync(0xFFFFFFFF, label, 12, 32);
	labels2 |=  labels3 << 28;

	labels3 >>= 4;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 13, 32) << 1;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 14, 32) << 6;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 15, 32) << 11;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 16, 32) << 16;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 17, 32) << 21;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 18, 32) << 26;
	labels4  = __shfl_sync(0xFFFFFFFF, label, 19, 32);
	labels3 |=  labels4 << 31;

	labels4 >>= 1;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 20, 32) << 4;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 21, 32) << 9;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 22, 32) << 14;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 23, 32) << 19;
	labels4 |= __shfl_sync(0xFFFFFFFF, label, 24, 32) << 24;
	labels5  = __shfl_sync(0xFFFFFFFF, label, 25, 32);
	labels4 |= labels5 << 29;

	labels5 >>= 3;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 26, 32) << 2;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 27, 32) << 7;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 28, 32) << 12;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 29, 32) << 17;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 30, 32) << 22;
	labels5 |= __shfl_sync(0xFFFFFFFF, label, 31, 32) << 27;

	if (0 == threadIdx.x % 32) {
		((uint32_t*)out)[0] = labels1;
		((uint32_t*)out)[1] = labels2;
		((uint32_t*)out)[2] = labels3;
		((uint32_t*)out)[3] = labels4;
		((uint32_t*)out)[4] = labels5;
	}
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_4(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x0f;

	CHECK_POW()

	out += thread * 16 / 32;

	uint4 labels;

	labels.x  = __shfl_sync(0xFFFFFFFF, label, 0, 32);
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 1, 32) << 4;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 2, 32) << 8;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 3, 32) << 12;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 4, 32) << 16;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 5, 32) << 20;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 6, 32) << 24;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 7, 32) << 28;

	labels.y  = __shfl_sync(0xFFFFFFFF, label,  8, 32);
	labels.y |= __shfl_sync(0xFFFFFFFF, label,  9, 32) << 4;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 10, 32) << 8;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 11, 32) << 12;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 12, 32) << 16;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 13, 32) << 20;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 14, 32) << 24;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 15, 32) << 28;

	labels.z  = __shfl_sync(0xFFFFFFFF, label, 16, 32);
	labels.z |= __shfl_sync(0xFFFFFFFF, label, 17, 32) << 4;
	labels.z |= __shfl_sync(0xFFFFFFFF, label, 18, 32) << 8;
	labels.z |= __shfl_sync(0xFFFFFFFF, label, 19, 32) << 12;
	labels.z |= __shfl_sync(0xFFFFFFFF, label, 20, 32) << 16;
	labels.z |= __shfl_sync(0xFFFFFFFF, label, 21, 32) << 20;
	labels.z |= __shfl_sync(0xFFFFFFFF, label, 22, 32) << 24;
	labels.z |= __shfl_sync(0xFFFFFFFF, label, 23, 32) << 28;

	labels.w  = __shfl_sync(0xFFFFFFFF, label, 24, 32);
	labels.w |= __shfl_sync(0xFFFFFFFF, label, 25, 32) << 4;
	labels.w |= __shfl_sync(0xFFFFFFFF, label, 26, 32) << 8;
	labels.w |= __shfl_sync(0xFFFFFFFF, label, 27, 32) << 12;
	labels.w |= __shfl_sync(0xFFFFFFFF, label, 28, 32) << 16;
	labels.w |= __shfl_sync(0xFFFFFFFF, label, 29, 32) << 20;
	labels.w |= __shfl_sync(0xFFFFFFFF, label, 30, 32) << 24;
	labels.w |= __shfl_sync(0xFFFFFFFF, label, 31, 32) << 28;

	if (0 == threadIdx.x % 32) {
			*(uint4*)out = labels;
	}
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_3(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x07;

	CHECK_POW()

	out += thread * 12 / 32;

	uint32_t labels1, labels2, labels3;

	labels1  = __shfl_sync(0xFFFFFFFF, label, 0, 32);
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 1, 32) << 3;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 2, 32) << 6;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 3, 32) << 9;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 4, 32) << 12;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 5, 32) << 15;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 6, 32) << 18;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 7, 32) << 21;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 8, 32) << 24;
	labels1 |= __shfl_sync(0xFFFFFFFF, label, 9, 32) << 27;
	labels2  = __shfl_sync(0xFFFFFFFF, label, 10, 32);
	labels1 |= labels2 << 30;

	labels2 >>= 2;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 11, 32) << 1;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 12, 32) << 4;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 13, 32) << 7;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 14, 32) << 10;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 15, 32) << 13;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 16, 32) << 16;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 17, 32) << 19;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 18, 32) << 22;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 19, 32) << 25;
	labels2 |= __shfl_sync(0xFFFFFFFF, label, 20, 32) << 28;
	labels3  = __shfl_sync(0xFFFFFFFF, label, 21, 32);
	labels2 |=  labels3 << 31;

	labels3  >>= 1;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 22, 32) << 2;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 23, 32) << 5;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 24, 32) << 8;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 25, 32) << 11;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 26, 32) << 14;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 27, 32) << 17;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 28, 32) << 20;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 29, 32) << 23;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 30, 32) << 26;
	labels3 |= __shfl_sync(0xFFFFFFFF, label, 31, 32) << 29;

	if (0 == threadIdx.x % 32) {
		((uint32_t*)out)[0] = labels1;
		((uint32_t*)out)[1] = labels2;
		((uint32_t*)out)[2] = labels3;
	}
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_2(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x03;

	CHECK_POW()

	out += thread * 8 / 32;

	uint2 labels;

	labels.x  = __shfl_sync(0xFFFFFFFF, label,  0, 32);
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  1, 32) << 2;
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  2, 32) << 4;
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  3, 32) << 6;
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  4, 32) << 8;
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  5, 32) << 10;
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  6, 32) << 12;
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  7, 32) << 14;
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  8, 32) << 16;
	labels.x |= __shfl_sync(0xFFFFFFFF, label,  9, 32) << 18;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 10, 32) << 20;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 11, 32) << 22;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 12, 32) << 24;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 13, 32) << 26;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 14, 32) << 28;
	labels.x |= __shfl_sync(0xFFFFFFFF, label, 15, 32) << 30;

	labels.y  = __shfl_sync(0xFFFFFFFF, label, 16, 32);
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 17, 32) << 2;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 18, 32) << 4;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 19, 32) << 6;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 20, 32) << 8;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 21, 32) << 10;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 22, 32) << 12;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 23, 32) << 14;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 24, 32) << 16;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 25, 32) << 18;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 26, 32) << 20;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 27, 32) << 22;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 28, 32) << 24;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 29, 32) << 26;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 30, 32) << 28;
	labels.y |= __shfl_sync(0xFFFFFFFF, label, 31, 32) << 30;

	if (0 == threadIdx.x % 32) {
		*(uint2*)out = labels;
	}
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_1(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x01;

	CHECK_POW()

	out += thread * 4 / 32;

	uint32_t labels;

	labels = __shfl_sync(0xFFFFFFFF, label, 0, 32);
	labels |= __shfl_sync(0xFFFFFFFF, label, 1, 32) << 1;
	labels |= __shfl_sync(0xFFFFFFFF, label, 2, 32) << 2;
	labels |= __shfl_sync(0xFFFFFFFF, label, 3, 32) << 3;
	labels |= __shfl_sync(0xFFFFFFFF, label, 4, 32) << 4;
	labels |= __shfl_sync(0xFFFFFFFF, label, 5, 32) << 5;
	labels |= __shfl_sync(0xFFFFFFFF, label, 6, 32) << 6;
	labels |= __shfl_sync(0xFFFFFFFF, label, 7, 32) << 7;
	labels |= __shfl_sync(0xFFFFFFFF, label, 8, 32) << 8;
	labels |= __shfl_sync(0xFFFFFFFF, label, 9, 32) << 9;
	labels |= __shfl_sync(0xFFFFFFFF, label, 10, 32) << 10;
	labels |= __shfl_sync(0xFFFFFFFF, label, 11, 32) << 11;
	labels |= __shfl_sync(0xFFFFFFFF, label, 12, 32) << 12;
	labels |= __shfl_sync(0xFFFFFFFF, label, 13, 32) << 13;
	labels |= __shfl_sync(0xFFFFFFFF, label, 14, 32) << 14;
	labels |= __shfl_sync(0xFFFFFFFF, label, 15, 32) << 15;
	labels |= __shfl_sync(0xFFFFFFFF, label, 16, 32) << 16;
	labels |= __shfl_sync(0xFFFFFFFF, label, 17, 32) << 17;
	labels |= __shfl_sync(0xFFFFFFFF, label, 18, 32) << 18;
	labels |= __shfl_sync(0xFFFFFFFF, label, 19, 32) << 19;
	labels |= __shfl_sync(0xFFFFFFFF, label, 20, 32) << 20;
	labels |= __shfl_sync(0xFFFFFFFF, label, 21, 32) << 21;
	labels |= __shfl_sync(0xFFFFFFFF, label, 22, 32) << 22;
	labels |= __shfl_sync(0xFFFFFFFF, label, 23, 32) << 23;
	labels |= __shfl_sync(0xFFFFFFFF, label, 24, 32) << 24;
	labels |= __shfl_sync(0xFFFFFFFF, label, 25, 32) << 25;
	labels |= __shfl_sync(0xFFFFFFFF, label, 26, 32) << 26;
	labels |= __shfl_sync(0xFFFFFFFF, label, 27, 32) << 27;
	labels |= __shfl_sync(0xFFFFFFFF, label, 28, 32) << 28;
	labels |= __shfl_sync(0xFFFFFFFF, label, 29, 32) << 29;
	labels |= __shfl_sync(0xFFFFFFFF, label, 30, 32) << 30;
	labels |= __shfl_sync(0xFFFFFFFF, label, 31, 32) << 31;

	if (0 == threadIdx.x % 32) {
		*(uint32_t*)out = labels;
	}
}

__device__ void labels_copy(uint8_t *hashes, uint32_t hashes_count, uint8_t *output, uint32_t hash_len_bits)
{
	uint32_t label_full_bytes = hash_len_bits / 8;
	uint32_t label_total_bytes = (hash_len_bits + 7) / 8;
	uint8_t label_last_byte_length = hash_len_bits & 7;
	uint8_t label_last_byte_mask = (1 << label_last_byte_length) - 1;
	uint32_t available = 8;
	uint8_t label;
	int use_byte_copy = 0 == (hash_len_bits % 8);

	output[0] = 0;

	while (hashes_count--) {
		uint8_t *hash = hashes;
		hashes += 32;
		if (use_byte_copy) {
			memcpy(output, hash, label_full_bytes);
			output += label_full_bytes;
		}
		else {
			if (label_full_bytes) {
				if (8 == available) {
					memcpy(output, hash, label_full_bytes);
					output += label_full_bytes;
					output[0] = 0;
				}
				else {
					uint8_t lo_part_mask = (1 << available) - 1;
					uint8_t lo_part_shift = 8 - available;
					uint8_t hi_part_shift = available;

					for (int i = 0; i < label_full_bytes; i++) {
						// get 8 bits
						label = hash[i];
//						*output++ |= (label & lo_part_mask) << lo_part_shift;
						*output++ |= label << lo_part_shift;
						*output = label >> hi_part_shift;
					}
				}
			}
			uint8_t label = hash[label_full_bytes] & label_last_byte_mask;
			if (label_last_byte_length > available) {
				uint8_t lo_part_mask = (1 << available) - 1;
				uint8_t lo_part_shift = 8 - available;
				*output++ |= (label & lo_part_mask) << lo_part_shift;
				*output = label >> available;
				available = 8 - (label_last_byte_length - available);
			}
			else {
				*output |= label << (8 - available);
				available -= label_last_byte_length;
				if (0 == available) {
					available = 8;
					output++;
					if (hashes_count) {
						output[0] = 0;
					}
				}
			}
		}
	}
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_9_255(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions, uint32_t hash_len_bits)
{
	__shared__ uint32_t hashes[128][8];

	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x01;

	CHECK_POW()
 
	*(uint64_t*)&hashes[threadIdx.x][0] = hmac_pw.outer.state[0];
	*(uint64_t*)&hashes[threadIdx.x][2] = hmac_pw.outer.state[1];
	*(uint64_t*)&hashes[threadIdx.x][4] = hmac_pw.outer.state[2];
	*(uint64_t*)&hashes[threadIdx.x][6] = hmac_pw.outer.state[3];

	if (0 == threadIdx.x % 32) {
		out += thread * hash_len_bits / 8;
		labels_copy((uint8_t*)&hashes[threadIdx.x], 32, out, hash_len_bits);
	}
}

__global__ __launch_bounds__(128)
void cuda_post_keccak512_256(uint32_t *g_odata, uint8_t *out, uint64_t nonce, uint64_t *solutions)
{
	uint32_t data[20];
	uint32_t label;

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 32;
	nonce += thread;

#pragma unroll
	for (int i = 0; i < 19; i++) {
		data[i] = c_data[i];
	}
	((uint64_t*)data)[4] = nonce;

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init72(&hmac_pw, data);

	/* hmac(password, salt...) */
	uint32_t buffered = pbkdf2_hmac_update(&hmac_pw, g_odata, 128);

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	buffered = pbkdf2_hmac_buffer_update4(&hmac_pw, be, buffered);
	label = pbkdf2_hmac_finish(&hmac_pw, buffered) & 0x01;

	CHECK_POW()

	uint64_t *hashes = (uint64_t *)(out + thread * 32);

	hashes[0] = hmac_pw.outer.state[0];
	hashes[1] = hmac_pw.outer.state[1];
	hashes[2] = hmac_pw.outer.state[2];
	hashes[3] = hmac_pw.outer.state[3];
}

__global__ __launch_bounds__(128)
void cuda_post_labels_copy(uint32_t *g_odata, uint8_t *out, uint32_t hash_len_bits)
{
	__shared__ uint32_t hashes[128][8];

	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata += thread * 8;

#pragma unroll
	for (int i = 0; i < 8; i++) {
		hashes[threadIdx.x][i] = g_odata[i];
	}

	if (0 == threadIdx.x % 32) {
		out += thread * hash_len_bits / 8;
		labels_copy((uint8_t*)&hashes[threadIdx.x], 32, out, hash_len_bits);
	}
}

//
// callable host code to initialize constants and to call kernels
//

extern "C" void prepare_keccak512(_cudaState *cudaState, const uint8_t *host_pdata, const uint32_t pdata_size)
{
	if (!cudaState->keccak_inited) {
		checkCudaErrors(cudaState->cuda_id, cudaMemcpyToSymbol(c_keccak_round_constants, host_keccak_round_constants, sizeof(host_keccak_round_constants), 0, cudaMemcpyHostToDevice));
		cudaState->keccak_inited = true;
	}
	checkCudaErrors(cudaState->cuda_id, cudaMemcpyToSymbol(c_data, host_pdata, pdata_size, 0, cudaMemcpyHostToDevice));
}

extern "C" void pre_keccak512(_cudaState *cudaState, int stream, uint64_t nonce, int throughput, uint32_t r)
{
	dim3 block(128);
	dim3 grid((throughput + 127) / 128);

	cuda_pre_keccak512 << <grid, block >> >(cudaState->context_idata, nonce, r);
}

extern "C" void pre_keccak512_1_1(_cudaState *cudaState, int stream, uint64_t nonce, int throughput)
{
	dim3 block(128);
	dim3 grid((throughput + 127) / 128);

	cuda_pre_keccak512_1_1 << <grid, block >> >(cudaState->context_idata, nonce);
}

extern "C" void post_keccak512(_cudaState *cudaState, int stream, uint64_t nonce, int throughput, uint32_t hash_len_bits)
{
	dim3 block(128);
	dim3 grid((throughput + 127) / 128);

	switch (hash_len_bits) {
	case 8:
		cuda_post_keccak512_8 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	case 7:
		cuda_post_keccak512_7 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	case 6:
		cuda_post_keccak512_6 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	case 5:
		cuda_post_keccak512_5 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	case 4:
		cuda_post_keccak512_4 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	case 3:
		cuda_post_keccak512_3 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	case 2:
		cuda_post_keccak512_2 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	case 1:
		cuda_post_keccak512_1 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	case 256:
		cuda_post_keccak512_256 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions);
		break;
	default:
		if (hash_len_bits < 256) {
			cuda_post_keccak512_9_255 << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, nonce, cudaState->context_solutions, hash_len_bits);
		}
	}
}

extern "C" void post_labels_copy(_cudaState *cudaState, int stream, int throughput, uint32_t hash_len_bits)
{
	dim3 block(128);
	dim3 grid((throughput + 127) / 128);
	if (hash_len_bits > 0 && hash_len_bits <= 256) {
		cuda_post_labels_copy << <grid, block >> > ((uint32_t *)cudaState->context_odata, cudaState->context_labels, hash_len_bits);
	}
}
