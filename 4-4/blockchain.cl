#include "sha256.cl"

int calc_hash(__global unsigned char* data,
              unsigned char* hash,
              int data_length,
              int difficulty,
              long nonce) {
    int i;
    int matched;
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, nonce, data, data_length);
	sha256_final(&ctx, hash);

    matched = 1;
    for (i = 0; i < difficulty && matched == 1; i++) {
        if (hash[i] != 0) {
            matched = 0;
            break;
        }
    }
    return matched;
}

__kernel void find_nonce(__global unsigned char* data,
                         __global unsigned char* hash,
                         volatile __global unsigned int* nonce,
                         int data_length,
                         int difficulty,
                         int nonce_group_size) {

    int global_id = get_global_id(0);
    int nonce_start = nonce_group_size * global_id;
    int nonce_end = nonce_start + nonce_group_size;
    unsigned char local_hash[32];
    int i, j;

    for (i = nonce_start; i < nonce_end; i++) {
        if (calc_hash(data, local_hash, data_length, difficulty, i) == 1) {
            // TODO: need a atomic memory copy
            printf("hash found: ");
            for (j = 0; j < 32; j++) {
                printf("%02x", local_hash[j]);
            }
            printf("\n");
            atomic_xchg(nonce, i);
            break;
        }
    }
}
