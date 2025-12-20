/*
 * WEMA - Memory allocation utilities
 */

#include "alloc.h"
#include <stdlib.h>
#include <string.h>

void *mem_alloc(size_t size) {
    return malloc(size);
}

void *mem_alloc_aligned(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

void mem_free(void *ptr) {
    free(ptr);
}

void *mem_calloc(size_t count, size_t size) {
    return calloc(count, size);
}
