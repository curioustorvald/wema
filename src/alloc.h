/*
 * WEMA - Memory allocation utilities
 */

#ifndef ALLOC_H
#define ALLOC_H

#include <stddef.h>

/*
 * Aligned memory allocation.
 * Returns NULL on failure.
 */
void *mem_alloc(size_t size);
void *mem_alloc_aligned(size_t alignment, size_t size);
void  mem_free(void *ptr);

/*
 * Allocate and zero memory.
 */
void *mem_calloc(size_t count, size_t size);

#endif /* ALLOC_H */
