/*
 * Alien Breed 3D I - PC Port
 * renderer_alloc.h - Cross-platform aligned heap allocation for render buffers
 *
 * MSVC: _aligned_malloc / _aligned_free
 * C11 POSIX: aligned_alloc / free
 */

#ifndef RENDERER_ALLOC_H
#define RENDERER_ALLOC_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <malloc.h>

static inline void *ab3d_aligned_alloc(size_t align, size_t size)
{
    if (align < sizeof(void *))
        align = sizeof(void *);
    if (size == 0)
        size = 1;
    return _aligned_malloc(size, align);
}

static inline void *ab3d_aligned_calloc(size_t align, size_t nmemb, size_t size)
{
    size_t total;
    if (nmemb != 0 && size > SIZE_MAX / nmemb)
        return NULL;
    total = nmemb * size;
    if (align < sizeof(void *))
        align = sizeof(void *);
    if (total == 0)
        total = 1;
    {
        void *p = _aligned_malloc(total, align);
        if (p)
            memset(p, 0, total);
        return p;
    }
}

static inline void ab3d_aligned_free(void *p)
{
    _aligned_free(p);
}

#else /* C11 + POSIX aligned_alloc */

static inline void *ab3d_aligned_alloc(size_t align, size_t size)
{
    size_t round;
    if (align < sizeof(void *))
        align = sizeof(void *);
    if (size == 0)
        size = 1;
    /* aligned_alloc: size must be a multiple of alignment */
    round = (size + align - 1u) & ~(align - 1u);
    return aligned_alloc(align, round);
}

static inline void *ab3d_aligned_calloc(size_t align, size_t nmemb, size_t size)
{
    size_t total, round;
    void *p;
    if (nmemb != 0 && size > SIZE_MAX / nmemb)
        return NULL;
    total = nmemb * size;
    if (align < sizeof(void *))
        align = sizeof(void *);
    if (total == 0)
        total = 1;
    round = (total + align - 1u) & ~(align - 1u);
    p = aligned_alloc(align, round);
    if (p)
        memset(p, 0, round);
    return p;
}

static inline void ab3d_aligned_free(void *p)
{
    free(p);
}

#endif

#endif /* RENDERER_ALLOC_H */
