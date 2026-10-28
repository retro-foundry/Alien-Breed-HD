/*
 * Alien Breed 3D I - PC Port
 * renderer_alloc.h - Cross-platform aligned heap allocation for render buffers
 *
 * MSVC: _aligned_malloc / _aligned_free
 * C11 POSIX: aligned_alloc / free
 */

#ifndef RENDERER_ALLOC_H
#define RENDERER_ALLOC_H

/* Typical CPU L1 cache line size (x86/x64, AArch64). Used for framebuffer and
 * clip-buffer base pointers so hot loops see aligned store streams. */
#ifndef AB3D_CACHE_LINE_SIZE
#define AB3D_CACHE_LINE_SIZE 64u
#endif

#include <stddef.h>
#include <stdint.h>
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

#elif defined(AB3D_DOS)

/* The bundled DJGPP memalign rejects ordinary unaligned malloc results.
 * Own the padding/header so every render allocation has the required alignment. */
static inline void *ab3d_aligned_alloc(size_t align, size_t size)
{
    if (align < sizeof(void *)) align = sizeof(void *);
    if ((align & (align - 1)) != 0 || size > SIZE_MAX - align - sizeof(void *)) return NULL;
    void *base = malloc(size + align - 1 + sizeof(void *));
    if (!base) return NULL;
    uintptr_t ptr = ((uintptr_t)base + sizeof(void *) + align - 1) & ~(uintptr_t)(align - 1);
    ((void **)ptr)[-1] = base;
    return (void *)ptr;
}
static inline void *ab3d_aligned_calloc(size_t align, size_t count, size_t size)
{
    if (count && size > SIZE_MAX / count) return NULL;
    size_t total = count * size;
    void *ptr = ab3d_aligned_alloc(align, total);
    if (ptr) memset(ptr, 0, total);
    return ptr;
}
static inline void ab3d_aligned_free(void *ptr)
{
    if (ptr) free(((void **)ptr)[-1]);
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
