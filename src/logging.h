#ifndef LOGGING_H
#define LOGGING_H

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static inline int ab3d_log_is_suppressed(const char *fmt)
{
    if (!fmt) return 0;
    if (strncmp(fmt, "[AUDIO]", 7) == 0) return 1;
    if (strncmp(fmt, "[IO]", 4) == 0) return 1;
    if (strncmp(fmt, "[SB]", 4) == 0) return 1;
    if (strncmp(fmt, "[LEVEL]", 7) == 0) return 1;
    if (strncmp(fmt, "[3DOBJ]", 7) == 0) return 1;
    if (strncmp(fmt, "[CONTROL]", 9) == 0) return 1;
    if (strncmp(fmt, "SetupGame complete", 18) == 0) return 1;
    return 0;
}

static inline int ab3d_log_printf(const char *fmt, ...)
{
    if (ab3d_log_is_suppressed(fmt)) return 0;
    va_list ap;
    va_start(ap, fmt);
    int out = vprintf(fmt, ap);
    va_end(ap);
    return out;
}

#endif /* LOGGING_H */
