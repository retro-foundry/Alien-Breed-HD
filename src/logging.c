/*
 * logging.c - Runtime log sink setup
 *
 * The game runs without a DOS console on Windows and writes logs to
 * ab3d.log beside the executable.
 */

#include "logging.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <SDL.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

static FILE *g_log_stdout = NULL;
static FILE *g_log_stderr = NULL;
static int g_log_ready = 0;
static char g_log_path[1024] = "ab3d.log";

static int ab3d_log_is_suppressed(const char *fmt)
{
    if (!fmt) return 0;
    if (strncmp(fmt, "[AUDIO]", 7) == 0) return 1;
    if (strncmp(fmt, "[IO]", 4) == 0) return 1;
    if (strncmp(fmt, "[SB]", 4) == 0) return 1;
    if (strncmp(fmt, "[LEVEL]", 7) == 0) return 1;
    if (strncmp(fmt, "[3DOBJ]", 7) == 0) return 1;
    if (strncmp(fmt, "SetupGame complete", 18) == 0) return 1;
    return 0;
}

static void ab3d_log_build_path(void)
{
    char *base = SDL_GetBasePath();
    if (!base || !base[0]) {
        if (base) SDL_free(base);
        snprintf(g_log_path, sizeof(g_log_path), "ab3d.log");
        return;
    }

    snprintf(g_log_path, sizeof(g_log_path), "%sab3d.log", base);
    SDL_free(base);
}

int ab3d_log_init_file(void)
{
    if (g_log_ready) return 1;

    ab3d_log_build_path();

#if defined(_WIN32) || defined(_WIN64)
    /* Detach from any inherited console; this build is file-log only. */
    FreeConsole();
#endif

    g_log_stdout = freopen(g_log_path, "w", stdout);
    if (!g_log_stdout) {
        return 0;
    }

    g_log_stderr = freopen(g_log_path, "a", stderr);
    if (!g_log_stderr) {
        return 0;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    g_log_ready = 1;
    fprintf(stdout, "[LOG] Writing output to: %s\n", g_log_path);
    return 1;
}

void ab3d_log_shutdown(void)
{
    if (stdout) fflush(stdout);
    if (stderr) fflush(stderr);
}

const char *ab3d_log_path(void)
{
    return g_log_path;
}

int ab3d_log_printf(const char *fmt, ...)
{
    int out;
    va_list ap;

    if (ab3d_log_is_suppressed(fmt)) return 0;

    va_start(ap, fmt);
    out = vfprintf(stdout, fmt, ap);
    va_end(ap);

    fflush(stdout);
    return out;
}
