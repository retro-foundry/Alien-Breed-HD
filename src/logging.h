#ifndef LOGGING_H
#define LOGGING_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize log redirection to ab3d.log next to the executable.
 * Returns non-zero on success, 0 on failure. */
int ab3d_log_init_file(void);

/* Flush and close log file handles. Safe to call multiple times. */
void ab3d_log_shutdown(void);

/* Returns absolute path of active log file, or "ab3d.log" fallback. */
const char *ab3d_log_path(void);

/* printf-compatible logger used by #define printf ab3d_log_printf in source files. */
int ab3d_log_printf(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* LOGGING_H */
