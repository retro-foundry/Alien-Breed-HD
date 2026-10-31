/*
 * Alien Breed 3D I - PC Port
 * io.h - PC file I/O backend
 *
 * Implements Amiga-style level, asset, prefs, and password loading
 * using standard C/SDL I/O primitives.
 */

#ifndef IO_H
#define IO_H

#include "game_state.h"

/* Lifecycle */
void io_init(void);
void io_shutdown(void);

/* Level loading */
int  io_load_level_data(LevelState *level, int level_num);
int  io_load_level_graphics(LevelState *level, int level_num);
int  io_load_level_clips(LevelState *level, int level_num);
int16_t io_load_level_exit_zone(LevelState *level, int level_num, int16_t original_exit);
void io_release_level_memory(LevelState *level);

/* Asset loading */
void io_load_walls(void);
void io_load_floor(void);
void io_load_sky(void);
void io_load_gun_graphics(void);
void io_load_vec_objects(void);  /* Load 3D vector (.vec) objects into POLYOBJECTS table */

/* Debug: dump all loaded textures as BMP images into textures/ */
void io_dump_textures(void);
void io_load_objects(void);
void io_load_sfx(void);
void io_load_panel(void);

/* Prefs */
void io_load_prefs(char *prefs_buf, int buf_size);
void io_save_prefs(const char *prefs_buf, int buf_size);

/* Passwords */
void io_load_passwords(void);
void io_save_passwords(void);

/* Build full path for a file under the executable-local data/ directory. */
void io_make_data_path(char *buf, size_t bufsize, const char *subpath);
/* Build full path for a file beside the executable (e.g. "savegame.bin"). */
void io_make_exe_path(char *buf, size_t bufsize, const char *subpath);

#endif /* IO_H */
