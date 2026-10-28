#ifndef AB3D_PLATFORM_DOS_H
#define AB3D_PLATFORM_DOS_H
#include <stddef.h>
#include "game_state.h"
enum { DOS_UPDATE, DOS_INPUT, DOS_RENDER, DOS_PRESENT, DOS_AUDIO, DOS_BUCKETS };
extern int dos_benchmark_frames;
extern int dos_capture_next_frame;
extern int dos_check_enabled;
extern int dos_frontend_check_enabled;
extern int dos_fixed_test_ticks;
void dos_platform_init(int argc,char **argv);
void dos_platform_bootstrap(const char *executable);
const char *dos_asset_alias(const char *logical);
void audio_dos_pump(void);
void dos_profile_begin(int bucket);
void dos_profile_end(void);
void dos_profile_frame(void);
void dos_profile_reset(void);
void dos_capture_frame(const char *filename);
void dos_memory_report(const char *phase);
void *dos_shared_asset_alloc(size_t size);
void dos_shared_asset_free(void *ptr);
void *dos_scene_asset_alloc(unsigned slot,size_t size);
void dos_scene_asset_free(void *ptr);
void dos_check_frontend(GameState *state);
void dos_check_saves(GameState *state);
void dos_check_zone_index(const LevelState *level);
void dos_check_ui_screens(GameState *state);
void audio_dos_reset_stats(void);
void audio_dos_report_stats(void);
#endif
