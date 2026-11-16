/*
 * Alien Breed 3D I - PC Port
 * audio.h - SDL2 audio backend
 *
 * Wraps SDL audio to replace the Amiga Paula + ProTracker-based system.
 */

#ifndef AUDIO_H
#define AUDIO_H
#include "audio_progress.h"

/* Lifecycle */
void audio_init(void);
void audio_shutdown(void);
/* Master SFX gain from ab3d.ini (0 = mute, 100 = full). Call after settings_load, typically right after audio_init. */
void audio_set_master_volume(int volume_0_to_100);
/* Amiga AB3DI.s fillscrnwater toggles the filter for both water states. */
void audio_set_underwater(int underwater);

/* Music (tracker module rendered to WAV) */
typedef void (*audio_blocking_tick_fn)(AudioProgress progress_0_to_1, void *userdata);

void audio_init_player(void);
void audio_stop_player(void);
void audio_rem_player(void);
void audio_load_module(const char *filename);
void audio_init_module(void);
void audio_play_module(void);
void audio_play_module_blocking_once(const char *filename);
void audio_play_module_blocking_once_with_tick(const char *filename,
                                               audio_blocking_tick_fn tick,
                                               void *userdata);
void audio_unload_module(void);

#if defined(__EMSCRIPTEN__)
/* One-shot module playback without blocking the main thread (Web Audio + rAF). */
int audio_start_one_shot_module(const char *filename);
unsigned int audio_music_duration_ms(void);
void audio_stop_one_shot_module(void);
#endif

/* Sound effects */
void audio_begin_frame(void); /* Reset per-frame SFX dedupe state (call once per game logic tick). */
void audio_play_sfx(int sfx_id, int volume, int channel);
void audio_play_sample(int sample_id, int volume); /* MakeSomeNoise simplified */
/* SoundPlayer.s MakeSomeNoise: original Noisevol and listener-relative world X/Z. */
int audio_distance_volume(int noise_volume, int dx, int dz);
void audio_stop_all(void);

/* In-game music (mt_init / mt_end from SoundPlayer.s) */
void audio_mt_init(void);
void audio_mt_end(void);

#endif /* AUDIO_H */
