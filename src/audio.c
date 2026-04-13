#/*
 * Alien Breed 3D I - PC Port
 * audio.c - SDL2 sound effects/music backend
 *
 * Sample names and order from Amiga LoadFromDisk.s SFX_NAMES.
 * Prefers Amiga originals: sounds/<name> (no extension) or sounds/<name>.raw,
 * falling back to WAV when needed.
 */

#include "audio.h"
#include "io.h"
#include <SDL.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logging.h"
#define printf ab3d_log_printf

#define MAX_SAMPLES      64
#define MAX_CHANNELS     24
#define NUM_NAMED_SFX    28
#define MUSIC_DEFAULT_VOL 176 /* 0..255 */
/* AB3DI.s sets AUDxPER=443 for SFX playback. Paula PAL audio clock is 3,546,895 Hz. */
#define AMIGA_SFX_PERIOD 443
#define AMIGA_PAULA_PAL_CLOCK 3546895
#define AMIGA_SFX_RATE ((AMIGA_PAULA_PAL_CLOCK + (AMIGA_SFX_PERIOD / 2)) / AMIGA_SFX_PERIOD) /* ~= 8007 Hz */
#define DEFAULT_FREQ     AMIGA_SFX_RATE
#define DEFAULT_FORMAT  AUDIO_S16SYS
#define DEFAULT_CHANNELS 1
#define PLAYBACK_SPEED_DIV 1
#if defined(__EMSCRIPTEN__)
/* Web: slightly larger buffer than desktop to reduce underruns; 2048 was too much latency. */
#define AUDIO_SAMPLES_DESIRED 1024
#else
#define AUDIO_SAMPLES_DESIRED 512
#endif

/* Amiga LoadFromDisk.s SFX_NAMES: size in bytes (one byte per sample, 8-bit signed). */
static const unsigned int amiga_sfx_sizes[NUM_NAMED_SFX] = {
    4400, 7200, 5400, 4600, 3400, 8400, 8000, 4000, 8600, 6200,
    1200, 4000, 2200, 3000, 5600, 11600, 7200, 7400, 9200, 5000,
    4000, 8800, 9000, 1800, 3400, 1600, 11000, 8400
};
/* Amiga LoadFromDisk.s SFX_NAMES order: sample_id -> disk/sounds/<name> (we use sounds/<name>.wav or raw) */
static const char *const sfx_names[NUM_NAMED_SFX] = {
    "scream",      /* 0 */
    "fire!",       /* 1 ShootName */
    "munch",       /* 2 */
    "shoot.dm",    /* 3 PooGunName */
    "collect",     /* 4 */
    "newdoor",     /* 5 */
    "splash",      /* 6 BassName */
    "footstep3",   /* 7 StompName */
    "lowscream",   /* 8 */
    "baddiegun",   /* 9 */
    "switch",      /* 10 */
    "switch1",     /* 11 ReloadName */
    "noammo",      /* 12 */
    "splotch",     /* 13 */
    "splatpop",    /* 14 */
    "boom",        /* 15 */
    "newhiss",     /* 16 */
    "howl1",       /* 17 */
    "howl2",       /* 18 */
    "pant",        /* 19 */
    "whoosh",      /* 20 */
    "shotgun",     /* 21 ShotGunName */
    "flame",       /* 22 */
    "muffledfoot", /* 23 (Amiga: MuffledFoot) */
    "footclop",    /* 24 */
    "footclank",   /* 25 */
    "teleport",    /* 26 */
    "halfwormpain" /* 27 (Amiga: HALFWORMPAIN) */
};

/* One preloaded sample (converted to device format) */
typedef struct {
    Uint8  *data;
    Uint32  length;   /* bytes */
    int     loaded;
} LoadedSample;

/* One mixing channel (playing a sample) */
typedef struct {
    const Uint8 *sample_data;
    Uint32       sample_len;
    Uint32       position;   /* bytes played */
    int          volume;     /* 0-255 -> SDL_MixAudio uses 0-128 */
    int          sample_id;
} Channel;

typedef struct {
    Uint8  *data;
    Uint32  length;
    Uint32  position;
    int     loaded;
    int     playing;
    int     loop;
    int     volume; /* 0..255 */
} MusicTrack;

static SDL_AudioDeviceID g_device = 0;
static SDL_AudioSpec     g_spec;
static LoadedSample      g_samples[MAX_SAMPLES];
static Channel           g_channels[MAX_CHANNELS];
static MusicTrack        g_music;
static int               g_audio_ready = 0;
static int               g_master_volume = 100; /* 0..100, scales per-sample volume in audio_play_sample */
/* Per-game-frame SFX dedupe:
 * g_sfx_frame_id is advanced by audio_begin_frame() once per logic tick.
 * A sample can be queued at most once for each frame id. */
static Uint32            g_sfx_frame_id = 0;
static Uint32            g_sample_last_played_frame[MAX_SAMPLES];

static int channel_is_free(const Channel *ch)
{
    return (ch->sample_data == NULL || ch->position >= ch->sample_len);
}

/* Pick the best channel for a new one-shot SFX:
 * 1) Prefer truly free channels.
 * 2) If all busy, steal the least intrusive one (lowest volume, nearest end). */
static int choose_channel_for_play(void)
{
    int best_free = -1;
    int best_busy = -1;
    int best_busy_score = 0x7FFFFFFF;

    for (int c = 0; c < MAX_CHANNELS; c++) {
        Channel *ch = &g_channels[c];
        if (channel_is_free(ch)) {
            best_free = c;
            break;
        }

        Uint32 remain = ch->sample_len - ch->position;
        int vol = ch->volume;
        if (vol < 0) vol = 0;
        if (vol > 255) vol = 255;
        /* Lower score = better steal candidate. Bias toward quieter + near-finished. */
        int score = (int)remain + (vol << 8);
        if (score < best_busy_score) {
            best_busy_score = score;
            best_busy = c;
        }
    }

    if (best_free >= 0) return best_free;
    if (best_busy >= 0) return best_busy;
    return 0;
}

static void music_stop_locked(void)
{
    g_music.playing = 0;
    g_music.position = 0;
}

static void music_unload_locked(void)
{
    if (g_music.data) {
        SDL_free(g_music.data);
    }
    g_music.data = NULL;
    g_music.length = 0;
    g_music.position = 0;
    g_music.loaded = 0;
    g_music.playing = 0;
    g_music.loop = 1;
    g_music.volume = MUSIC_DEFAULT_VOL;
}

static void audio_callback(void *userdata, Uint8 *stream, int len)
{
    (void)userdata;
    memset(stream, 0, (size_t)len);

    if (g_music.playing && g_music.loaded && g_music.data && g_music.length > 0) {
        Uint32 out_pos = 0;
        Uint32 out_remain = (Uint32)len;

        while (out_remain > 0 && g_music.playing) {
            if (g_music.position >= g_music.length) {
                if (g_music.loop) {
                    g_music.position = 0;
                } else {
                    g_music.playing = 0;
                    break;
                }
            }

            Uint32 in_remain = g_music.length - g_music.position;
            Uint32 to_mix = (out_remain < in_remain) ? out_remain : in_remain;

            int mix_vol = g_music.volume;
            if (mix_vol < 0) mix_vol = 0;
            if (mix_vol > 255) mix_vol = 255;
            if (g_master_volume <= 0) {
                mix_vol = 0;
            } else {
                mix_vol = (mix_vol * g_master_volume + 50) / 100;
                if (mix_vol > 255) mix_vol = 255;
            }
            mix_vol = (mix_vol * 128) / 255;

            if (mix_vol > 0) {
                SDL_MixAudioFormat(stream + out_pos, g_music.data + g_music.position,
                                   g_spec.format, to_mix, mix_vol);
            }

            g_music.position += to_mix;
            out_pos += to_mix;
            out_remain -= to_mix;
        }
    }

    for (int c = 0; c < MAX_CHANNELS; c++) {
        Channel *ch = &g_channels[c];
        if (ch->sample_data == NULL || ch->position >= ch->sample_len)
            continue;

        /* Normal speed (PLAYBACK_SPEED_DIV 1): consume len bytes per callback */
        Uint32 remain = ch->sample_len - ch->position;
        Uint32 to_mix = (Uint32)len / PLAYBACK_SPEED_DIV;
        if (to_mix > remain)
            to_mix = remain;

        int mix_vol = ch->volume;
        if (mix_vol < 0) mix_vol = 0;
        if (mix_vol > 255) mix_vol = 255;
        mix_vol = (mix_vol * 128) / 255;

        SDL_MixAudioFormat(stream, ch->sample_data + ch->position,
                           g_spec.format, to_mix, mix_vol);
#if PLAYBACK_SPEED_DIV > 1
        if (to_mix * 2 <= (Uint32)len)
            SDL_MixAudioFormat(stream + to_mix, ch->sample_data + ch->position,
                               g_spec.format, to_mix, mix_vol);
#endif
        ch->position += to_mix;

        if (ch->position >= ch->sample_len) {
            ch->sample_data = NULL;
            ch->sample_id = -1;
        }
    }
}

/* Lowercase the filename part of path in place (from last '/' or '\\' to end). */
static void path_filename_to_lower(char *path)
{
    char *p = strrchr(path, '/');
    char *b = strrchr(path, '\\');
    if (b && (!p || b > p)) p = b;
    if (p) p++; else p = path;
    for (; *p; p++) *p = (char)tolower((unsigned char)*p);
}

static int load_wav_converted(const char *subpath, char *path_out, size_t path_size,
                              Uint8 **buf_out, Uint32 *len_out)
{
    SDL_AudioSpec want;
    Uint8 *buf = NULL;
    Uint32 len = 0;
    char path[512];

    io_make_data_path(path, sizeof(path), subpath);
    if (!SDL_LoadWAV(path, &want, &buf, &len)) {
        char path_lower[512];
        snprintf(path_lower, sizeof(path_lower), "%s", path);
        path_filename_to_lower(path_lower);
        if (!SDL_LoadWAV(path_lower, &want, &buf, &len)) {
            return 0;
        }
        snprintf(path, sizeof(path), "%s", path_lower);
    }

    SDL_AudioCVT cvt;
    if (SDL_BuildAudioCVT(&cvt, want.format, want.channels, want.freq,
                          g_spec.format, g_spec.channels, g_spec.freq) < 0) {
        SDL_FreeWAV(buf);
        return 0;
    }

    cvt.len = (int)len;
    cvt.buf = (Uint8 *)SDL_malloc((size_t)len * cvt.len_mult);
    if (!cvt.buf) {
        SDL_FreeWAV(buf);
        return 0;
    }
    memcpy(cvt.buf, buf, (size_t)len);
    SDL_FreeWAV(buf);

    if (SDL_ConvertAudio(&cvt) < 0) {
        SDL_free(cvt.buf);
        return 0;
    }

    *buf_out = cvt.buf;
    *len_out = (Uint32)cvt.len_cvt;
    if (path_out && path_size > 0) {
        snprintf(path_out, path_size, "%s", path);
    }
    return 1;
}

/* Load Amiga raw SFX: no header, 8-bit signed PCM, one byte per sample, ~8007 Hz.
 * Tries sounds/<name> (no extension) then sounds/<name>.raw. Returns 1 with buf/len/spec set, 0 on failure. */
static int load_amiga_raw(int id, char *path_out, size_t path_size,
                          SDL_AudioSpec *spec_out, Uint8 **buf_out, Uint32 *len_out)
{
    if (id >= NUM_NAMED_SFX) return 0;

    const char *name = sfx_names[id];
    unsigned int expect_len = amiga_sfx_sizes[id];
    char path[512];
    char path_lower[512];
    FILE *f = NULL;

    /* Prefer Amiga originals: sounds/<name> (no extension), then sounds/<name>.raw */
    io_make_data_path(path, sizeof(path), "sounds/");
    size_t base_len = strlen(path);
    snprintf(path + base_len, sizeof(path) - base_len, "%s", name);
    f = fopen(path, "rb");
    if (!f) {
        strncpy(path_lower, path, sizeof(path_lower) - 1);
        path_lower[sizeof(path_lower) - 1] = '\0';
        path_filename_to_lower(path_lower);
        f = fopen(path_lower, "rb");
        if (f) { (void)strncpy(path, path_lower, sizeof(path) - 1); path[sizeof(path) - 1] = '\0'; }
    }
    if (!f) {
        snprintf(path + base_len, sizeof(path) - base_len, "%s.raw", name);
        f = fopen(path, "rb");
        if (!f) {
            strncpy(path_lower, path, sizeof(path_lower) - 1);
            path_lower[sizeof(path_lower) - 1] = '\0';
            path_filename_to_lower(path_lower);
            f = fopen(path_lower, "rb");
            if (f) { (void)strncpy(path, path_lower, sizeof(path) - 1); path[sizeof(path) - 1] = '\0'; }
        }
    }
    if (!f) return 0;

    fseek(f, 0, SEEK_END);
    long file_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_len <= 0) {
        fclose(f);
        return 0;
    }
    /* One byte per sample; cap at 2x expected size to avoid bad dumps. */
    unsigned int to_read = (unsigned int)file_len;
    if (to_read > expect_len * 2u) to_read = expect_len * 2;
    Uint8 *raw = (Uint8 *)SDL_malloc(to_read);
    if (!raw) { fclose(f); return 0; }
    if (fread(raw, 1, to_read, f) != to_read) {
        SDL_free(raw);
        fclose(f);
        return 0;
    }
    fclose(f);

    /* Amiga raw: 8-bit signed PCM, one byte per sample (LoadFromDisk.s / raw_to_wav.py). */
    Uint32 samples = to_read;
    Uint32 s16_len = samples * 2;
    Sint16 *s16 = (Sint16 *)SDL_malloc(s16_len);
    if (!s16) { SDL_free(raw); return 0; }
    for (Uint32 i = 0; i < samples; i++) {
        /* Sign-extend byte to 16-bit and scale to full range for better level. */
        int v = (int)((int8_t)raw[i]) * 256;
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        s16[i] = (Sint16)v;
    }
    SDL_free(raw);

    spec_out->freq = AMIGA_SFX_RATE;
    spec_out->format = AUDIO_S16SYS;
    spec_out->channels = 1;
    *buf_out = (Uint8 *)s16;
    *len_out = s16_len;
    if (path_out && path_size) {
        strncpy(path_out, path, path_size - 1);
        path_out[path_size - 1] = '\0';
    }
    return 1;
}

static int load_one_sample(int id)
{
    char path[512];
    SDL_AudioSpec want;
    Uint8 *buf = NULL;
    Uint32 len = 0;
    int from_amiga = 0;

    /* Prefer Amiga originals (no .wav): sounds/<name> or sounds/<name>.raw; then fall back to .wav */
    if (id < NUM_NAMED_SFX && load_amiga_raw(id, path, sizeof(path), &want, &buf, &len)) {
        from_amiga = 1;
    } else {
        char subpath[80];
        if (id < NUM_NAMED_SFX) {
            snprintf(subpath, sizeof(subpath), "sounds/%s.wav", sfx_names[id]);
        } else {
            snprintf(subpath, sizeof(subpath), "sounds/%d.wav", id);
        }
        io_make_data_path(path, sizeof(path), subpath);

        if (!SDL_LoadWAV(path, &want, &buf, &len)) {
            char path_lower[512];
            snprintf(path_lower, sizeof(path_lower), "%s", path);
            path_filename_to_lower(path_lower);
            if (!SDL_LoadWAV(path_lower, &want, &buf, &len)) {
                strncpy(path, path_lower, sizeof(path) - 1);
                path[sizeof(path) - 1] = '\0';
                return 0;  /* not found - skip silently for high IDs */
            }
            strncpy(path, path_lower, sizeof(path) - 1);
            path[sizeof(path) - 1] = '\0';
        }
    }

    /* Convert to device format so we can mix in callback */
    SDL_AudioCVT cvt;
    if (SDL_BuildAudioCVT(&cvt, want.format, want.channels, want.freq,
                           g_spec.format, g_spec.channels, g_spec.freq) < 0) {
        printf("[AUDIO] sample %d: unsupported format (%s)\n", id, path);
        if (from_amiga) SDL_free(buf); else SDL_FreeWAV(buf);
        return 0;
    }

    cvt.len = (int)len;
    cvt.buf = (Uint8 *)SDL_malloc((size_t)len * cvt.len_mult);
    if (!cvt.buf) {
        printf("[AUDIO] sample %d: out of memory\n", id);
        if (from_amiga) SDL_free(buf); else SDL_FreeWAV(buf);
        return 0;
    }
    memcpy(cvt.buf, buf, (size_t)len);
    if (from_amiga) SDL_free(buf); else SDL_FreeWAV(buf);

    if (SDL_ConvertAudio(&cvt) < 0) {
        printf("[AUDIO] sample %d: convert failed\n", id);
        SDL_free(cvt.buf);
        return 0;
    }

    g_samples[id].data = cvt.buf;
    g_samples[id].length = (Uint32)(cvt.len_cvt);
    g_samples[id].loaded = 1;
    printf("[AUDIO] loaded %d (%s): %s%s (%u bytes)\n",
           id, id < NUM_NAMED_SFX ? sfx_names[id] : "?", path, from_amiga ? " [Amiga raw]" : "", (unsigned)g_samples[id].length);
    return 1;
}

static void free_samples(void)
{
    for (int i = 0; i < MAX_SAMPLES; i++) {
        if (g_samples[i].loaded && g_samples[i].data) {
            SDL_free(g_samples[i].data);
            g_samples[i].data = NULL;
            g_samples[i].length = 0;
            g_samples[i].loaded = 0;
        }
    }
}

void audio_init(void)
{
    printf("[AUDIO] init\n");
    memset(g_samples, 0, sizeof(g_samples));
    memset(g_channels, 0, sizeof(g_channels));
    memset(&g_music, 0, sizeof(g_music));
    g_music.loop = 1;
    g_music.volume = MUSIC_DEFAULT_VOL;
    memset(g_sample_last_played_frame, 0, sizeof(g_sample_last_played_frame));
    for (int i = 0; i < MAX_CHANNELS; i++) {
        g_channels[i].sample_id = -1;
    }
    g_audio_ready = 0;
    g_sfx_frame_id = 0;
    g_device = 0;

    if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) {
        printf("[AUDIO] SDL_Init AUDIO failed: %s\n", SDL_GetError());
        return;
    }

    g_spec.freq = DEFAULT_FREQ;
    g_spec.format = DEFAULT_FORMAT;
    g_spec.channels = (Uint8)DEFAULT_CHANNELS;
    g_spec.samples = AUDIO_SAMPLES_DESIRED;
    g_spec.callback = audio_callback;
    g_spec.userdata = NULL;

    g_device = SDL_OpenAudioDevice(NULL, 0, &g_spec, &g_spec, 0);
    if (g_device == 0) {
        printf("[AUDIO] SDL_OpenAudioDevice failed: %s\n", SDL_GetError());
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return;
    }

    /* Load samples: prefer Amiga originals (sounds/<name> no extension), else sounds/<name>.wav */
    {
        char first_path[512];
        io_make_data_path(first_path, sizeof(first_path), "sounds/scream");
        printf("[AUDIO] Loading sound effects from data/sounds (e.g. %s or %s.wav)\n", first_path, first_path);
    }
    int loaded_count = 0;
    for (int i = 0; i < MAX_SAMPLES; i++) {
        loaded_count += load_one_sample(i);
    }

    SDL_PauseAudioDevice(g_device, 0);
    g_audio_ready = 1;
    if (loaded_count > 0) {
        printf("[AUDIO] Loaded %d sound effect(s): ", loaded_count);
        for (int i = 0, n = 0; i < MAX_SAMPLES && n < loaded_count; i++) {
            if (g_samples[i].loaded) {
                printf("%s%s", n ? ", " : "", i < NUM_NAMED_SFX ? sfx_names[i] : "?");
                n++;
            }
        }
        printf("\n");
    }
    if (loaded_count < NUM_NAMED_SFX) {
        fprintf(stderr, "[AUDIO] FATAL: required sounds missing (loaded %d of %d)\n",
                loaded_count, NUM_NAMED_SFX);
        exit(1);
    }
}

void audio_set_master_volume(int volume_0_to_100)
{
    if (volume_0_to_100 < 0)
        volume_0_to_100 = 0;
    else if (volume_0_to_100 > 100)
        volume_0_to_100 = 100;
    g_master_volume = volume_0_to_100;
}

void audio_shutdown(void)
{
    if (g_device) {
        SDL_LockAudioDevice(g_device);
        music_unload_locked();
        SDL_UnlockAudioDevice(g_device);
        SDL_CloseAudioDevice(g_device);
        g_device = 0;
    } else {
        music_unload_locked();
    }
    free_samples();
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    g_audio_ready = 0;
    printf("[AUDIO] shutdown\n");
}

static int str_ends_with_ci(const char *s, const char *suffix)
{
    size_t slen = strlen(s);
    size_t tlen = strlen(suffix);
    if (slen < tlen) return 0;
    const char *tail = s + (slen - tlen);
    for (size_t i = 0; i < tlen; i++) {
        if (tolower((unsigned char)tail[i]) != tolower((unsigned char)suffix[i])) {
            return 0;
        }
    }
    return 1;
}

static void normalize_module_subpath(const char *filename, char *out, size_t out_size)
{
    if (!filename || !*filename || !out || out_size == 0) {
        if (out && out_size > 0) out[0] = '\0';
        return;
    }

    snprintf(out, out_size, "%s", filename);
    for (char *p = out; *p; p++) {
        if (*p == '\\') *p = '/';
    }

    if (str_ends_with_ci(out, ".mt")) {
        size_t n = strlen(out);
        if (n + 2 < out_size) {
            out[n - 3] = '.';
            out[n - 2] = 'w';
            out[n - 1] = 'a';
            out[n] = 'v';
            out[n + 1] = '\0';
        }
    }
}

void audio_init_player(void)       { /* no-op: single mixed callback backend */ }
void audio_stop_player(void)
{
    if (!g_audio_ready || g_device == 0) return;
    SDL_LockAudioDevice(g_device);
    music_stop_locked();
    SDL_UnlockAudioDevice(g_device);
}

void audio_rem_player(void)        { audio_unload_module(); }

void audio_load_module(const char *filename)
{
    if (!g_audio_ready || g_device == 0 || !filename || !*filename) return;

    char subpath[256];
    char loaded_path[512];
    Uint8 *data = NULL;
    Uint32 len = 0;

    normalize_module_subpath(filename, subpath, sizeof(subpath));
    if (!subpath[0]) return;

    if (!load_wav_converted(subpath, loaded_path, sizeof(loaded_path), &data, &len)) {
        SDL_LockAudioDevice(g_device);
        music_unload_locked();
        SDL_UnlockAudioDevice(g_device);
        printf("[MUSIC] missing/unreadable module wav: %s\n", subpath);
        return;
    }

    SDL_LockAudioDevice(g_device);
    music_unload_locked();
    g_music.data = data;
    g_music.length = len;
    g_music.position = 0;
    g_music.loaded = 1;
    g_music.playing = 0;
    g_music.loop = 1;
    g_music.volume = MUSIC_DEFAULT_VOL;
    SDL_UnlockAudioDevice(g_device);

    printf("[MUSIC] loaded: %s (%u bytes converted)\n", loaded_path, (unsigned)len);
}

void audio_init_module(void)
{
    if (!g_audio_ready || g_device == 0) return;
    SDL_LockAudioDevice(g_device);
    if (g_music.loaded) {
        g_music.position = 0;
    }
    SDL_UnlockAudioDevice(g_device);
}

void audio_play_module(void)
{
    if (!g_audio_ready || g_device == 0) return;
    SDL_LockAudioDevice(g_device);
    if (g_music.loaded && g_music.data && g_music.length > 0) {
        g_music.playing = 1;
    }
    SDL_UnlockAudioDevice(g_device);
}

void audio_play_module_blocking_once_with_tick(const char *filename,
                                               audio_blocking_tick_fn tick,
                                               void *userdata)
{
    if (!g_audio_ready || g_device == 0) return;
    if (!filename || !*filename) return;

    audio_load_module(filename);

    int started = 0;
    SDL_LockAudioDevice(g_device);
    if (g_music.loaded && g_music.data && g_music.length > 0) {
        g_music.loop = 0;
        g_music.position = 0;
        g_music.playing = 1;
        started = 1;
    }
    SDL_UnlockAudioDevice(g_device);

    if (!started) return;

    if (tick) tick(0.0f, userdata);

    printf("[MUSIC] playing once: %s\n", filename);

    for (;;) {
        int playing;
        Uint32 pos = 0;
        Uint32 len = 0;
        SDL_LockAudioDevice(g_device);
        playing = g_music.playing;
        pos = g_music.position;
        len = g_music.length;
        SDL_UnlockAudioDevice(g_device);
        if (tick) {
            float progress = 0.0f;
            if (len > 0) {
                progress = (float)pos / (float)len;
                if (progress < 0.0f) progress = 0.0f;
                if (progress > 1.0f) progress = 1.0f;
            }
            tick(progress, userdata);
        }
        if (!playing) break;
        SDL_Delay(10);
    }

    if (tick) tick(1.0f, userdata);
}

#if defined(__EMSCRIPTEN__)
int audio_start_one_shot_module(const char *filename)
{
    if (!g_audio_ready || g_device == 0) return 0;
    if (!filename || !*filename) return 0;

    audio_load_module(filename);

    int started = 0;
    SDL_LockAudioDevice(g_device);
    if (g_music.loaded && g_music.data && g_music.length > 0) {
        g_music.loop = 0;
        g_music.position = 0;
        g_music.playing = 1;
        started = 1;
    }
    SDL_UnlockAudioDevice(g_device);

    if (started)
        printf("[MUSIC] one-shot start: %s\n", filename);
    return started;
}

unsigned int audio_music_duration_ms(void)
{
    if (!g_audio_ready || g_device == 0) return 1u;
    int bpf = (int)g_spec.channels * (SDL_AUDIO_BITSIZE(g_spec.format) / 8);
    if (bpf < 1) bpf = 1;
    if (g_music.length == 0) return 1u;
    double bytes_per_sec = (double)g_spec.freq * (double)bpf;
    Uint32 duration_ms = (Uint32)((double)g_music.length / bytes_per_sec * 1000.0 + 0.5);
    if (duration_ms < 1u) duration_ms = 1u;
    if (duration_ms > 600000u) duration_ms = 600000u;
    return (unsigned int)duration_ms;
}

void audio_stop_one_shot_module(void)
{
    if (!g_device) return;
    SDL_LockAudioDevice(g_device);
    music_stop_locked();
    SDL_UnlockAudioDevice(g_device);
}
#endif

void audio_play_module_blocking_once(const char *filename)
{
    audio_play_module_blocking_once_with_tick(filename, NULL, NULL);
}

void audio_unload_module(void)
{
    if (g_device) {
        SDL_LockAudioDevice(g_device);
        music_unload_locked();
        SDL_UnlockAudioDevice(g_device);
    } else {
        music_unload_locked();
    }
}

void audio_begin_frame(void)
{
    /* 0 means "dedupe disabled" before first frame marker. */
    g_sfx_frame_id++;
    if (g_sfx_frame_id == 0) {
        /* Wrapped after ~4 billion frames: clear table and keep 0 reserved. */
        memset(g_sample_last_played_frame, 0, sizeof(g_sample_last_played_frame));
        g_sfx_frame_id = 1;
    }
}

void audio_play_sfx(int sfx_id, int volume, int channel)
{
    (void)channel;
    audio_play_sample(sfx_id, volume);
}

void audio_play_sample(int sample_id, int volume)
{
    if (!g_audio_ready || g_device == 0)
        return;
    if (sample_id < 0 || sample_id >= MAX_SAMPLES)
        return;
    if (!g_samples[sample_id].loaded || g_samples[sample_id].data == NULL) {
        /* Log once per missing sample (no spam) - helps debug "no sound" */
        static unsigned char logged[MAX_SAMPLES];
        if (sample_id < MAX_SAMPLES && !logged[sample_id]) {
            logged[sample_id] = 1;
            if (sample_id < NUM_NAMED_SFX) {
                printf("[AUDIO] play_sample(%d (%s), %d) - not loaded (add %s or %s.wav to data/sounds)\n",
                       sample_id, sfx_names[sample_id], volume, sfx_names[sample_id], sfx_names[sample_id]);
            } else {
                printf("[AUDIO] play_sample(%d, %d) - not loaded (add %d.wav to data/sounds)\n",
                       sample_id, volume, sample_id);
            }
        }
        return;
    }
    if (volume < 0) volume = 0;
    if (volume > 255) volume = 255;
    if (g_master_volume <= 0)
        return;
    volume = (volume * g_master_volume + 50) / 100;
    if (volume <= 0)
        return;
    if (g_sfx_frame_id != 0) {
        if (g_sample_last_played_frame[sample_id] == g_sfx_frame_id)
            return;
        g_sample_last_played_frame[sample_id] = g_sfx_frame_id;
    }

    SDL_LockAudioDevice(g_device);

    /* Pick a channel that minimizes audible stomping. */
    int chan = choose_channel_for_play();
    Channel *ch = &g_channels[chan];

    ch->sample_data = g_samples[sample_id].data;
    ch->sample_len  = g_samples[sample_id].length;
    ch->position    = 0;
    ch->volume      = volume;
    ch->sample_id   = sample_id;

    SDL_UnlockAudioDevice(g_device);
}

void audio_stop_all(void)
{
    if (!g_audio_ready || g_device == 0) return;
    SDL_LockAudioDevice(g_device);
    for (int c = 0; c < MAX_CHANNELS; c++) {
        g_channels[c].sample_data = NULL;
        g_channels[c].position = 0;
        g_channels[c].sample_id = -1;
    }
    SDL_UnlockAudioDevice(g_device);
}

void audio_mt_init(void)
{
    static int logged_disabled = 0;
    if (!g_audio_ready || g_device == 0) return;
    /* User preference: disable in-level background music. */
    audio_unload_module();
    if (!logged_disabled) {
        logged_disabled = 1;
        printf("[MUSIC] in-game music disabled\n");
    }
}

void audio_mt_end(void)
{
    if (!g_audio_ready || g_device == 0) return;
    SDL_LockAudioDevice(g_device);
    if (g_music.playing) {
        music_stop_locked();
        SDL_UnlockAudioDevice(g_device);
        printf("[MUSIC] in-game music stopped\n");
        return;
    }
    SDL_UnlockAudioDevice(g_device);
}
