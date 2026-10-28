/* DOS transport for the original ~8007 Hz signed Paula samples.
 * GloomAmiga's manual Sound Blaster IRQ ring avoids cooperative SDL threads.
 * SFX keep their native samples/rate; music streams from disk in bounded blocks. */
#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dpmi.h>
#include <crt0.h>
#include "audio.h"
#include "platform_dos.h"
#include "audio_samples.h"
#include "io.h"
#include "logging.h"
#define RATE 8007
#define VOICES 24
#define SAMPLES 64
extern int SDL_DOSSoundBlasterGetQueuedAudioSize(SDL_AudioDeviceID);
extern int SDL_DOSSoundBlasterGetAudioBufferSize(SDL_AudioDeviceID);
extern int SDL_DOSSoundBlasterGetAudioChunkSize(SDL_AudioDeviceID);
extern bool SDL_DOSSoundBlasterQueueAudio(SDL_AudioDeviceID,const void *,int);
extern void SDL_DOSSoundBlasterClearQueuedAudio(SDL_AudioDeviceID);
extern unsigned SDL_DOSSoundBlasterGetUnderruns(SDL_AudioDeviceID);
static unsigned underruns_start;

typedef struct { int8_t *data; unsigned length; } Sample;
typedef struct { int id,volume; unsigned pos,frame; } Voice;
static Sample samples[SAMPLES];
static int sample_selector=-1;
static unsigned char *sample_bank;
static Voice voices[VOICES];
static unsigned frame_id,last_frame[SAMPLES];
static SDL_AudioStream *stream;
static SDL_AudioDeviceID device;
static SDL_AudioSpec spec;
static unsigned char *mix_buffer;
static int chunk_bytes,target_bytes,master=100,underwater,wet;
static FILE *music;
static unsigned music_length,music_pos,music_start;
static int music_playing,music_loop=1;
static unsigned char music_buf[4096];
static unsigned music_at,music_have;
static int32_t b0,b1,b2,a1,a2,filter_x1,filter_x2,filter_y1,filter_y2;

static void fatal(const char *op,const char *path)
{
    fprintf(stderr,"[DOS AUDIO] %s: %s (%s)\n",op,path,SDL_GetError());
    exit(1);
}
static unsigned le32(const unsigned char *p)
{ return p[0]|((unsigned)p[1]<<8)|((unsigned)p[2]<<16)|((unsigned)p[3]<<24); }
static int clamp16(int v) { return v>32767?32767:v< -32768?-32768:v; }
static int filter(int dry)
{
    const int ramp=RATE/50;
    if(!underwater && !wet) { filter_x1=filter_x2=filter_y1=filter_y2=0; return dry; }
    int filtered=(b0*dry+b1*filter_x1+b2*filter_x2-a1*filter_y1-a2*filter_y2)>>14;
    filtered=clamp16(filtered); filter_x2=filter_x1;filter_x1=dry;filter_y2=filter_y1;filter_y1=filtered;
    if(underwater) { if(wet<ramp)wet++; } else if(wet>0)wet--;
    return dry+(filtered-dry)*wet/ramp;
}
static int music_sample(void)
{
    if(!music_playing || !music) return 0;
    if(music_at==music_have) {
        if(music_pos>=music_length) {
            if(!music_loop) { music_playing=0;return 0; }
            fseek(music,music_start,SEEK_SET);music_pos=0;
        }
        unsigned n=music_length-music_pos;
        if(n>sizeof(music_buf))n=sizeof(music_buf);
        music_have=fread(music_buf,1,n,music);music_at=0;
        if(music_have!=n)fatal("Truncated music stream","WAV data");
    }
    music_pos++;
    return ((int)music_buf[music_at++]-128)*176*256/255;
}
void audio_dos_pump(void)
{
    if(!device)return;
    dos_profile_begin(DOS_AUDIO);
    int queued=SDL_DOSSoundBlasterGetQueuedAudioSize(device);
    if(queued<0)fatal("Invalid Sound Blaster ring","queue state");
    while(queued+chunk_bytes<=target_bytes) {
        int active[VOICES],active_count=0;
        for(int c=0;c<VOICES;c++) {
            Voice *v=&voices[c];
            if(v->id>=0 && v->pos<samples[v->id].length)active[active_count++]=c;
            else v->id=-1;
        }
        if(!active_count && !music_playing && !wet && !underwater) {
            memset(mix_buffer,spec.format==SDL_AUDIO_U8?128:0,chunk_bytes);
            if(!SDL_DOSSoundBlasterQueueAudio(device,mix_buffer,chunk_bytes))fatal("Queue silence","Sound Blaster");
            queued+=chunk_bytes;
            continue;
        }
        int bytes_per_frame=spec.channels*(spec.format==SDL_AUDIO_U8?1:2);
        int count=chunk_bytes/bytes_per_frame;
        for(int i=0;i<count;i++) {
            int sum=music_sample();
            for(int c=0;c<active_count;c++) {
                Voice *v=&voices[active[c]];
                if(v->id<0)continue;
                Sample *s=&samples[v->id];
                if(v->pos>=s->length){v->id=-1;continue;}
                sum+=(int)s->data[v->pos++]*v->volume*256/255;
            }
            int out=filter(clamp16(sum*master/100));
            for(int ch=0;ch<spec.channels;ch++) {
                if(spec.format==SDL_AUDIO_U8)mix_buffer[i*spec.channels+ch]=(unsigned char)((out>>8)+128);
                else ((int16_t *)mix_buffer)[i*spec.channels+ch]=(int16_t)out;
            }
        }
        if(!SDL_DOSSoundBlasterQueueAudio(device,mix_buffer,chunk_bytes))
            fatal("Queue failed","Sound Blaster ring");
        queued+=chunk_bytes;
    }
    dos_profile_end();
}
void audio_dos_reset_stats(void)
{
    audio_dos_pump();
    underruns_start=device?SDL_DOSSoundBlasterGetUnderruns(device):0;
}
void audio_dos_report_stats(void)
{
    printf("[DOSCHECK] audio underruns during profile=%u\n",
        device?SDL_DOSSoundBlasterGetUnderruns(device)-underruns_start:0);
}
void audio_init(void)
{
    for(int i=0;i<VOICES;i++)voices[i].id=-1;
    if(getenv("AB3D_DOS_SOUND") && !strcmp(getenv("AB3D_DOS_SOUND"),"0")) {
        printf("[DOS AUDIO] Explicit sound-off profile\n");return;
    }
    SDL_SetHint(SDL_HINT_AUDIO_DRIVER,"soundblaster");
    SDL_SetHint("SDL_DOS_MANUAL_AUDIO_PUMP","1");
    SDL_SetHint("SDL_DOS_MANUAL_AUDIO_FREQ","8007");
    /* The backend selects its scheduler during subsystem initialization. */
    if(SDL_InitSubSystem(SDL_INIT_AUDIO)<0)fatal("Initialize SDL audio","Sound Blaster");
    SDL_AudioSpec want={SDL_AUDIO_S16LE,1,RATE};
    stream=SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,&want,NULL,NULL);
    if(!stream)fatal("Open device","set BLASTER=A220 I7 D1 H5 T6 as appropriate for your card");
    device=SDL_GetAudioStreamDevice(stream);
    int frames;
    if(!SDL_GetAudioDeviceFormat(device,&spec,&frames) || spec.freq!=RATE ||
       (spec.format!=SDL_AUDIO_U8 && spec.format!=SDL_AUDIO_S16LE))fatal("Unsupported format","Sound Blaster");
    chunk_bytes=SDL_DOSSoundBlasterGetAudioChunkSize(device);
    int capacity=SDL_DOSSoundBlasterGetAudioBufferSize(device);
    target_bytes=chunk_bytes*3;
    if(chunk_bytes<=0 || target_bytes>capacity)fatal("Invalid buffer sizes","Sound Blaster");
    mix_buffer=malloc(chunk_bytes);if(!mix_buffer)fatal("Out of memory","mixer chunk");
    unsigned lengths[NUM_NAMED_SFX],total=0;
    for(int id=0;id<NUM_NAMED_SFX;id++) {
        char sub[80],path[512];snprintf(sub,sizeof(sub),"sounds/%s",sfx_names[id]);
        io_make_data_path(path,sizeof(path),sub);FILE *f=fopen(path,"rb");
        if(!f)fatal("Missing original sample",path);
        if(fseek(f,0,SEEK_END) || ftell(f)<=0)fatal("Invalid sample size",path);
        unsigned n=(unsigned)ftell(f);fclose(f);
        if(n>amiga_sfx_sizes[id]*2)n=amiga_sfx_sizes[id]*2;
        lengths[id]=n;total+=n;
    }
    int segment=__dpmi_allocate_dos_memory((total+15u)/16u,&sample_selector);
    if(segment<0)fatal("Not enough conventional RAM for original samples","load fewer DOS resident programs");
    sample_bank=(unsigned char *)(__djgpp_conventional_base+(unsigned long)segment*16u);
    unsigned offset=0;
    for(int id=0;id<NUM_NAMED_SFX;id++) {
        char sub[80],path[512];snprintf(sub,sizeof(sub),"sounds/%s",sfx_names[id]);
        io_make_data_path(path,sizeof(path),sub);FILE *f=fopen(path,"rb");
        if(!f)fatal("Missing original sample",path);
        fseek(f,0,SEEK_END);unsigned n=ftell(f);rewind(f);
        if(n>amiga_sfx_sizes[id]*2)n=amiga_sfx_sizes[id]*2;
        if(!n)fatal("Empty sample",path);
        if(n!=lengths[id])fatal("Sample size changed during loading",path);
        samples[id].data=(int8_t *)sample_bank+offset;offset+=n;samples[id].length=n;
        if(!samples[id].data || fread(samples[id].data,1,n,f)!=n)fatal("Load sample",path);
        fclose(f);
    }
    double k=tan(3.141592653589793*2200.0/RATE);
    double norm=1.0/(1.0+1.41421356237*k+k*k);
    b0=(int32_t)(k*k*norm*16384);b1=2*b0;b2=b0;
    a1=(int32_t)(2*(k*k-1)*norm*16384);a2=(int32_t)((1-1.41421356237*k+k*k)*norm*16384);
    SDL_ResumeAudioStreamDevice(stream);
    printf("[DOS AUDIO] rate=%d channels=%d chunk=%d target=%d; integer mixing, manual IRQ ring\n",spec.freq,spec.channels,chunk_bytes,target_bytes);
}
void audio_begin_frame(void) { if(++frame_id==0){memset(last_frame,0,sizeof(last_frame));frame_id=1;} }
void audio_play_sample(int id,int volume)
{
    if(!device || id<0 || id>=NUM_NAMED_SFX)return;
    if(volume<0)volume=0;if(volume>255)volume=255;
    if(last_frame[id]==frame_id && frame_id) {
        for(int c=0;c<VOICES;c++)if(voices[c].id==id && voices[c].frame==frame_id && voices[c].volume<volume)voices[c].volume=volume;
        return;
    }
    int slot=-1,score=0x7fffffff;
    for(int c=0;c<VOICES;c++) {
        Voice *v=&voices[c];
        if(v->id<0 || v->pos>=samples[v->id].length){slot=c;break;}
        int s=(int)(samples[v->id].length-v->pos)+(v->volume<<8);
        if(s<score){score=s;slot=c;}
    }
    voices[slot]=(Voice){id,volume,0,frame_id};last_frame[id]=frame_id;
}
void audio_play_sfx(int id,int volume,int channel) { (void)channel;audio_play_sample(id,volume); }
void audio_set_master_volume(int volume) { master=volume<0?0:volume>100?100:volume; }
void audio_set_underwater(int enabled) { underwater=enabled!=0; }
void audio_stop_all(void)
{
    for(int c=0;c<VOICES;c++)voices[c].id=-1;
    if(device)SDL_DOSSoundBlasterClearQueuedAudio(device);
}
void audio_unload_module(void) { music_playing=0;if(music)fclose(music);music=NULL;music_length=music_pos=music_have=music_at=0; }
void audio_load_module(const char *filename)
{
    audio_unload_module();if(!device)return;
    char sub[256],path[512];snprintf(sub,sizeof(sub),"%s",filename);
    char *dot=strrchr(sub,'.');if(dot)snprintf(dot,sizeof(sub)-(dot-sub),".wav");
    io_make_data_path(path,sizeof(path),sub);music=fopen(path,"rb");if(!music)fatal("Missing music",path);
    unsigned char h[12];if(fread(h,1,12,music)!=12 || memcmp(h,"RIFF",4) || memcmp(h+8,"WAVE",4))fatal("Invalid WAV",path);
    int valid=0;
    for(;;) {
        unsigned char c[8];if(fread(c,1,8,music)!=8)fatal("Missing WAV data",path);
        unsigned n=le32(c+4);
        if(!memcmp(c,"fmt ",4)) {
            unsigned char fmt[16];if(n<16 || fread(fmt,1,16,music)!=16)fatal("Invalid WAV format",path);
            valid=fmt[0]==1 && fmt[1]==0 && fmt[2]==1 && fmt[3]==0 && le32(fmt+4)==RATE && fmt[14]==8;
            fseek(music,n-16+(n&1),SEEK_CUR);
        } else if(!memcmp(c,"data",4)) {
            if(!valid)fatal("Expected mono unsigned 8-bit WAV at 8007 Hz",path);
            music_length=n;music_start=ftell(music);break;
        } else fseek(music,n+(n&1),SEEK_CUR);
    }
    music_loop=1;
}
void audio_init_player(void) { audio_stop_player(); }
void audio_stop_player(void) { music_playing=0; }
void audio_rem_player(void) { audio_unload_module(); }
void audio_init_module(void) { if(music){fseek(music,music_start,SEEK_SET);music_pos=music_have=music_at=0;} }
void audio_play_module(void) { if(music)music_playing=1; }
void audio_play_module_blocking_once_with_tick(const char *file,audio_blocking_tick_fn tick,void *arg)
{
    audio_load_module(file);if(!music)return;
    audio_init_module();music_loop=0;music_playing=1;
    while(music_playing) {
        audio_dos_pump();
        if(tick)tick(music_length?(float)music_pos/music_length:1.0f,arg);
        SDL_PumpEvents();SDL_Delay(10);
    }
    if(tick)tick(1.0f,arg);
}
void audio_play_module_blocking_once(const char *f) { audio_play_module_blocking_once_with_tick(f,NULL,NULL); }
void audio_mt_init(void) { audio_unload_module(); }
void audio_mt_end(void) { underwater=0;audio_stop_player(); }
void audio_shutdown(void)
{
    audio_unload_module();if(stream)SDL_DestroyAudioStream(stream);stream=NULL;device=0;
    for(int i=0;i<SAMPLES;i++){samples[i].data=NULL;samples[i].length=0;}
    if(sample_selector>=0){__dpmi_free_dos_memory(sample_selector);sample_selector=-1;sample_bank=NULL;}
    free(mix_buffer);mix_buffer=NULL;
}
