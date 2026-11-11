/* Compare the actual DOS mixer with the previous sample-major equations. */
#include "../src/platform/dos/audio_dos.c"
#include <assert.h>
#ifdef NDEBUG
#error This PCM oracle requires assertions
#endif

enum { FRAMES=513 };
static int8_t pcm[FRAMES+8];
static int16_t actual_words[FRAMES*2+8];
static uint8_t expected_bytes[sizeof(actual_words)];
static int32_t accum[FRAMES+2];
static Voice expected_voices[VOICES];
static unsigned checked;
static uint32_t random_state=0x1745abcd;
static uint32_t random_value(void)
{ random_state=random_state*1664525u+1013904223u;return random_state; }

typedef struct { int x1,x2,y1,y2,wet; } RefFilter;
static int ref_clip(int64_t value)
{ return value>32767?32767:value< -32768?-32768:(int)value; }
__attribute__((noinline)) static int ref_gain(int value,int volume)
{ return (int)((int64_t)value*volume*256/255); }
static int ref_filter(int dry,RefFilter *f)
{
    if(!underwater && !f->wet) { f->x1=f->x2=f->y1=f->y2=0;return dry; }
    int filtered=ref_clip(((int64_t)b0*dry+(int64_t)b1*f->x1+(int64_t)b2*f->x2-
                          (int64_t)a1*f->y1-(int64_t)a2*f->y2)>>14);
    f->x2=f->x1;f->x1=dry;f->y2=f->y1;f->y1=filtered;
    if(underwater) { if(f->wet<RATE/50)f->wet++; } else if(f->wet)f->wet--;
    return dry+(int)((int64_t)(filtered-dry)*f->wet/(RATE/50));
}

static uint8_t music_data[4097];
static unsigned ref_music_pos;
static int ref_music_playing;
static int ref_music_sample(void)
{
    if(!ref_music_playing || !music)return 0;
    if(ref_music_pos>=music_length) {
        if(!music_loop){ref_music_playing=0;return 0;}
        ref_music_pos=0;
    }
    return ref_gain((int)music_data[ref_music_pos++]-128,176);
}

static void check_chunk(unsigned frames)
{
    unsigned stride=(unsigned)spec.channels*(spec.format==SDL_AUDIO_U8?1u:2u);
    chunk_bytes=(int)(frames*stride);
    memset(actual_words,0xa5,sizeof(actual_words));
    memcpy(expected_bytes,actual_words,sizeof(actual_words));
    for(unsigned i=0;i<FRAMES+2;i++)accum[i]=0x6b6b6b6b;
    mix_buffer=(unsigned char *)(actual_words+4);mix_sums=accum+1;
    memcpy(expected_voices,voices,sizeof(voices));
    RefFilter f={filter_x1,filter_x2,filter_y1,filter_y2,wet};
    ref_music_pos=music_pos;ref_music_playing=music_playing;
    unsigned active_count=0;
    for(unsigned c=0;c<VOICES;c++) {
        Voice *v=expected_voices+c;
        if(v->id>=0 && v->pos<samples[v->id].length)active_count++;
        else v->id=-1;
    }
    int silence=!active_count && !music_playing && !wet && !underwater;
    for(unsigned i=0;i<frames;i++) {
        int sum=ref_music_sample();
        if(!silence)for(unsigned c=0;c<VOICES;c++) {
            Voice *v=expected_voices+c;
            if(v->id<0)continue;
            Sample *s=samples+v->id;
            if(v->pos>=s->length){v->id=-1;continue;}
            sum+=ref_gain(s->data[v->pos++],v->volume);
        }
        int out=silence?0:ref_filter(ref_clip((int64_t)sum*master/100),&f);
        for(int ch=0;ch<spec.channels;ch++) {
            unsigned at=8+i*stride+ch*(spec.format==SDL_AUDIO_U8?1:2);
            if(spec.format==SDL_AUDIO_U8)expected_bytes[at]=(uint8_t)((out>>8)+128);
            else { int16_t value=(int16_t)out;memcpy(expected_bytes+at,&value,2); }
        }
    }
    mix_chunk();
    assert(!memcmp(actual_words,expected_bytes,sizeof(actual_words)));
    assert(!memcmp(voices,expected_voices,sizeof(voices)));
    assert(filter_x1==f.x1 && filter_x2==f.x2 && filter_y1==f.y1 && filter_y2==f.y2 && wet==f.wet);
    assert(music_pos==ref_music_pos && music_playing==ref_music_playing);
    assert(accum[0]==0x6b6b6b6b);
    for(unsigned i=frames+1;i<FRAMES+2;i++)assert(accum[i]==0x6b6b6b6b);
    checked++;
}

static void reset_state(void)
{
    for(unsigned c=0;c<VOICES;c++)voices[c]=(Voice){-1,0,0,0};
    master=100;underwater=wet=0;music_playing=0;music_pos=music_at=music_have=0;
    filter_x1=1234;filter_x2=-5678;filter_y1=9123;filter_y2=-301;
}

int main(void)
{
    freopen("AUDIO.ERR","w",stderr);setvbuf(stderr,NULL,_IONBF,0);setvbuf(stdout,NULL,_IOLBF,0);
    for(unsigned i=0;i<sizeof(pcm);i++) {
        int value=(int)((i*73+19)&255);pcm[i]=(int8_t)(value<128?value:value-256);
    }
    double k=tan(3.141592653589793*2200.0/RATE),norm=1.0/(1.0+1.41421356237*k+k*k);
    b0=(int32_t)(k*k*norm*16384);b1=2*b0;b2=b0;
    a1=(int32_t)(2*(k*k-1)*norm*16384);a2=(int32_t)((1-1.41421356237*k+k*k)*norm*16384);
    spec.channels=1;spec.format=SDL_AUDIO_S16LE;samples[0]=(Sample){pcm,256};
    for(unsigned volume=0;volume<256;volume++) {
        reset_state();voices[0]=(Voice){0,(int)volume,0,17};check_chunk(256);
    }
    puts("PASS: all 65536 signed PCM/volume combinations match wide scalar division");
    for(unsigned test=0;test<384;test++) {
        reset_state();spec.channels=1+(test&1);spec.format=test&2?SDL_AUDIO_U8:SDL_AUDIO_S16LE;
        static const unsigned lengths[]={0,1,2,31,255,256,512,513};
        static const unsigned counts[]={1,2,7,31,255,256,511,512,513};
        for(unsigned i=0;i<8;i++)samples[i]=(Sample){pcm+i,lengths[(i+test)%8]};
        unsigned count=test%25;
        for(unsigned c=0;c<count;c++) {
            unsigned id=c%8,len=samples[id].length;
            unsigned pos=test%4==0?0:test%4==1?len:test%4==2?(len?len-1:0):random_value()%(len+2);
            int volume=test%3==0?255:test%3==1?(c&1?0:255):(int)(random_value()%256);
            voices[c]=(Voice){(int)id,volume,pos,c+3};
        }
        master=(int)(test%101);underwater=!!(test&4);wet=(int)(test%(RATE/50+1));
        check_chunk(counts[test%9]);check_chunk(counts[(test+4)%9]);
    }
    puts("PASS: 768 mixed chunks preserve 0-24 voices, exact endings, clipping, master volume, wet ramp, filter state, both PCM formats/channels and guards");
    FILE *file=fopen("MUSIC.BIN","w+b");assert(file);
    for(unsigned i=0;i<sizeof(music_data);i++)music_data[i]=(uint8_t)(i*197+53);
    assert(fwrite(music_data,1,sizeof(music_data),file)==sizeof(music_data));
    static const unsigned music_lengths[]={1,73,4096,4097};
    for(unsigned test=0;test<16;test++) {
        reset_state();music=file;music_start=0;music_length=music_lengths[test%4];music_loop=!!(test&4);music_playing=1;
        assert(!fseek(file,0,SEEK_SET));
        spec.channels=1+(test&1);spec.format=test&2?SDL_AUDIO_U8:SDL_AUDIO_S16LE;
        samples[0]=(Sample){pcm,513};samples[1]=(Sample){pcm+1,512};
        voices[0]=(Voice){0,255,0,1};voices[1]=(Voice){1,127,0,1};
        underwater=!!(test&8);wet=159;
        for(unsigned chunk=0;chunk<12;chunk++)check_chunk(512);
    }
    fclose(file);music=NULL;
    printf("PASS: music EOF, looping, 4096-byte refills and concurrent voices; %u total chunks exact\n",checked);
    puts("DOS audio regression passed");return 0;
}
