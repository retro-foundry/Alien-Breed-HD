#include "platform_dos.h"
#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dpmi.h>
#include <crt0.h>
#include <unistd.h>
#include "assets_dos.h"
#include "game_loop.h"
#include "control_loop.h"
#include "game_data.h"
#include "level.h"
#include "audio.h"
#include "player.h"
#include "renderer_alloc.h"
#include "display.h"
#include "renderer.h"
#include "logging.h"
int dos_benchmark_frames;
int dos_capture_next_frame;
int dos_check_enabled;
int dos_frontend_check_enabled;
int dos_fixed_test_ticks;
static int stack[16],depth;
static Uint64 last,cost[DOS_BUCKETS];
static unsigned frames;
static Uint64 frame_start,frame_total,frame_times[1024];
/* Original floor/bump/smooth/water files each occupy one 64 KiB bank. Keep
 * those four resident banks in conventional RAM instead of wasting that RAM
 * while exhausting the much smaller XMS heap of a 4 MB DOS machine. */
static unsigned char *asset_banks;
static int asset_bank_selector=-1;
static unsigned asset_bank_mask;
static unsigned scene_mask,scene_offset[4],scene_size[4];
static void ensure_asset_banks(void)
{
    if(asset_banks)return;
    int segment=__dpmi_allocate_dos_memory(20480,&asset_bank_selector);
    if(segment<0){fprintf(stderr,"DOS needs 320 KiB free conventional RAM for resident texture banks\n");exit(1);}
    asset_banks=(unsigned char *)(__djgpp_conventional_base+(unsigned long)segment*16u);
}
static void release_unused_banks(void)
{
    if(asset_banks && !asset_bank_mask && !scene_mask) {
        __dpmi_free_dos_memory(asset_bank_selector);asset_bank_selector=-1;asset_banks=NULL;
    }
}
void *dos_shared_asset_alloc(size_t size)
{
    if(!size || size>65536u){fprintf(stderr,"DOS shared bank size %lu exceeds 64 KiB\n",(unsigned long)size);exit(1);}
    ensure_asset_banks();
    for(unsigned i=0;i<4;i++)if(!(asset_bank_mask&(1u<<i))) {
        asset_bank_mask|=1u<<i;return asset_banks+i*65536u;
    }
    fprintf(stderr,"DOS shared texture banks exhausted\n");exit(1);
}
void dos_shared_asset_free(void *ptr)
{
    if(!ptr)return;
    uintptr_t offset=(uintptr_t)ptr-(uintptr_t)asset_banks;
    if(!asset_banks || offset>=262144u || offset%65536u || !(asset_bank_mask&(1u<<(offset/65536u)))) {
        fprintf(stderr,"DOS invalid shared texture bank free\n");exit(1);
    }
    asset_bank_mask&=~(1u<<(offset/65536u));
    release_unused_banks();
}
void *dos_scene_asset_alloc(unsigned slot,size_t size)
{
    if(slot>=4 || !size || size>65536u || (scene_mask&(1u<<slot))) {
        fprintf(stderr,"Invalid DOS scene bank allocation\n");exit(1);
    }
    ensure_asset_banks();
    unsigned start=0,n=((unsigned)size+15u)&~15u;
    for(;;) {
        bool collision=false;
        for(unsigned i=0;i<4;i++)if((scene_mask&(1u<<i)) && start<scene_offset[i]+scene_size[i] && start+n>scene_offset[i]) {
            start=scene_offset[i]+scene_size[i];collision=true;break;
        }
        if(!collision)break;
    }
    if(start+n>65536u){fprintf(stderr,"DOS sky/gun bank exceeds 64 KiB\n");exit(1);}
    scene_mask|=1u<<slot;scene_offset[slot]=start;scene_size[slot]=n;
    return asset_banks+262144u+start;
}
void dos_scene_asset_free(void *ptr)
{
    if(!ptr)return;
    for(unsigned i=0;i<4;i++)if((scene_mask&(1u<<i)) && ptr==asset_banks+262144u+scene_offset[i]) {
        scene_mask&=~(1u<<i);release_unused_banks();return;
    }
    fprintf(stderr,"Invalid DOS scene bank free\n");exit(1);
}
void dos_platform_bootstrap(const char *executable)
{
    /* SDL_MAIN_HANDLED bypasses SDL_RunApp, whose DOS filesystem backend
     * records argv[0] here for SDL_GetBasePath/searchpath. Set it before logging. */
    extern const char *SDL_argv0;
    _crt0_startup_flags &= ~_CRT0_FLAG_LOCK_MEMORY;
    /* DJGPP's non-moving sbrk rounds each new DPMI zone to 64 KiB. Give its
     * normal allocator one contiguous game heap first, rather than losing
     * the padding of many small zones while loading differently-sized assets.
     * The 1.5 MiB heap plus this EXE/stack fits the 4 MB physical target. */
    const size_t heap_size=1536u*1024u;
    volatile unsigned char *heap=malloc(heap_size);
    if(!heap){fprintf(stderr,"Cannot reserve the 1.5 MiB DOS game heap\n");exit(1);}
    heap[0]=0;heap[heap_size-1]=0;
    free((void *)heap);

    SDL_argv0=executable;
    SDL_SetMainReady();
}

static void check_key(GameState *state,GameLoopCtx *ctx,SDL_Scancode code)
{
    if(!Ab3dDos_PushKey(code,true)) {
        fprintf(stderr,"DOS diagnostic key enqueue failed: %s\n",SDL_GetError());exit(1);
    }
    game_loop_front_menu_tick(state,ctx);
    SDL_Delay(10);game_loop_front_menu_tick(state,ctx);
    display_dos_check_text_screen("menu",0);
    if(!Ab3dDos_PushKey(code,false)) {
        fprintf(stderr,"DOS diagnostic key release failed: %s\n",SDL_GetError());exit(1);
    }
    game_loop_front_menu_tick(state,ctx);
    SDL_Delay(10);game_loop_front_menu_tick(state,ctx);
}
void dos_check_frontend(GameState *state)
{
    for(size_t size=1;size<2048;size+=31) {
        unsigned char *p=ab3d_aligned_calloc(64,size,1);
        if(!p || (uintptr_t)p%64) { fprintf(stderr,"DOS aligned allocation check failed\n");exit(1); }
        for(size_t i=0;i<size;i++) {
            if(p[i]) { fprintf(stderr,"DOS calloc zero-fill check failed\n");exit(1); }
            p[i]=0xa5;
        }
        ab3d_aligned_free(p);
    }
    int benchmark=dos_benchmark_frames;
    dos_benchmark_frames=0;
    GameLoopCtx ctx;
    game_loop_front_menu_init(&ctx,state);
    for(int i=0;i<12;i++) { game_loop_front_menu_tick(state,&ctx);SDL_Delay(20); }
    dos_capture_frame("MENU.BMP");
    check_key(state,&ctx,SDL_SCANCODE_RETURN);
    check_key(state,&ctx,SDL_SCANCODE_UP);
    check_key(state,&ctx,SDL_SCANCODE_RETURN);
    if(ctx.ingame_menu_result!=GAME_LOOP_FRONT_MENU_NEW_GAME) {
        fprintf(stderr,"DOS keyboard/menu check failed: result=%d\n",ctx.ingame_menu_result);exit(1);
    }
    printf("[DOSCHECK] title, streamed music and keyboard New Game passed\n");
    dos_benchmark_frames=benchmark;
}
void dos_check_zone_index(const LevelState *level)
{
    int slots=level_zone_slot_count(level),checked=0;
    for(int i=0;i<slots;i++) {
        const uint8_t *p=level->zone_adds+i*4;
        int32_t offset=(int32_t)(((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3]);
        for(int delta=-1;delta<=1;delta++) {
            int64_t value=(int64_t)offset+delta;if(value<INT32_MIN || value>INT32_MAX)continue;
            int expected=-1;
            if(value>=0)for(int z=0;z<slots;z++) {
                p=level->zone_adds+z*4;
                int32_t candidate=(int32_t)(((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3]);
                if(candidate==value){expected=z;break;}
            }
            if(level_zone_index_from_room_offset(level,(int32_t)value)!=expected) {
                fprintf(stderr,"DOS zone index differs from original linear lookup at %ld\n",(long)value);exit(1);
            }
            checked++;
        }
    }
    printf("[DOSCHECK] %d compact zone lookups match original zoneAdds ordering\n",checked);
}
static unsigned save_hash_bytes(unsigned hash,const void *data,size_t size)
{
    const unsigned char *bytes=data;
    for(size_t i=0;i<size;i++)hash=(hash^bytes[i])*16777619u;
    return hash;
}
static unsigned save_level_hash(const GameState *state)
{
    const LevelState *level=&state->level;unsigned hash=2166136261u;
    hash=save_hash_bytes(hash,level->data,level->data_byte_count);
    hash=save_hash_bytes(hash,level->graphics,level->graphics_byte_count);
    hash=save_hash_bytes(hash,level->clips,level->clips_byte_count);
    hash=save_hash_bytes(hash,level->object_points,(size_t)level->num_object_points*8u);
    /* reserved stores renderer-derived zone hints; it is deliberately absent
     * from the existing v3 disk format and is rebuilt after a load. */
    for(unsigned i=0;i<level->automap_seen_count;i++)
        hash=save_hash_bytes(hash,&level->automap_seen_walls[i],offsetof(AutomapSeenWall,reserved));
    return hash;
}
void dos_check_saves(GameState *state)
{
    player_save_position(state);
    for(int i=0;i<PLAYER_AUTOSAVE_SLOT_COUNT;i++)player_save_autosave(state);
    for(int i=0;i<PLAYER_AUTOSAVE_SLOT_COUNT;i++) {
        PlayerAutosaveInfo info;
        if(!player_read_autosave_info(i,&info)) {
            fprintf(stderr,"DOS autosave slot %d failed\n",i);exit(1);
        }
    }
    unsigned hash=save_level_hash(state);
    int energy=state->energy;
    for(int reload=0;reload<8;reload++) {
        state->energy=energy-1;
        if(player_load_save_from_file(state)!=PLAYER_SAVE_LOAD_NEED_LEVEL_RELOAD) {
            fprintf(stderr,"DOS full quickload staging failed\n");exit(1);
        }
        state->debug_f9_need_level_reload=true;
        if(!play_the_game_after_game_loop(state)){fprintf(stderr,"DOS quickload reload not requested\n");exit(1);}
        bool ready=true;
        play_the_game_prepare_level(state,&ready);
        if(state->energy!=energy || save_level_hash(state)!=hash){fprintf(stderr,"DOS quickload state mismatch\n");exit(1);}
    }
    FILE *save=fopen("savegame.bin","r+b");
    if(!save || fseek(save,0,SEEK_END) || ftell(save)<1 || ftruncate(fileno(save),ftell(save)-1)) {
        fprintf(stderr,"Cannot create truncated DOS save diagnostic\n");exit(1);
    }
    fclose(save);
    if(player_load_save_from_file(state)!=PLAYER_SAVE_LOAD_FAILED || state->energy!=energy || save_level_hash(state)!=hash) {
        fprintf(stderr,"DOS truncated save was accepted or changed live state\n");exit(1);
    }
    player_save_position(state);
    printf("[DOSCHECK] eight full quickloads and all five 8.3 autosave slots passed; truncated payload rejected\n");
}
void dos_check_ui_screens(GameState *state)
{
    /* Repeat after gameplay, with discovered map walls and live enemy state. */
    dos_check_saves(state);
    display_draw_display(state);
    state->automap_visible=true;
    display_present_last_frame(state);dos_capture_frame("MAP.BMP");
    state->automap_visible=false;
    int bars=state->cfg_hud_bars;state->cfg_hud_bars=0;
    display_present_last_frame(state);dos_capture_frame("HUD.BMP");
    state->cfg_hud_bars=bars;
    int16_t conditions = game_conditions;
    game_conditions = (game_conditions & ~15) | 5;
    display_present_last_frame(state);dos_capture_frame("KEYS.BMP");
    game_conditions = conditions;
    dos_ui_check_images();
    display_dos_check_hud_bounds(state);
    play_the_game_finalize_session(state);
    GameLoopCtx ctx;
    game_loop_front_menu_init(&ctx,state);
    for(int i=0;i<3;i++)check_key(state,&ctx,SDL_SCANCODE_DOWN);
    check_key(state,&ctx,SDL_SCANCODE_RETURN);
    dos_capture_frame("OPTIONS.BMP");
    for(int i=0;i<9;i++)check_key(state,&ctx,SDL_SCANCODE_DOWN);
    check_key(state,&ctx,SDL_SCANCODE_RETURN);
    check_key(state,&ctx,SDL_SCANCODE_UP);
    check_key(state,&ctx,SDL_SCANCODE_RETURN);
    dos_capture_frame("AUTOS.BMP");
    for(int i=0;i<4;i++)check_key(state,&ctx,SDL_SCANCODE_DOWN);
    check_key(state,&ctx,SDL_SCANCODE_RETURN);
    dos_capture_frame("LOAD.BMP");
    check_key(state,&ctx,SDL_SCANCODE_ESCAPE);
    check_key(state,&ctx,SDL_SCANCODE_ESCAPE);
    check_key(state,&ctx,SDL_SCANCODE_UP);
    check_key(state,&ctx,SDL_SCANCODE_RETURN);
    dos_capture_frame("NEWGAME.BMP");
    check_key(state,&ctx,SDL_SCANCODE_ESCAPE);
    /* The same menu in its in-game presentation, over the last world frame. */
    ctx.ingame_menu_frontend=0;
    check_key(state,&ctx,SDL_SCANCODE_DOWN);
    dos_capture_frame("PAUSE.BMP");
    printf("[DOSCHECK] main/options/autosaves/load/new-game/pause menus fit 320x200/240\n");
    display_dos_check_text_layouts();
#ifdef AB3D_NATIVE_RGB565
    display_release_title_memory();
    renderer_dos_check_gun_frames(state);
#endif
    printf("[DOSCHECK] native pixel SDL UI, numeric HUD and map passed\n");
}
void dos_memory_report(const char *phase)
{
    if(!strcmp(phase,"startup")) {
        int paragraphs=0;
        if(__dpmi_allocate_dos_memory(65535,&paragraphs)<0)
            printf("[DOSMEM] free conventional RAM after DPMI startup=%uKB\n",(unsigned)paragraphs/64u);
    }
    __dpmi_free_mem_info info;
    if(__dpmi_get_free_memory_information(&info)==0)
        printf("[DOSMEM] %s physical=%luKB free=%luKB largest-block=%luKB\n",phase,
            info.total_number_of_physical_pages*4,info.total_number_of_free_pages*4,
            info.largest_available_free_block_in_bytes/1024);
    if(strstr(phase,"failed")) {
        for(int i=0;i<256;i++)if(__djgpp_memory_handle_size[i])
            printf("[DOSHEAP] zone %d reserved=%uKB offset=%08x\n",i,
                __djgpp_memory_handle_size[i]/1024,__djgpp_memory_handle_list[i].address);
    }
}
static int compare_path(const char *a,const char *b)
{
    for(;;a++,b++) {
        int x=tolower((unsigned char)*a),y=tolower((unsigned char)*b);
        if(x=='\\')x='/';if(y=='\\')y='/';
        if(x!=y || !x)return x-y;
    }
}
const char *dos_asset_alias(const char *path)
{
    int low=0,high=(int)(sizeof(dos_assets)/sizeof(dos_assets[0]))-1;
    while(low<=high) {
        int mid=(low+high)/2,c=compare_path(path,dos_assets[mid].logical);
        if(!c)return dos_assets[mid].short_path;
        if(c<0)high=mid-1;else low=mid+1;
    }
    /* Preserve original handling of optional replacements and writable files. */
    if(!compare_path(path,"autosave1.bin"))return "AUTO1.BIN";
    if(!compare_path(path,"autosave2.bin"))return "AUTO2.BIN";
    if(!compare_path(path,"autosave3.bin"))return "AUTO3.BIN";
    if(!compare_path(path,"autosave4.bin"))return "AUTO4.BIN";
    return path;
}
void dos_platform_init(int argc,char **argv)
{
    SDL_SetHint("SDL_DOS_SKIP_EVENT_YIELD","1");
    const char *mouse=getenv("AB3D_DOS_MOUSE");
    SDL_SetHint("SDL_DOS_SKIP_MOUSE_PUMP",mouse && !strcmp(mouse,"1")?"0":"1");
    SDL_SetHint(SDL_HINT_DOS_ALLOW_DIRECT_FRAMEBUFFER,"1");
    SDL_SetHint(SDL_HINT_RENDER_DRIVER,"software");
    setenv("AB3D_RENDER_DRIVER","software",1);
    setenv("AB3D_DISABLE_GL_UNPACK","1",1);
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--dos-bench")) {
            char *end;
            long count = i+1<argc ? strtol(argv[++i],&end,10) : 0;
            if(count<1 || count>1024 || *end) {
                fprintf(stderr,"Use --dos-bench 1..1024 [--dos-check]\n");exit(1);
            }
            dos_benchmark_frames=(int)count;
        }
        else if(!strcmp(argv[i],"--dos-check"))dos_check_enabled=1;
        else if(!strcmp(argv[i],"--dos-check-frontend"))dos_frontend_check_enabled=1;
        else if(!strcmp(argv[i],"--dos-fixed-ticks"))dos_fixed_test_ticks=1;
    }
    if(dos_benchmark_frames<0 || dos_benchmark_frames>1024 || ((dos_check_enabled || dos_frontend_check_enabled) && !dos_benchmark_frames)) {
        fprintf(stderr,"Use --dos-bench 1..1024 [--dos-check]\n");exit(1);
    }
    printf("[DOS] 486 DX2/DX4 target; software VESA; i386 instructions / i486 tuning\n");
    if(dos_fixed_test_ticks)printf("[DOS] Fixed diagnostic ticks; FPS is not a real-game benchmark\n");
    dos_memory_report("startup");
}
void dos_profile_begin(int bucket)
{
    if(!dos_benchmark_frames)return;
    Uint64 now=SDL_GetPerformanceCounter();
    if(!depth && bucket==DOS_UPDATE)frame_start=now;
    if(depth)cost[stack[depth-1]]+=now-last;
    if(depth>=16){fprintf(stderr,"DOS profiler scope overflow\n");exit(1);}
    stack[depth++]=bucket;last=now;
}
void dos_profile_reset(void)
{
    depth=0;frames=0;frame_total=0;memset(cost,0,sizeof(cost));
}
void dos_profile_end(void)
{
    if(!dos_benchmark_frames)return;
    Uint64 now=SDL_GetPerformanceCounter();
    if(depth)cost[stack[--depth]]+=now-last;
    last=now;
}
void dos_profile_frame(void)
{
    if(!dos_benchmark_frames)return;
    Uint64 elapsed=SDL_GetPerformanceCounter()-frame_start;
    frame_times[frames++]=elapsed;frame_total+=elapsed;
    if((frames%16)==0 || frames==(unsigned)dos_benchmark_frames) {
        double unit=1000.0/SDL_GetPerformanceFrequency()/frames;
        double ms=frame_total*unit;
        Uint64 sorted[1024];memcpy(sorted,frame_times,frames*sizeof(Uint64));
        for(unsigned i=1;i<frames;i++) {
            Uint64 v=sorted[i];unsigned j=i;
            while(j && sorted[j-1]>v) { sorted[j]=sorted[j-1];j--; }
            sorted[j]=v;
        }
        printf("[DOSPROFILE] n=%u fps=%.2f frame=%.2f update=%.2f input=%.2f render=%.2f present=%.2f audio=%.2f ms\n",
            frames,1000.0/ms,ms,cost[DOS_UPDATE]*unit,cost[DOS_INPUT]*unit,
            cost[DOS_RENDER]*unit,cost[DOS_PRESENT]*unit,cost[DOS_AUDIO]*unit);
        double tick_ms=1000.0/SDL_GetPerformanceFrequency();
        printf("[DOSPACING] min=%.2f p95=%.2f max=%.2f ms\n",sorted[0]*tick_ms,
            sorted[(frames*95+99)/100-1]*tick_ms,sorted[frames-1]*tick_ms);
        fflush(stdout);
    }
}
