/* Run real enemy sound paths with the mixer output captured for assertions. */
#define audio_play_sample feedback_backend_play_sample
#include "../src/audio.c"
#undef audio_play_sample
static int played_sample, played_volume, played_count;
void audio_play_sample(int sample, int volume)
{
    played_sample=sample; played_volume=volume; played_count++;
}
#include "../src/objects.c"
#include <assert.h>
#undef printf
GameState g_state;

static void check_shotgun_soldier(void)
{
    /* FlameMarineAttackPLR1: timer <20, sample 21 at volume 200,
     * five pellets of damage 2, then ThirdTimer=200+(rand&255).
     * Apply that contract to both targets, independent of the rifle option. */
    static GameState state;
    static GameObject objects[3], shots[NASTY_SHOT_SLOT_COUNT];
    static uint8_t points[8];
    int cases=0;
    for(int player=1;player<=2;player++) {
        for(int option=0;option<=1;option++) {
            for(int far=0;far<=1;far++) {
                for(int ticks=1;ticks<=4;ticks+=3) {
                    memset(&state,0,sizeof(state));
                    memset(objects,0,sizeof(objects));
                    memset(shots,0,sizeof(shots));
                    for(int i=0;i<NASTY_SHOT_SLOT_COUNT;i++)
                        OBJ_SET_ZONE(&shots[i],-1);
                    state.level.object_data=(uint8_t*)objects;
                    state.level.object_points=points;
                    state.level.num_object_points=1;
                    state.level.plr1_obj=(uint8_t*)&objects[1];
                    state.level.plr2_obj=(uint8_t*)&objects[2];
                    state.level.nasty_shot_data=(uint8_t*)shots;
                    state.nasty=1;
                    state.temp_frames=ticks;
                    state.cfg_marine_hitscan_projectiles=option;
                    state.plr1.p_xoff=state.plr2.p_xoff=far?4000:0;
                    GameObject *obj=&objects[0];
                    OBJ_SET_CID(obj,0); OBJ_SET_ZONE(obj,0);
                    obj->obj.number=OBJ_NBR_FLAME_MARINE;
                    obj->obj.can_see=1<<(player-1);
                    NASTY_LIVES(*obj)=100;
                    OBJ_SET_TD_W(obj,ENEMY_SEC_TIMER_OFF,32767);
                    OBJ_SET_TD_W(obj,ENEMY_FOURTH_TIMER_OFF,20+ticks);
                    int before=played_count;
                    object_handle_marine(obj,&state);
                    assert(played_count==before); /* At 20: still winding up. */
                    assert(NASTY_DAMAGE(objects[1])==0 && NASTY_DAMAGE(objects[2])==0);
                    srand(1234);
                    object_handle_marine(obj,&state);
                    assert(played_count==before+1 && played_sample==21);
                    assert(played_volume==audio_distance_volume(200,far?4000:0,0));
                    assert(NASTY_DAMAGE(objects[player])==(far?0:10));
                    assert(NASTY_DAMAGE(objects[3-player])==0);
                    int cooldown=OBJ_TD_W(obj,ENEMY_THIRD_TIMER_OFF);
                    assert(cooldown>=200 && cooldown<=455);
                    object_handle_marine(obj,&state);
                    assert(played_count==before+1); /* No repeated flame burst. */
                    assert(OBJ_TD_W(obj,ENEMY_THIRD_TIMER_OFF)==cooldown-ticks);
                    for(int i=0;i<NASTY_SHOT_SLOT_COUNT;i++)
                        assert(OBJ_ZONE(&shots[i])==-1);
                    cases++;
                }
            }
        }
    }
    printf("PASS: %d shotgun soldier cases: both targets/options, near/far, tick batching, windup, damage, sound, cooldown, no projectiles\n",cases);
}

int main(void)
{
    math_tables_init();
    check_shotgun_soldier();
    static GameState state;
    static GameObject objects[2];
    static uint8_t points[16];
    state.level.object_data=(uint8_t*)objects;
    state.level.object_points=points;
    state.level.num_object_points=2;
    state.level.plr1_obj=(uint8_t*)&objects[1];
    state.level.plr2_obj=(uint8_t*)&objects[1];
    state.temp_frames=1;
    GameObject *obj=&objects[0];
    OBJ_SET_CID(obj,1); OBJ_SET_CID(&objects[1],-1);
    OBJ_SET_ZONE(obj,0);
    /* Emitter CID differs from its object slot; use ObjectPoints, not slot 0. */
    obj_sw(points+8,1600); obj_sw(points+12,-800);
    state.plr1.p_xoff=0; state.plr1.p_zoff=-800;
    state.plr2.p_xoff=1600; state.plr2.p_zoff=-800;
    const int types[]={OBJ_NBR_ALIEN,OBJ_NBR_ROBOT,OBJ_NBR_HUGE_RED_THING,
        OBJ_NBR_WORM,OBJ_NBR_FLAME_MARINE,OBJ_NBR_TOUGH_MARINE,OBJ_NBR_MARINE,
        OBJ_NBR_BIG_NASTY,OBJ_NBR_SMALL_RED_THING,OBJ_NBR_FLYING_NASTY,
        OBJ_NBR_EYEBALL,OBJ_NBR_TREE};
    for(int mode=0;mode<2;mode++) {
        state.mode=mode?MODE_SLAVE:MODE_SINGLE;
        int expected=mode?255:127; /* Noisevol=200 at distance 0 / 1600. */
        for(unsigned t=0;t<sizeof(types)/sizeof(types[0]);t++) {
            obj->obj.number=types[t];
            const EnemyParams *params=&enemy_params[obj_type_to_enemy_index(types[t])];
            NASTY_LIVES(*obj)=100;
            OBJ_SET_TD_W(obj,ENEMY_SEC_TIMER_OFF,32767);
            obj->raw[19]=(uint8_t)(1<<params->damage_shift);
            int before=played_count;
            assert(!enemy_check_damage(obj,params,&state));
            if(params->scream_sound>=0) {
                assert(played_count==before+1);
                assert(played_sample==params->scream_sound && played_volume==expected);
            }
            if(params->periodic_vocal1>=0 &&
                (params->hiss_timer_min>0 || params->hiss_timer_range>0)) {
                for(int attacking=0;attacking<2;attacking++) {
                    OBJ_SET_TD_W(obj,ENEMY_SEC_TIMER_OFF,0);
                    enemy_tick_sec_timer_vocal(obj,&state,params,attacking);
                    int noise=attacking?params->periodic_vol_attack:params->periodic_vol_idle;
                    assert(played_volume==audio_distance_volume(noise,mode?0:1600,0));
                    assert(played_sample>=params->periodic_vocal1 &&
                           played_sample<=(params->periodic_vocal2>=params->periodic_vocal1
                               ?params->periodic_vocal2:params->periodic_vocal1));
                }
            }
        }
        /* Marine death scream, without spawning debris. */
        obj->obj.number=OBJ_NBR_MARINE;
        enemy_death_marine_like(obj,&enemy_params[6],&state,1,false,0,false);
        assert(played_sample==0 && played_volume==expected);
        const int shooters[]={OBJ_NBR_MARINE,OBJ_NBR_FLAME_MARINE,
                              OBJ_NBR_TOUGH_MARINE,OBJ_NBR_ROBOT};
        state.nasty=1;
        for(unsigned i=0;i<sizeof(shooters)/sizeof(shooters[0]);i++) {
            memset(obj,0,sizeof(*obj));
            OBJ_SET_CID(obj,1); OBJ_SET_ZONE(obj,0);
            obj->obj.number=shooters[i]; obj->obj.can_see=1;
            NASTY_LIVES(*obj)=100;
            OBJ_SET_TD_W(obj,ENEMY_SEC_TIMER_OFF,32767);
            obj_sw(points+8,1600); obj_sw(points+12,-800);
            int before=played_count;
            if(shooters[i]==OBJ_NBR_ROBOT) object_handle_robot(obj,&state);
            else object_handle_marine(obj,&state);
            assert(played_count==before+1);
            int index=obj_type_to_enemy_index(shooters[i]);
            int sample=shooters[i]==OBJ_NBR_ROBOT?9:enemy_params[index].attack_sound;
            int noise=(shooters[i]==OBJ_NBR_MARINE ||
                       shooters[i]==OBJ_NBR_FLAME_MARINE)?200:100;
            const PlayerState *listener=mode?&state.plr2:&state.plr1;
            assert(played_sample==sample);
            assert(played_volume==audio_distance_volume(noise,
                obj_w(points+8)-listener->p_xoff,obj_w(points+12)-listener->p_zoff));
        }
        obj_sw(points+8,1600); obj_sw(points+12,-800);
        /* Gas emitter uses exactly the same listener and falloff path. */
        obj->obj.number=OBJ_NBR_GAS_PIPE;
        OBJ_SET_TD_W(obj,ENEMY_THIRD_TIMER_OFF,0);
        OBJ_SET_TD_W(obj,22,5); OBJ_SET_TD_W(obj,ENEMY_FOURTH_TIMER_OFF,0);
        object_handle_gas_pipe(obj,&state);
        assert(played_sample==22 && played_volume==expected);
    }
    int before=played_count;
    OBJ_SET_CID(obj,-1);
    object_play_noise(obj,&state,9,100);
    assert(played_count==before);
    puts("PASS: all 12 enemy types' pain/vocals, marine/robot attacks, marine death and fire spouts; both listeners and emitter CID lookup");
    return 0;
}
