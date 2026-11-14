/* Exercise actual shot selection after a turn/strafe with stale target caches.
 * Deterministic hit rolls isolate aim-cache behavior from random accuracy.
 * Audio, controller input and projection dimensions are fixed test inputs. */
#define rand feedback_test_rand
#include "../src/player.c"
#undef rand
#include <assert.h>
int feedback_test_rand(void) { return 32767; }
GameState g_state;
RendererState g_renderer;
int renderer_get_width(void) { return 320; }
int renderer_get_height(void) { return 200; }
void audio_play_sample(int id, int volume) { (void)id; (void)volume; }
static bool gamepad_firing;
bool input_gamepad_fire_held(void) { return gamepad_firing; }

/* ObjectMove.s room/exit geometry: two rooms sharing an open or solid edge.
 * The enemy centre is just beyond the edge; the reticle hits its near side
 * in the player's room. CanItBeSeen must use that impact room, not the centre.
 * Repeat with low-FPS TempFrames batches; these weapons have no flight step. */
static void check_portal_shot(int player, int gun, int frames, int closed, int mirror, int aim)
{
 static GameState state;
 static GameObject objects[2], shots[PLAYER_SHOT_SLOT_COUNT];
 static uint8_t points[(PLAYER_SHOT_SLOT_COUNT+1)*8];
 static uint8_t rooms[2*128], adds[2*4], lines[8*16];
 memset(&state,0,sizeof(state));memset(objects,0,sizeof(objects));
 memset(shots,0,sizeof(shots));memset(points,0,sizeof(points));
 memset(rooms,0,sizeof(rooms));memset(lines,0,sizeof(lines));
 state.level.data=rooms;state.level.data_byte_count=sizeof(rooms);
 state.level.zone_adds=adds;state.level.num_zones=state.level.num_zone_slots=2;
 state.level.floor_lines=lines;state.level.num_floor_lines=8;
 for(int zone=0;zone<2;zone++) {
  uint8_t *room=rooms+zone*128;
  obj_sl(adds+zone*4,zone*128);obj_sw(room,zone);
  obj_sl(room+2,128*256);obj_sl(room+6,-128*256);
  obj_sw(room+32,64);obj_sw(room+48,-1);
  int x0=(zone-1)*128,x1=zone*128;
  const int corners[4][2]={{x0,0},{x0,1024},{x1,1024},{x1,0}};
  for(int edge=0;edge<4;edge++) {
   uint8_t *line=lines+(zone*4+edge)*16;
   int dx=corners[(edge+1)%4][0]-corners[edge][0];
   int dz=corners[(edge+1)%4][1]-corners[edge][1];
   obj_sw(line,corners[edge][0]);obj_sw(line+2,corners[edge][1]);
   obj_sw(line+4,dx);obj_sw(line+6,dz);obj_sw(line+10,dx?128:1024);
   int portal=(zone==0 && edge==2)||(zone==1 && edge==0);
   obj_sw(line+8,portal && !closed?1-zone:-1);
   obj_sw(room+64+edge*2,zone*4+edge);
  }
  obj_sw(room+72,-1);obj_sw(room+74,-2);
 }
 state.cfg_mouse_look=true;state.cfg_y_proj_scale=100;state.temp_frames=frames;
 state.plr1_p_aim_speed=state.plr2_p_aim_speed=aim;
 state.level.object_data=(uint8_t*)objects;state.level.object_points=points;
 state.level.num_object_points=PLAYER_SHOT_SLOT_COUNT+1;
 state.level.player_shot_data=(uint8_t*)shots;
 OBJ_SET_CID(&objects[0],0);OBJ_SET_ZONE(&objects[0],1-mirror);
 objects[0].obj.number=OBJ_NBR_MARINE;NASTY_LIVES(objects[0])=80;
 OBJ_SET_CID(&objects[1],-1);obj_sw(points,mirror?-2:2);obj_sw(points+4,512);
 for(int i=0;i<PLAYER_SHOT_SLOT_COUNT;i++) {OBJ_SET_CID(&shots[i],i+1);OBJ_SET_ZONE(&shots[i],-1);}
 PlayerState *p=player==1?&state.plr1:&state.plr2;
 p->xoff=(mirror?4:-4)*65536;p->zoff=32*65536;p->cosval=cos_lookup(0);p->sinval=sin_lookup(0);
 p->zone=mirror;p->roompt=mirror*128;
 p->gun_selected=gun;p->gun_data[gun].visible=-1;p->gun_data[gun].ammo=160;
 p->p_fire=p->p_clicked=-1;gamepad_firing=false;
 uint8_t before[sizeof(lines)];memcpy(before,lines,sizeof(lines));
 assert(hitscan_target_has_line_of_sight(&state,p,rooms+mirror*128,
          mirror?4:-4,32,0,&objects[0],mirror?4:-4,512,
          player_mouse_look_ray_y_delta_at_dist(aim,960)>>7)==!closed);
 assert(!memcmp(before,lines,sizeof(lines)));
 if(player==1)player1_shoot(&state);else player2_shoot(&state);
 int hit=!closed && aim>=-1024 && aim<=1024;
 if(objects[0].raw[19]!=(hit?(gun==7?28:4):0))
  fprintf(stderr,"portal shot: player=%d gun=%d frames=%d closed=%d damage=%d inline=%d distance=%d\n",
          player,gun,frames,closed,objects[0].raw[19],plr1_obs_in_line[0],plr1_obj_dists[0]);
 assert(objects[0].raw[19]==(hit?(gun==7?28:4):0));
 assert(p->gun_data[gun].ammo==152);
}
static void check_shot(int player, int gun, int mouse_look, int old_angle, int old_x, int new_angle, int new_x, int expected) {
 static GameState state;
 static GameObject objects[2], shots[PLAYER_SHOT_SLOT_COUNT];
 static uint8_t points[(PLAYER_SHOT_SLOT_COUNT+1)*8];
 memset(&state,0,sizeof(state));memset(objects,0,sizeof(objects));memset(shots,0,sizeof(shots));memset(points,0,sizeof(points));
 state.cfg_mouse_look=mouse_look;
 state.cfg_y_proj_scale=100;
 gamepad_firing=(gun==1);
 state.level.object_data=(uint8_t*)objects;
 state.level.object_points=points;
 state.level.num_object_points=PLAYER_SHOT_SLOT_COUNT+1;
 state.level.player_shot_data=(uint8_t*)shots;
 GameObject *worm=&objects[0];
 OBJ_SET_CID(worm,0);OBJ_SET_ZONE(worm,0);
 worm->obj.number=OBJ_NBR_WORM;worm->obj.can_see=3;
 NASTY_LIVES(*worm)=80; /* Original level_o/twolev.bin worms. */
 if(gun==1)obj_sw(worm->raw+4,12);
 OBJ_SET_CID(&objects[1],-1);
 obj_sw(points,0);obj_sw(points+4,200);
 for(int i=0;i<PLAYER_SHOT_SLOT_COUNT;i++) {OBJ_SET_CID(&shots[i],i+1);OBJ_SET_ZONE(&shots[i],-1);}
 PlayerState *p=player==1?&state.plr1:&state.plr2;
 p->gun_selected=gun;p->gun_data[gun].visible=-1;p->gun_data[gun].ammo=160;p->p_fire=-1;p->p_clicked=-1;
 p->sinval=sin_lookup(old_angle);p->cosval=cos_lookup(old_angle);p->xoff=old_x*65536;
 if(player==1)calc_plr1_in_line(&state);else calc_plr2_in_line(&state);
 p->sinval=sin_lookup(new_angle);p->cosval=cos_lookup(new_angle);p->xoff=new_x*65536;
 if(player==1)player1_shoot(&state);else player2_shoot(&state);
 if(gun==1) {
  assert(SHOT_SIZE(shots[0])==1 && SHOT_POWER(shots[0])==16);
  assert(SHOT_YVEL(shots[0])==(expected?128:0));
 } else {
  assert(worm->raw[19]==(expected?(gun==7?28:4):0));
 }
 assert(p->gun_data[gun].ammo==152);
}
int main(void) {
 math_tables_init();
 for(int player=1;player<=2;player++) {
  const int guns[]={0,7,1};
  for(int mode=0;mode<2;mode++)for(int g=0;g<3;g++) {
   int gun=guns[g];
   if(gun==1 && !mode)continue;
   check_shot(player,gun,mode,ANGLE_90,0,0,0,1);
   check_shot(player,gun,mode,0,0,ANGLE_90,0,0);
   check_shot(player,gun,mode,0,400,0,0,1);
   check_shot(player,gun,mode,0,0,0,400,0);
  }
 }
 const int batches[]={1,2,3,4,8,16};
 const int aims[]={-1536,-1024,0,1024,1536};
 for(int player=1;player<=2;player++)for(int gun=0;gun<=7;gun+=7)
  for(int b=0;b<6;b++)for(int closed=0;closed<2;closed++)for(int mirror=0;mirror<2;mirror++)
   for(int a=0;a<5;a++)check_portal_shot(player,gun,batches[b],closed,mirror,aims[a]);
 puts("PASS: 40 current-aim/position cases, both players, mouse/fixed-view pistol/shotgun and controller plasma aim");
 puts("PASS: 144 cross-zone hits, 240 solid-wall and 96 vertical-aim rejections; both players and instant guns, TempFrames 1/2/3/4/8/16");
 return 0;
}
