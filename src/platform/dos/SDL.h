/* SDL2-facing transport adapter, derived from GloomAmiga src/sdl_dos_compat.h. */
#ifndef AB3D_SDL_DOS_COMPAT_H
#define AB3D_SDL_DOS_COMPAT_H

#define SDL_DISABLE_OLD_NAMES 1
#include <SDL3/SDL.h>
#ifdef AB3D_DOS_INDEXED
#include "palette_dos.h"
#endif
#include <SDL3/SDL_main.h>
#include <sys/nearptr.h>
#include "ui_dos.h"
#define SDL_Renderer DosUiRenderer
#define SDL_Texture DosUiImage

extern int __djgpp_nearptr_enable(void);

#define SDL_INIT_GAMECONTROLLER SDL_INIT_GAMEPAD

#define SDL_QUIT SDL_EVENT_QUIT
#define SDL_KEYDOWN SDL_EVENT_KEY_DOWN
#define SDL_MOUSEMOTION SDL_EVENT_MOUSE_MOTION
#define SDL_MOUSEBUTTONDOWN SDL_EVENT_MOUSE_BUTTON_DOWN
#define SDL_MOUSEBUTTONUP SDL_EVENT_MOUSE_BUTTON_UP
#define SDL_CONTROLLERBUTTONDOWN SDL_EVENT_GAMEPAD_BUTTON_DOWN
#define SDL_CONTROLLERDEVICEADDED SDL_EVENT_GAMEPAD_ADDED
#define SDL_CONTROLLERDEVICEREMOVED SDL_EVENT_GAMEPAD_REMOVED

#define SDL_BUTTON(X) SDL_BUTTON_MASK(X)
#define SDL_bool bool
#define SDL_CONTROLLER_BUTTON_MAX SDL_GAMEPAD_BUTTON_COUNT
#define SDL_QUERY -1
#define SDL_ENABLE 1
#define SDL_DISABLE 0
#define SDL_TRUE true
#define SDL_FALSE false
#define AUDIO_F32SYS SDL_AUDIO_F32
#define SDL_RENDERER_ACCELERATED 0u
#define SDL_RENDERER_PRESENTVSYNC 0u
#define SDL_RENDERER_SOFTWARE 0u
#define SDL_WINDOW_FULLSCREEN_DESKTOP SDL_WINDOW_FULLSCREEN
#define SDL_ScaleModeNearest SDL_SCALEMODE_NEAREST
#define SDLK_p SDLK_P
#define SDLK_s SDLK_S
#define SDLK_w SDLK_W

#define SDL_CONTROLLER_BUTTON_A SDL_GAMEPAD_BUTTON_SOUTH
#define SDL_CONTROLLER_BUTTON_B SDL_GAMEPAD_BUTTON_EAST
#define SDL_CONTROLLER_BUTTON_START SDL_GAMEPAD_BUTTON_START
#define SDL_CONTROLLER_BUTTON_BACK SDL_GAMEPAD_BUTTON_BACK
#define SDL_CONTROLLER_BUTTON_DPAD_UP SDL_GAMEPAD_BUTTON_DPAD_UP
#define SDL_CONTROLLER_BUTTON_DPAD_DOWN SDL_GAMEPAD_BUTTON_DPAD_DOWN
#define SDL_CONTROLLER_BUTTON_DPAD_LEFT SDL_GAMEPAD_BUTTON_DPAD_LEFT
#define SDL_CONTROLLER_BUTTON_DPAD_RIGHT SDL_GAMEPAD_BUTTON_DPAD_RIGHT
#define SDL_CONTROLLER_BUTTON_RIGHTSHOULDER SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER
#define SDL_CONTROLLER_AXIS_LEFTX SDL_GAMEPAD_AXIS_LEFTX
#define SDL_CONTROLLER_AXIS_LEFTY SDL_GAMEPAD_AXIS_LEFTY
#define SDL_CONTROLLER_AXIS_RIGHTX SDL_GAMEPAD_AXIS_RIGHTX
#define SDL_CONTROLLER_AXIS_TRIGGERRIGHT SDL_GAMEPAD_AXIS_RIGHT_TRIGGER
#define SDL_GameController SDL_Gamepad
#define SDL_GameControllerButton SDL_GamepadButton

static SDL_Window *ab3d_dos_window = NULL;

static inline int Ab3dDos_Init(SDL_InitFlags flags) {
  SDL_SetMainReady();
  if (__djgpp_nearptr_enable() == 0) {
    SDL_SetError("DOSVESA: __djgpp_nearptr_enable() failed");
    return -1;
  }
  return SDL_Init(flags) ? 0 : -1;
}

typedef struct {
  SDL_Keycode sym;
  SDL_Scancode scancode;
} Ab3dDosKeySym;

typedef struct {
  SDL_EventType type;
  Uint32 reserved;
  Uint64 timestamp;
  SDL_WindowID windowID;
  SDL_KeyboardID which;
  SDL_Scancode scancode;
  Ab3dDosKeySym keysym;
  SDL_Keymod mod;
  Uint16 raw;
  bool down;
  bool repeat;
} Ab3dDosKeyboardEvent;

typedef union Ab3dDosEvent {
  SDL_EventType type;
  SDL_CommonEvent common;
  SDL_DisplayEvent display;
  struct { SDL_EventType type; Uint32 reserved; Uint64 timestamp; SDL_WindowID windowID; Uint8 event; int data1, data2; } window;
  Ab3dDosKeyboardEvent key;
  SDL_TextEditingEvent edit;
  SDL_TextInputEvent text;
  SDL_MouseMotionEvent motion;
  SDL_MouseButtonEvent button;
  SDL_MouseWheelEvent wheel;
  SDL_JoyAxisEvent jaxis;
  SDL_JoyBallEvent jball;
  SDL_JoyHatEvent jhat;
  SDL_JoyButtonEvent jbutton;
  SDL_JoyDeviceEvent jdevice;
  SDL_GamepadAxisEvent gaxis;
  SDL_GamepadButtonEvent cbutton;
  SDL_GamepadDeviceEvent cdevice;
  SDL_QuitEvent quit;
  SDL_UserEvent user;
  SDL_TouchFingerEvent tfinger;
  SDL_PenProximityEvent pproximity;
  SDL_PenTouchEvent ptouch;
  SDL_PenMotionEvent pmotion;
  SDL_PenButtonEvent pbutton;
  SDL_PenAxisEvent paxis;
  SDL_RenderEvent render;
  SDL_DropEvent drop;
  Uint8 padding[sizeof(SDL_Event)];
} Ab3dDosEvent;

static inline int Ab3dDos_PollEvent(Ab3dDosEvent *event) {
  SDL_Event raw;
  if (!SDL_PollEvent(&raw)) {
    return 0;
  }
  SDL_memset(event, 0, sizeof(*event));
  SDL_memcpy(event, &raw, sizeof(raw));
  if (raw.type == SDL_EVENT_KEY_DOWN || raw.type == SDL_EVENT_KEY_UP) {
    event->key.keysym.sym = raw.key.key;
    event->key.keysym.scancode = raw.key.scancode;
    event->key.scancode = raw.key.scancode;
    event->key.mod = raw.key.mod;
    event->key.raw = raw.key.raw;
    event->key.down = raw.key.down;
    event->key.repeat = raw.key.repeat;
  }
  if (raw.type >= SDL_EVENT_WINDOW_FIRST && raw.type <= SDL_EVENT_WINDOW_LAST) {
    event->type = 0x200;
    event->window.event = raw.type == SDL_EVENT_WINDOW_RESIZED ? 5 :
                          raw.type == SDL_EVENT_WINDOW_FOCUS_LOST ? 13 : 0;
    event->window.data1 = raw.window.data1;
    event->window.data2 = raw.window.data2;
  }
  return 1;
}

/* Diagnostic input travels through the same SDL event translation as IRQ input. */
static inline bool Ab3dDos_PushKey(SDL_Scancode code,bool down) {
  SDL_Event event;
  SDL_zero(event);
  event.type=down?SDL_EVENT_KEY_DOWN:SDL_EVENT_KEY_UP;
  event.key.scancode=code;
  event.key.key=SDL_GetKeyFromScancode(code,SDL_KMOD_NONE,false);
  event.key.down=down;
  return SDL_PushEvent(&event);
}


static inline SDL_Renderer *Ab3dDos_CreateRendererPtr(SDL_Window *window, int index, Uint32 flags) {
  (void)index;
  (void)flags;
  return dos_ui_create_renderer(window);
}

static inline SDL_Window *Ab3dDos_CreateWindow(const char *title, int x, int y, int w, int h, Uint32 flags) {
  (void)x;
  (void)y;
  ab3d_dos_window = SDL_CreateWindow(title, w, h, flags);
  if (ab3d_dos_window) {
    int count = 0;
    SDL_DisplayMode **modes = SDL_GetFullscreenDisplayModes(SDL_GetPrimaryDisplay(), &count);
    const SDL_DisplayMode *best = NULL;
    for (int i = 0; modes && i < count; i++) {
      const SDL_DisplayMode *m = modes[i];
      if (m->format ==
#ifdef AB3D_DOS_INDEXED
          SDL_PIXELFORMAT_INDEX8
#else
          SDL_PIXELFORMAT_RGB565
#endif
          && m->w >= w && m->h >= h &&
          (!best || m->w * m->h < best->w * best->h)) best = m;
    }
    bool ok = best && SDL_SetWindowFullscreenMode(ab3d_dos_window, best) &&
                      SDL_SetWindowFullscreen(ab3d_dos_window, true);
    SDL_free(modes);
    if (!ok) {
      SDL_DestroyWindow(ab3d_dos_window);
      ab3d_dos_window = NULL;
      SDL_SetError("DOS requires a native %s mode of at least 320x200",
#ifdef AB3D_DOS_INDEXED
                   "INDEX8"
#else
                   "RGB565"
#endif
                   );
    }
  }
#ifdef AB3D_DOS_INDEXED
  if(ab3d_dos_window)dos_palette_install(ab3d_dos_window);
#endif
  return ab3d_dos_window;
}



static inline int Ab3dDos_RenderCopy(SDL_Renderer *renderer, SDL_Texture *texture, const SDL_Rect *src,
                                      const SDL_Rect *dst) {
  return dos_ui_blit(renderer, texture, src, dst);
}

/* UI surfaces are native RGB565; uploads accept the shared ARGB asset decoder. */
static inline int Ab3dDos_LockTexture(SDL_Texture *t,const SDL_Rect *rect,void **pixels,int *pitch) {
  if(!t || !t->normal || !SDL_LockSurface(t->normal))return -1;
  SDL_DestroySurface(t->grey);t->grey=NULL;
  *pixels=(Uint8 *)t->normal->pixels+(rect?rect->y*t->normal->pitch+rect->x*2:0);
  *pitch=t->normal->pitch;return 0;
}
static inline void Ab3dDos_UnlockTexture(SDL_Texture *t) { if(t)SDL_UnlockSurface(t->normal); }
static inline int Ab3dDos_UpdateTexture(SDL_Texture *t,const SDL_Rect *rect,const void *pixels,int pitch) {
  dos_ui_texture_upload(t,rect,pixels,pitch);return 0;
}
static inline void Ab3dDos_DestroyTexture(SDL_Texture *t) { dos_ui_texture_destroy(t); }
static inline int Ab3dDos_SetTextureScaleMode(SDL_Texture *t,SDL_ScaleMode mode) {
  if(!t || mode!=SDL_SCALEMODE_NEAREST){SDL_SetError("DOS UI uses native pixel sizes");return -1;}return 0;
}
static inline int Ab3dDos_SetTextureBlendMode(SDL_Texture *t,SDL_BlendMode mode) {
  if(!t || (mode!=SDL_BLENDMODE_NONE && mode!=SDL_BLENDMODE_BLEND))return -1;
  /* DOS image alpha is an on/off cutout mask, as requested. */
  return SDL_SetSurfaceBlendMode(t->normal,SDL_BLENDMODE_NONE)?0:-1;
}
static inline int Ab3dDos_SetTextureAlphaMod(SDL_Texture *t,Uint8 alpha) { if(!t)return -1;t->alpha=alpha;return 0; }
static inline int Ab3dDos_GetTextureAlphaMod(SDL_Texture *t,Uint8 *alpha) { if(!t)return -1;if(alpha)*alpha=t->alpha;return 0; }
static inline int Ab3dDos_GetTextureColorMod(SDL_Texture *t,Uint8 *r,Uint8 *g,Uint8 *b) {
  if(!t)return -1;if(r)*r=t->red;if(g)*g=t->green;if(b)*b=t->blue;return 0;
}
static inline int Ab3dDos_SetTextureColorMod(SDL_Texture *t,Uint8 r,Uint8 g,Uint8 b) {
  if(!t)return -1;t->red=r;t->green=g;t->blue=b;return 0;
}
static inline int Ab3dDos_SetRenderDrawBlendMode(SDL_Renderer *r,SDL_BlendMode mode) {
  return r && (mode==SDL_BLENDMODE_NONE || mode==SDL_BLENDMODE_BLEND)?0:-1;
}
/* All surface drawing completes immediately; there is no queued work to flush. */
static inline bool Ab3dDos_FlushRenderer(SDL_Renderer *r) { return r && r->surface; }
static inline int Ab3dDos_RenderPoint(SDL_Renderer *r,float x,float y) {
  SDL_Rect point={(int)x,(int)y,1,1};return dos_ui_fill(r,&point);
}
static inline bool Ab3dDos_GetRenderClipRect(SDL_Renderer *r,SDL_Rect *rect) {
  if(!r || !rect)return false;*rect=r->clipped?r->clip:(SDL_Rect){0,0,0,0};return true;
}
static inline bool Ab3dDos_RenderClipEnabled(SDL_Renderer *r) { return r && r->clipped; }
static inline int Ab3dDos_RenderFillRect(SDL_Renderer *r,const SDL_Rect *rect) { return dos_ui_fill(r,rect); }
static inline int Ab3dDos_RenderDrawRect(SDL_Renderer *r,const SDL_Rect *rect) { return dos_ui_outline(r,rect); }
static inline int Ab3dDos_RenderDrawLine(SDL_Renderer *r,int x0,int y0,int x1,int y1) { return dos_ui_line(r,x0,y0,x1,y1); }

static inline int Ab3dDos_SetRelativeMouseMode(bool enabled) {
  SDL_Window *window = ab3d_dos_window != NULL ? ab3d_dos_window : SDL_GetKeyboardFocus();
  if (window == NULL) {
    SDL_SetError("No DOS SDL window for relative mouse mode");
    return -1;
  }
  return SDL_SetWindowRelativeMouseMode(window, enabled) ? 0 : -1;
}

static inline bool Ab3dDos_GetRelativeMouseMode(void) {
  SDL_Window *window = ab3d_dos_window != NULL ? ab3d_dos_window : SDL_GetKeyboardFocus();
  return window != NULL && SDL_GetWindowRelativeMouseMode(window);
}

static inline int Ab3dDos_CaptureMouse(bool enabled) {
  return SDL_CaptureMouse(enabled) ? 0 : -1;
}

static inline void Ab3dDos_SetWindowGrab(SDL_Window *window, bool grabbed) {
  (void)SDL_SetWindowMouseGrab(window, grabbed);
}

static inline int Ab3dDos_ShowCursor(int toggle) {
  if (toggle == SDL_ENABLE) {
    return SDL_ShowCursor() ? SDL_ENABLE : SDL_DISABLE;
  }
  if (toggle == SDL_DISABLE) {
    return SDL_HideCursor() ? SDL_DISABLE : SDL_ENABLE;
  }
  return SDL_CursorVisible() ? SDL_ENABLE : SDL_DISABLE;
}

static inline Uint32 Ab3dDos_GetMouseState(int *x, int *y) {
  float fx = 0.0f;
  float fy = 0.0f;
  Uint32 buttons = SDL_GetMouseState(&fx, &fy);
  if (x != NULL) *x = (int)fx;
  if (y != NULL) *y = (int)fy;
  return buttons;
}

static inline int Ab3dDos_SetWindowFullscreen(SDL_Window *window, Uint32 flags) {
  /* DOS has one hardware display. Keep the explicitly selected RGB565 mode
   * when desktop code toggles its window/fullscreen preference. */
  (void)flags;
  return SDL_SetWindowFullscreen(window, true) ? 0 : -1;
}

static inline char *Ab3dDos_GetBasePath(void) {
  const char *path = SDL_GetBasePath();
  return path ? SDL_strdup(path) : NULL;
}

static inline void Ab3dDos_free(void *ptr) {
  SDL_free(ptr);
}

static inline const Uint8 *Ab3dDos_GetKeyboardState(int *numkeys) {
  return (const Uint8 *)SDL_GetKeyboardState(numkeys);
}

static inline int Ab3dDos_GetCurrentDisplayMode(SDL_DisplayID displayID, SDL_DisplayMode *mode) {
  const SDL_DisplayMode *current = SDL_GetCurrentDisplayMode(displayID);
  if (current == NULL || mode == NULL) {
    return -1;
  }
  *mode = *current;
  return 0;
}

static inline int Ab3dDos_NumJoysticks(void) {
  int count = 0;
  SDL_JoystickID *ids = SDL_GetJoysticks(&count);
  SDL_free(ids);
  return count;
}

static inline bool Ab3dDos_IsGameController(int device_index) {
  int count = 0;
  bool result = false;
  SDL_JoystickID *ids = SDL_GetJoysticks(&count);
  if (ids != NULL && device_index >= 0 && device_index < count) {
    result = SDL_IsGamepad(ids[device_index]);
  }
  SDL_free(ids);
  return result;
}

static inline SDL_JoystickID Ab3dDos_JoystickGetDeviceInstanceID(int device_index) {
  int count = 0;
  SDL_JoystickID result = 0;
  SDL_JoystickID *ids = SDL_GetJoysticks(&count);
  if (ids != NULL && device_index >= 0 && device_index < count) {
    result = ids[device_index];
  }
  SDL_free(ids);
  return result;
}

static inline SDL_Gamepad *Ab3dDos_GameControllerOpen(int device_index) {
  SDL_JoystickID id = Ab3dDos_JoystickGetDeviceInstanceID(device_index);
  return id != 0 ? SDL_OpenGamepad(id) : NULL;
}


#undef SDL_VERSION_ATLEAST
#define SDL_VERSION_ATLEAST(a,b,c) 0
#define SDL_WINDOW_SHOWN 0
#define SDL_WINDOWEVENT 0x200
#define SDL_WINDOWEVENT_SIZE_CHANGED 6
#define SDL_JOYBUTTONDOWN SDL_EVENT_JOYSTICK_BUTTON_DOWN
#define SDL_FINGERDOWN SDL_EVENT_FINGER_DOWN
#define SDL_WINDOWEVENT_RESIZED 5
#define SDL_WINDOWEVENT_FOCUS_LOST 13
#define SDL_MOUSEWHEEL SDL_EVENT_MOUSE_WHEEL
#define SDL_KEYUP SDL_EVENT_KEY_UP
#define SDL_HINT_RENDER_SCALE_QUALITY "SDL_RENDER_SCALE_QUALITY"
#define SDL_HINT_RENDER_DRIVER "SDL_RENDER_DRIVER"
#define SDL_CONTROLLER_AXIS_RIGHTY SDL_GAMEPAD_AXIS_RIGHTY
#define SDL_CONTROLLER_AXIS_TRIGGERLEFT SDL_GAMEPAD_AXIS_LEFT_TRIGGER
#define SDL_CONTROLLER_BUTTON_LEFTSHOULDER SDL_GAMEPAD_BUTTON_LEFT_SHOULDER
#define SDL_CONTROLLER_BUTTON_RIGHTSTICK SDL_GAMEPAD_BUTTON_RIGHT_STICK
#define SDL_CONTROLLER_BUTTON_LEFTSTICK SDL_GAMEPAD_BUTTON_LEFT_STICK
#define SDL_CONTROLLER_BUTTON_X SDL_GAMEPAD_BUTTON_WEST
#define SDL_CONTROLLER_BUTTON_Y SDL_GAMEPAD_BUTTON_NORTH
#define SDL_GameControllerAxis SDL_GamepadAxis
#define SDL_mutex SDL_Mutex
#define SDL_atomic_t SDL_AtomicInt
#define SDL_AtomicLock SDL_LockSpinlock
#define SDL_AtomicUnlock SDL_UnlockSpinlock
#define SDL_RendererInfo Ab3dDosRendererInfo
#define SDL_ScaleModeLinear SDL_SCALEMODE_LINEAR
#define SDL_WINDOW_ALLOW_HIGHDPI SDL_WINDOW_HIGH_PIXEL_DENSITY

typedef struct { const char *name; } Ab3dDosRendererInfo;
static inline int Ab3dDos_GetRendererInfo(SDL_Renderer *r,Ab3dDosRendererInfo *info) {
  if(!r || !info)return -1;info->name="native SDL surface";return 0;
}
static inline int Ab3dDos_GetRendererOutputSize(SDL_Renderer *r,int *w,int *h) {
  if(!r)return -1;if(w)*w=r->surface->w;if(h)*h=r->surface->h;return 0;
}
static inline int Ab3dDos_QueryTexture(SDL_Texture *t,Uint32 *format,int *access,int *w,int *h) {
  if(!t)return -1;if(w)*w=t->width;if(h)*h=t->height;
  if(format)*format=SDL_PIXELFORMAT_RGB565;if(access)*access=t->access;return 0;
}
static inline int Ab3dDos_RenderSetViewport(SDL_Renderer *r,const SDL_Rect *v) { return dos_ui_viewport(r,v)?0:-1; }
static inline int Ab3dDos_GetDisplayMode(int unused,SDL_DisplayMode *out) {
  (void)unused; return Ab3dDos_GetCurrentDisplayMode(SDL_GetPrimaryDisplay(),out);
}
static inline int Ab3dDos_GetDisplayBounds(int unused,SDL_Rect *r) {
  (void)unused; return SDL_GetDisplayBounds(SDL_GetPrimaryDisplay(),r) ? 0 : -1;
}
static inline int Ab3dDos_TryLockMutex(SDL_Mutex *m) { return SDL_TryLockMutex(m) ? 0 : 1; }
static inline int Ab3dDos_InitSubSystem(Uint32 f) { return SDL_InitSubSystem(f) ? 0 : -1; }
static inline int Ab3dDos_ControllerEvents(int toggle) {
  if(toggle!=SDL_QUERY) SDL_SetGamepadEventsEnabled(toggle==SDL_ENABLE);
  return SDL_GamepadEventsEnabled() ? SDL_ENABLE : SDL_DISABLE;
}
static inline uintptr_t Ab3dDos_TLSCreate(void) {
  return (uintptr_t)SDL_calloc(1,sizeof(SDL_TLSID));
}
static inline void *Ab3dDos_TLSGet(uintptr_t id) { return SDL_GetTLS((SDL_TLSID *)id); }
static inline int Ab3dDos_TLSSet(uintptr_t id,const void *v,void (*d)(void *)) {
  return SDL_SetTLS((SDL_TLSID *)id,v,d) ? 0 : -1;
}
#define SDL_TLSID uintptr_t
#define SDL_TLSCreate Ab3dDos_TLSCreate
#define SDL_TLSGet Ab3dDos_TLSGet
#define SDL_TLSSet Ab3dDos_TLSSet
#define SDL_InitSubSystem Ab3dDos_InitSubSystem
#define SDL_GameControllerEventState Ab3dDos_ControllerEvents
#define SDL_GetRendererInfo Ab3dDos_GetRendererInfo
#define SDL_GetRendererOutputSize Ab3dDos_GetRendererOutputSize
#define SDL_QueryTexture Ab3dDos_QueryTexture
#define SDL_RenderSetViewport Ab3dDos_RenderSetViewport
#define SDL_RenderDrawPoint SDL_RenderPoint
#define SDL_RenderFlush SDL_FlushRenderer
#define SDL_GetDesktopDisplayMode Ab3dDos_GetDisplayMode
#undef SDL_GetCurrentDisplayMode
#define SDL_GetCurrentDisplayMode Ab3dDos_GetDisplayMode
#define SDL_GetDisplayBounds Ab3dDos_GetDisplayBounds
#define SDL_GL_GetDrawableSize SDL_GetWindowSizeInPixels
#define SDL_TryLockMutex Ab3dDos_TryLockMutex
#define SDL_getenv SDL_getenv_unsafe

static inline SDL_Texture *Ab3dDos_CreateTexture(SDL_Renderer *r,SDL_PixelFormat format,SDL_TextureAccess access,int w,int h) {
  if(!r || format!=SDL_PIXELFORMAT_ARGB8888){SDL_SetError("DOS UI expects ARGB8888 uploads");return NULL;}
  return dos_ui_create_image(w,h,access);
}
#define SDL_DestroyRenderer dos_ui_destroy_renderer
#define SDL_SetRenderDrawColor dos_ui_color
#define SDL_SetRenderDrawBlendMode Ab3dDos_SetRenderDrawBlendMode
#define SDL_RenderClear dos_ui_clear
#define SDL_RenderPresent dos_ui_present
#define SDL_FlushRenderer Ab3dDos_FlushRenderer
#define SDL_RenderPoint Ab3dDos_RenderPoint
#define SDL_RenderReadPixels dos_ui_readback
#define SDL_RenderGeometry dos_ui_geometry
#define SDL_SetRenderClipRect dos_ui_clip
#define SDL_GetRenderClipRect Ab3dDos_GetRenderClipRect
#define SDL_RenderClipEnabled Ab3dDos_RenderClipEnabled
#define SDL_UnlockTexture Ab3dDos_UnlockTexture
#define SDL_SetTextureAlphaMod Ab3dDos_SetTextureAlphaMod
#define SDL_GetTextureAlphaMod Ab3dDos_GetTextureAlphaMod
#define SDL_GetTextureColorMod Ab3dDos_GetTextureColorMod
#define SDL_SetTextureColorMod Ab3dDos_SetTextureColorMod
#define SDL_CreateTexture Ab3dDos_CreateTexture
#define SDL_Init Ab3dDos_Init
#define SDL_PollEvent Ab3dDos_PollEvent
#define SDL_CreateWindow Ab3dDos_CreateWindow
#define SDL_CreateRenderer Ab3dDos_CreateRendererPtr
#define SDL_RenderCopy Ab3dDos_RenderCopy
#define SDL_RenderFillRect Ab3dDos_RenderFillRect
#define SDL_RenderDrawRect Ab3dDos_RenderDrawRect
#define SDL_RenderDrawLine Ab3dDos_RenderDrawLine
#define SDL_LockTexture Ab3dDos_LockTexture
#define SDL_UpdateTexture Ab3dDos_UpdateTexture
#define SDL_DestroyTexture Ab3dDos_DestroyTexture
#define SDL_SetTextureScaleMode Ab3dDos_SetTextureScaleMode
#define SDL_SetTextureBlendMode Ab3dDos_SetTextureBlendMode
#define SDL_SetRelativeMouseMode Ab3dDos_SetRelativeMouseMode
#define SDL_GetRelativeMouseMode Ab3dDos_GetRelativeMouseMode
#define SDL_CaptureMouse Ab3dDos_CaptureMouse
#define SDL_SetWindowGrab Ab3dDos_SetWindowGrab
#define SDL_ShowCursor Ab3dDos_ShowCursor
#define SDL_GetMouseState Ab3dDos_GetMouseState
#define SDL_SetWindowFullscreen Ab3dDos_SetWindowFullscreen
#define SDL_GetBasePath Ab3dDos_GetBasePath
#define SDL_free Ab3dDos_free
#define SDL_GetKeyboardState Ab3dDos_GetKeyboardState
#define SDL_NumJoysticks Ab3dDos_NumJoysticks
#define SDL_IsGameController Ab3dDos_IsGameController
#define SDL_JoystickGetDeviceInstanceID Ab3dDos_JoystickGetDeviceInstanceID
#define SDL_GameControllerOpen Ab3dDos_GameControllerOpen
#define SDL_GameControllerName SDL_GetGamepadName
#define SDL_GameControllerClose SDL_CloseGamepad
#define SDL_GameControllerGetJoystick SDL_GetGamepadJoystick
#define SDL_JoystickInstanceID SDL_GetJoystickID
#define SDL_GameControllerGetButton SDL_GetGamepadButton
#define SDL_GameControllerGetAxis SDL_GetGamepadAxis
#define SDL_Event Ab3dDosEvent

#endif
