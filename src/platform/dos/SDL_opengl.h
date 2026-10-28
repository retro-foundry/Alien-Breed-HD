/* GL declarations are shared; DOS never creates a GL context. */
#include <SDL3/SDL_opengl.h>

#define DGL_TEXTURE_2D GL_TEXTURE_2D
#define DGL_TEXTURE_2D_ARRAY GL_TEXTURE_2D_ARRAY
#define DGL_TEXTURE_MIN_FILTER GL_TEXTURE_MIN_FILTER
#define DGL_TEXTURE_MAG_FILTER GL_TEXTURE_MAG_FILTER
#define DGL_TEXTURE_WRAP_S GL_TEXTURE_WRAP_S
#define DGL_TEXTURE_WRAP_T GL_TEXTURE_WRAP_T
#define DGL_LINEAR_MIPMAP_LINEAR GL_LINEAR_MIPMAP_LINEAR
#define DGL_LINEAR GL_LINEAR
#define DGL_CLAMP_TO_EDGE GL_CLAMP_TO_EDGE
static inline int SDL_GL_BindTexture(SDL_Texture *texture,float *w,float *h) {
    (void)texture;(void)w;(void)h;
    SDL_SetError("OpenGL textures are unavailable in the DOS software backend");return -1;
}
static inline int SDL_GL_UnbindTexture(SDL_Texture *texture) {
    (void)texture;SDL_SetError("OpenGL textures are unavailable in the DOS software backend");return -1;
}
