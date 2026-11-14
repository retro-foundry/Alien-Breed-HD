/* Water colour representation shared by both phase samplers. */
#ifdef AB3D_NATIVE_RGB565
#define WATER_BRIGHTEN_TYPE RendererPixel
#define WATER_BRIGHTEN_SAMPLES 1024
#define WATER_SAMPLE(ROWS,LEVEL,PIXEL) ROWS[LEVEL][((PIXEL)&0x079e)>>1]
#define WATER_COLOR_LANES(COLOR) water_color_lanes(COLOR)
#define WATER_MASK0 "andl $0x079e,%%eax\n\t"
#define WATER_MASK1 "andl $0x079e,%%ecx\n\t"
#define WATER_GB0 ""
#define WATER_GB1 ""
#define WATER_BRIGHTEN0 "movzwl (%%eax,%%edx,2),%%eax\n\t"
#define WATER_BRIGHTEN1 "movzwl (%%ecx,%%edx,2),%%edx\n\t"
#define WATER_RED_DIFFERENCE "shrl $5,%%ecx\n\t"
#define WATER_GREEN_DIFFERENCE ""
#define WATER_BLUE_DIFFERENCE "shrl $10,%%edx\n\t"
#define WATER_BRIGHTEN_ADDRESS brighten_half
#define WATER_GB_INPUT
#else
#define WATER_BRIGHTEN_TYPE uint32_t
#define WATER_BRIGHTEN_SAMPLES 256
#define WATER_SAMPLE(ROWS,LEVEL,PIXEL) ROWS[LEVEL][water_gb_index[PIXEL&2047]]
#define WATER_COLOR_LANES(COLOR) COLOR>>16
#define WATER_MASK0 "andl $2047,%%eax\n\t"
#define WATER_MASK1 "andl $2047,%%ecx\n\t"
#define WATER_GB0 "movzbl %c[gb](%%eax),%%eax\n\t"
#define WATER_GB1 "movzbl %c[gb](%%ecx),%%ecx\n\t"
#define WATER_BRIGHTEN0 "movl (%%edx,%%eax,4),%%eax\n\t"
#define WATER_BRIGHTEN1 "movl (%%edx,%%ecx,4),%%edx\n\t"
#define WATER_RED_DIFFERENCE "shrl $10,%%ecx\n\t"
#define WATER_GREEN_DIFFERENCE "shrl $5,%%ecx\n\t"
#define WATER_BLUE_DIFFERENCE "andl $31,%%edx\n\t"
#define WATER_BRIGHTEN_ADDRESS brighten
#define WATER_GB_INPUT [gb] "i"(water_gb_index),
#endif
#ifdef AB3D_DOS_INDEXED
#define WATER_LOAD0 "movzbl (%[source0]),%%eax\n\t"
#define WATER_LOAD1 "movzbl (%[source1]),%%ecx\n\t"
#define WATER_STORE "movb %%al,(%[source0],%%edx)\n\t"
#define WATER_LANE_LOAD "movzbl %c[lanes](%%eax),%%eax\n\t"
#define WATER_ADVANCE "addl $1,%[source0]\n\t" "addl $1,%[source1]\n\t"
#else
#define WATER_LOAD0 "movzwl (%[source0]),%%eax\n\t"
#define WATER_LOAD1 "movzwl (%[source1]),%%ecx\n\t"
#define WATER_STORE "movw %%ax,(%[source0],%%edx)\n\t"
#define WATER_LANE_LOAD \
    "movl %%eax,%%ecx\n\t" \
    "shrl $5,%%ecx\n\t" \
    "movzwl %c[lanes](%%ecx,%%ecx),%%ecx\n\t" \
    "andl $15,%%eax\n\t" \
    "movzwl %c[blue](%%eax,%%eax),%%eax\n\t" \
    "orl %%ecx,%%eax\n\t"
#define WATER_ADVANCE "addl $2,%[source0]\n\t" "addl $2,%[source1]\n\t"
#endif

#ifdef AB3D_NATIVE_RGB565
/* Extract the original four-bit channels with one zero gap, then subtract
 * behind their guard bits. Preserve EAX's ready word until blending changes it. */
#define WATER_NATIVE_DIFFERENCE(COMPARE, BEFORE_BLEND) \
    "movl %%eax,%%ecx\n\t" \
    "andl $0xf79e,%%ecx\n\t" \
    "andl $0xf79e,%%edx\n\t" \
    "orl $0x10820,%%edx\n\t" \
    "subl %%ecx,%%edx\n\t" \
    "shrl $1,%%edx\n\t" \
    "rorw $6,%%dx\n\t" \
    "movzbl %c[zero](%%edx),%%ecx\n\t" \
    COMPARE \
    "jae 2f\n\t" \
    BEFORE_BLEND \
    "movl %%eax,%%ecx\n\t" \
    "shrl $2,%%eax\n\t" \
    "andl $0x3de0,%%eax\n\t" \
    "shrl $1,%%ecx\n\t" \
    "andl $15,%%ecx\n\t" \
    "orl %%ecx,%%eax\n\t"
#define WATER_DIFFERENCE WATER_NATIVE_DIFFERENCE("cmpl 32(%[blend]),%%ecx\n\t", )
#else
#define WATER_DIFFERENCE \
    "shrl $16,%%eax\n\t" \
    "shrl $16,%%edx\n\t" \
    "orl $0x4210,%%edx\n\t" \
    "subl %%eax,%%edx\n\t"
#endif

#ifdef AB3D_NATIVE_RGB565
#define WATER_TABLE_INPUTS [lanes] "i"(water_rg),[blue] "i"(water_blue),[zero] "i"(water_zero_threshold)
#else
#define WATER_TABLE_INPUTS [lanes] "i"(water_lanes)
#endif

