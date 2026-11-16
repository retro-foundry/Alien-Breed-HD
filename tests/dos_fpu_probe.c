/* Verify that x87 operations really reach software instead of merely trusting
 * an emulator config setting. CR0.EM faults every x87 instruction on a 486. */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
extern int _8087;
typedef union { double value;uint32_t words[2]; } DoubleBits;

int main(void)
{
    unsigned short control;
    __asm__ volatile("smsw %0":"=r"(control));
    if(_8087 || !(control&4)) {
        printf("Software FPU check failed: hardware=%d control=%04x\n",_8087,control);
        return 1;
    }
    volatile double a=3.0,b=4.0;
    DoubleBits product,quotient,root;
    product.value=a*b;quotient.value=a/b;root.value=sqrt(b);
    __asm__ volatile("smsw %0":"=r"(control));
    if(!(control&4) || product.words[0] || product.words[1]!=0x40280000u ||
       quotient.words[0] || quotient.words[1]!=0x3fe80000u ||
       root.words[0] || root.words[1]!=0x40000000u) {
        puts("Software FPU check failed: multiply/divide/sqrt or CPU emulation flag changed");
        return 1;
    }
    printf("Software FPU traps active: control=%04x; multiply/divide/sqrt exact\n",control);
    return 0;
}
