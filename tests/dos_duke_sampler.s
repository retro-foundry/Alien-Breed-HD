/* Read the unmodified Duke3D DNRATE pixels once per 18 BIOS timer ticks.
 * No file I/O occurs in the interrupt handler. Invoke with any argument after
 * Duke exits to save the resident records. Used only in an isolated DOSBox. */
.code16
.ifdef SAMPLE_S3
.equ RECORD_SIZE,264
.else
.equ RECORD_SIZE,260
.endif
.text
.globl _start
_start:
    cld
    pushw %cs
    popw %ds
    cmpb $0,0x80
    jne dump
    movw $0x351c,%ax
    int $0x21
    movw %bx,(old - _start + 0x100)
    movw %es,(old - _start + 0x102)
    movw $(timer - _start + 0x100),%dx
    movw $0x251c,%ax
    int $0x21
    movw $(service - _start + 0x100),%dx
    movw $0x2560,%ax
    int $0x21
    movw $((end - _start + 0x10f)/16),%dx
    movw $0x3100,%ax
    int $0x21
dump:
    movw $0xad40,%ax
    int $0x60
    cmpw $0x40ad,%ax
    jne failed
    pushw %es
    pushw %bx
    pushw %cx
    movw $(filename - _start + 0x100),%dx
    xorw %cx,%cx
    movw $0x3c00,%ax
    int $0x21
    jc failed
    movw %ax,%bx
    popw %cx
    popw %dx
    popw %ds
    movb $0x40,%ah
    int $0x21
    jc failed
    cmpw %cx,%ax
    jne failed
    movb $0x3e,%ah
    int $0x21
    movw $0x4c00,%ax
    int $0x21
failed:
    movw $0x4c01,%ax
    int $0x21
service:
    cmpw $0xad40,%ax
    jne no_service
    pushw %cs
    popw %es
    movw $(records - _start + 0x100),%bx
    movw %cs:(records - _start + 0x100),%cx
    imulw $RECORD_SIZE,%cx
    addw $4,%cx
    movw $0x40ad,%ax
no_service:
    iret
timer:
    cld
    pushal
    pushw %ds
    pushw %es
    pushw %cs
    popw %es
    decw %cs:(delay - _start + 0x100)
    jnz timer_done
    movw $18,%cs:(delay - _start + 0x100)
    cmpw $64,%cs:(records - _start + 0x100)
    jae timer_done
    movw $0x40,%ax
    movw %ax,%ds
    /* VESA uses a different BIOS mode byte. Its 320-pixel bank window
       retains the same counter layout; sample memory without BIOS calls. */
.ifndef SAMPLE_VESA
    cmpb $0x13,0x49
    jne timer_done
.endif
    movw %cs:(records - _start + 0x100),%di
    imulw $RECORD_SIZE,%di
    addw $(records - _start + 0x104),%di
    movl 0x6c,%eax
    stosl
.ifdef SAMPLE_S3
    /* S3 Trio CR6A controls the CPU bank, independently of display start.
       Save the interrupted CRTC index and bank; bank zero contains DNRATE.
       DOSBox Staging v0.82.2 src/hardware/vga_s3.cpp supplies this contract. */
    movw $0x3d4,%dx
    inb %dx,%al
    movb %al,%ah
    pushw %ax
    movb $0x6a,%al
    outb %al,%dx
    incw %dx
    inb %dx,%al
    pushw %ax
    stosw
    xorb %al,%al
    outb %al,%dx
.endif
    movw $0xa000,%ax
    movw %ax,%ds
    xorw %si,%si
    movw $8,%dx
copy_row:
    movw $32,%cx
    rep movsb
    addw $288,%si
    decw %dx
    jnz copy_row
.ifdef SAMPLE_S3
    movw $0x3d4,%dx
    movb $0x6a,%al
    outb %al,%dx
    incw %dx
    popw %ax
    outb %al,%dx
    inb %dx,%al
    stosb
    decw %dx
    popw %ax
    outb %al,%dx
    inb %dx,%al
    stosb
.endif
    incw %cs:(records - _start + 0x100)
timer_done:
    popw %es
    popw %ds
    popal
    ljmp *%cs:(old - _start + 0x100)
old: .word 0,0
delay: .word 18
filename: .asciz "FPS.BIN"
records: .word 0,RECORD_SIZE
    .space 64*RECORD_SIZE
end:
