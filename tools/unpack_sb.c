/* Offline DOS asset preparation: reuse the runtime's original =SB= decoder. */
#include "sb_decompress.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv)
{
    if(argc!=3){fprintf(stderr,"Usage: sb_unpack input output\n");return 1;}
    FILE *input=fopen(argv[1],"rb");unsigned char header[12];
    if(!input || fread(header,1,sizeof(header),input)!=sizeof(header) ||
       !sb_is_compressed(header,sizeof(header))) {
        fprintf(stderr,"Invalid =SB= input: %s\n",argv[1]);if(input)fclose(input);return 1;
    }
    fclose(input);
    unsigned char *bytes=NULL;size_t size=0;
    if(sb_load_file(argv[1],&bytes,&size) || size!=sb_unpacked_size(header,sizeof(header))) {
        fprintf(stderr,"Incomplete =SB= decode: %s\n",argv[1]);free(bytes);return 1;
    }
    FILE *output=fopen(argv[2],"wb");
    if(!output){fprintf(stderr,"Cannot write %s\n",argv[2]);free(bytes);return 1;}
    int ok=fwrite(bytes,1,size,output)==size;
    if(fclose(output))ok=0;
    free(bytes);return ok?0:1;
}
