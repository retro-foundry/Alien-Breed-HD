/* Exercise the DOS batch contract independently of graphics initialization. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
int main(int argc,char **argv)
{
    if(argc>1 && !strcmp(argv[1],"--dos-vga")) {
        FILE *file=fopen("VGA.OK","w");if(!file)return 1;
        for(int i=1;i<argc;i++)fprintf(file,"%s\n",argv[i]);
        return fclose(file)!=0;
    }
    const char *status=getenv("TESTEXIT");
    return status?atoi(status):1;
}
