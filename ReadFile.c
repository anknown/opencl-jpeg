#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

size_t read_all_bytes(const char * fileName,void ** mem)
{
    FILE * file;
    struct stat mystat;
    void * content;

    *mem = NULL;

    if(0 != stat(fileName,&mystat))
    {
        return 0;
    }

    file = fopen(fileName,"rb");
    if(!file)
    {	
	printf("Read file failed.\n");
        return 0;
    }

    content = malloc(mystat.st_size + 1);
    if(content == NULL)
    {	
	printf("Malloc failed.\n");
        return 0;
    }
    ((char*)content)[mystat.st_size] = 0;


    if( mystat.st_size != fread(content,1,mystat.st_size,file))
    {
        goto FREE_CONTENT;
    }
    *mem = content;
    return mystat.st_size;

FREE_CONTENT:
    free(content);
    return 0;
}

void free_all_bytes(void * mem)
{
    free(mem);
}

