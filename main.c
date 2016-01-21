/*
 * jpegtran.c
 */

//#include "cdjpeg.h"     /* Common decls for cjpeg/djpeg applications */
//#include "ocl_transupp.h"       /* Support routines for jpegtran */
#include "jpeglib.h"
#include <stdio.h>
//#include <CL/cl.h>
#include "jinclude.h"
//#include <ctype.h>
#include "jversion.h"
#include "jpegint.h"
//#ifdef USE_CCOMMAND     /* command-line reader for Macintosh */
//#ifdef __MWERKS__
//#include <SIOUX.h>              /* Metrowerks needs this */
//#include <console.h>        /* ... and this */
//#endif
//#ifdef THINK_C
//#include <console.h>        /* Think declares it here */
//#endif
//#endif
//#include <pthread.h>
//#include <dirent.h>
//#include <sys/stat.h>
//#include <sys/types.h>
//#include <unistd.h>
/*
 * Argument-parsing code.
 * The switch parser is designed to be useful with DOS-style command line
 * syntax, ie, intermixed switches and file names, where only the switches
 * to the left of a given file name affect processing of that file.
 * The main program in this file doesn't actually use this capability...
 */
//__thread const char * progname;   /* program name for error messages */
//__thread char * outfilename;  /* for -outfile switch */
//__thread char * scaleoption;  /* -scale switch */
//__thread JCOPY_OPTION copyoption; /* -copy switch */
int main(int argc, char **argv) 
{
    /* pub_env should be global variable */
    ocl_pub_env_struct* pub_env;
    pub_env = (ocl_pub_env_struct*)malloc(sizeof(ocl_pub_env_struct));
    ocl_pub_env_init(pub_env);
    /* pri_env should be thread own */
    ocl_pri_env_struct* pri_env;
    pri_env = (ocl_pri_env_struct*)malloc(sizeof(ocl_pri_env_struct));
    ocl_pri_env_init(pub_env, pri_env);

    FILE* fp;
    unsigned char* srcdata;
    unsigned char* dstdata = NULL;
    int outsize,insize;
    fp = fopen("./input_images/448_298_90_-256126272.jpg","r");
    if (fp == NULL)
    {
        printf("Couldn't open src_image_file\n");
        exit(1);
    }
    fseek(fp,0,SEEK_END);
    insize = ftell(fp);
    srcdata = (unsigned char*)malloc(insize+1);
    /* should malloc dstdata as big as enough */
    dstdata = (unsigned char*)malloc(2*insize+1);
    fseek(fp,0,SEEK_SET);
    fread(srcdata, insize+1, 1, fp);
    outsize = resize_pic(srcdata,insize,dstdata,500,300,90,pub_env,pri_env);
    printf("Outsize = %d\n",outsize);
    if (dstdata == NULL)
        exit(1);
    fclose(fp);
    fp = fopen("./dst_images/test.jpg","w");
    fwrite(dstdata,outsize,1,fp);
    fclose(fp);
    /* srcdata/dstdata/pri_env/pub_env  should be free by application */
    free(srcdata);
    free(dstdata);
    ocl_pri_env_release(pri_env);
    free(pri_env);
    ocl_pub_env_release(pub_env);
    free(pub_env);        

    return 0;
}


