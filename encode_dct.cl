/*
 * encode_idct.cl
 * author: xiaoE
 * describe:
 *   This file contains the coefficient DCT kernel.
 *   The output of this file: coefficient buffer lies between entropy decoding and inverse-DCT steps.
 *   The input of this file:the result of dehuffman buffer and relevant channel struct&info.
 */
#define MAX_COMPONENT_INFO_COUNT 10
#define DCTSIZE2 64
#define DCTSIZE 8
#define DCTSIZE16_16 256
#define DCTSIZE16_8 128
#define DCTSIZE16 16
#define MAXJSAMPLE  255
#define CENTERJSAMPLE   128
typedef short JCOEF;
typedef JCOEF JBLOCK[DCTSIZE2]; /* one block of coefficients */
typedef JBLOCK *JBLOCKROW;
typedef JBLOCKROW *JBLOCKARRAY;
typedef JBLOCKARRAY *JBLOCKIMAGE;

typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;

typedef unsigned int JDIMENSION;
typedef int INT32;
typedef short INT16;
typedef float FAST_FLOAT;
typedef FAST_FLOAT FLOAT_MULT_TYPE; /* preferred floating type */

typedef float DCT_FLOAT;
typedef int DCT_INT;

#define MAX_COMPONENT_INFO_COUNT 10


#define CONST_BITS  13
#define PASS1_BITS  2
#define FIX_0_298631336  ((INT32)  2446)        /* FIX(0.298631336) */
#define FIX_0_390180644  ((INT32)  3196)        /* FIX(0.390180644) */
#define FIX_0_541196100  ((INT32)  4433)        /* FIX(0.541196100) */
#define FIX_0_765366865  ((INT32)  6270)        /* FIX(0.765366865) */
#define FIX_0_899976223  ((INT32)  7373)        /* FIX(0.899976223) */
#define FIX_1_175875602  ((INT32)  9633)        /* FIX(1.175875602) */
#define FIX_1_501321110  ((INT32)  12299)       /* FIX(1.501321110) */
#define FIX_1_847759065  ((INT32)  15137)       /* FIX(1.847759065) */
#define FIX_1_961570560  ((INT32)  16069)       /* FIX(1.961570560) */
#define FIX_2_053119869  ((INT32)  16819)       /* FIX(2.053119869) */
#define FIX_2_562915447  ((INT32)  20995)       /* FIX(2.562915447) */
#define FIX_3_072711026  ((INT32)  25172)       /* FIX(3.072711026) */
#define MULTIPLY16C16(var,const)  (((INT16) (var)) * ((INT16) (const)))
//#define MULTIPLY(var,const)  MULTIPLY16C16(var,const)
#define MULTIPLY(var,const)  ((var)*(const))
#define DESCALE(x,n)  RIGHT_SHIFT((x) + (ONE << ((n)-1)), n)
#define ONE     ((INT32) 1)
#define CONST_SCALE (ONE << CONST_BITS)
#define RIGHT_SHIFT(x,shft)     ((x) >> (shft))
#define RANGE_MASK  (MAXJSAMPLE * 4 + 3) /* 2 bits wider than legal samples */
#define FIX(x)  ((INT32) ((x) * CONST_SCALE + 0.5))
#define MAX_COMPONENT_COUNT 4

struct ComponentInfo
{
    /* resize kernel need:*/
    unsigned int src_width; //unpadded num of sample_width = compptr->downsampled_width
    unsigned int src_height;//unpadded num of sample_height
    unsigned int resize_blks_in_row; //blk width in coef_array without round8 : coef_array[0]->blocksperrow
    unsigned int resize_blks_in_height;//blk height in coef_array without round8: coef_array[0]->rows_in_array;
    unsigned int resize_blks_out_row; //blk width in coef_array without round8 : coef_array[0]->blocksperrow
    unsigned int resize_blks_out_height;//blk height in coef_array without round8: coef_array[0]->rows_in_array;

    /* idct_kernel need:*/
    int idct_blks_in_row;//unpadded num of blk_width:
    /* dct_kernel need:*/
    unsigned int dst_blks_in_row;//
    unsigned int dst_blks_in_height;
    float idct_table[DCTSIZE2];//coefficient for AAN-idct2
    float fdct_table[DCTSIZE2];//coefficient for AAN-fdct2
    int idct_int_table[DCTSIZE2];//coefficient for AAN-idct2
    int fdct_int_table[DCTSIZE2];//coefficient for AAN-fdct2
};

struct DecodeInfo
{
    struct ComponentInfo compInfo[MAX_COMPONENT_INFO_COUNT];
};

struct QuantiTable
{
    float idct_table_f[4][64];
    float fdct_table_f[4][64];
    int idct_table_i[4][64];
    int fdct_table_i[4][64];
};

void FDCT_8x8(__local JSAMPLE* samp_buffer,
    __local DCT_FLOAT* q_tbl_ptr,
    short8* fdct_out,
    __local FAST_FLOAT* workspace)
{
    FAST_FLOAT tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    FAST_FLOAT tmp10, tmp11, tmp12, tmp13;
    FAST_FLOAT z1, z2, z3, z4, z5, z11, z13;
    __local DCT_FLOAT* quantptr;
    __local FAST_FLOAT* wsptr;
    __local JSAMPLE* inptr;
    inptr = samp_buffer;
    int ctr = get_global_id(1);
    wsptr = workspace;
    quantptr = q_tbl_ptr;
  
    inptr += ctr * DCTSIZE;
    wsptr += ctr * DCTSIZE;

    tmp0 = (FAST_FLOAT)(inptr[0] + inptr[7]);
    tmp7 = (FAST_FLOAT)(inptr[0] - inptr[7]);
    tmp1 = (FAST_FLOAT)(inptr[1] + inptr[6]);
    tmp6 = (FAST_FLOAT)(inptr[1] - inptr[6]);
    tmp2 = (FAST_FLOAT)(inptr[2] + inptr[5]);
    tmp5 = (FAST_FLOAT)(inptr[2] - inptr[5]);
    tmp3 = (FAST_FLOAT)(inptr[3] + inptr[4]);
    tmp4 = (FAST_FLOAT)(inptr[3] - inptr[4]);

    /* Event part */
    tmp10 = tmp0 + tmp3;    
    tmp13 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp12 = tmp1 - tmp2;

    /* Apply unsigned->signed conversion */
    wsptr[0] = tmp10 + tmp11 - 8 * CENTERJSAMPLE; 
    wsptr[4] = tmp10 - tmp11;

    z1 = (tmp12 + tmp13) * ((FAST_FLOAT) 0.707106781); 
    wsptr[2] = tmp13 + z1;    
    wsptr[6] = tmp13 - z1;

    /* Odd part */
    tmp10 = tmp4 + tmp5;    
    tmp11 = tmp5 + tmp6;
    tmp12 = tmp6 + tmp7;

    /* The rotator is modified from fig 4-8 to avoid extra negations. */
    z5 = (tmp10 - tmp12) * ((FAST_FLOAT) 0.382683433); /* c6 */
    z2 = ((FAST_FLOAT) 0.541196100) * tmp10 + z5; /* c2-c6 */
    z4 = ((FAST_FLOAT) 1.306562965) * tmp12 + z5; /* c2+c6 */
    z3 = tmp11 * ((FAST_FLOAT) 0.707106781); /* c4 */

    z11 = tmp7 + z3;        /* phase 5 */
    z13 = tmp7 - z3;

    wsptr[5] = z13 + z2;  /* phase 6 */
    wsptr[3] = z13 - z2;
    wsptr[1] = z11 + z4;
    wsptr[7] = z11 - z4;

    barrier(CLK_LOCAL_MEM_FENCE);

    /*FDCT process columns*/
    wsptr = workspace;
    wsptr +=ctr;

    tmp0 = wsptr[DCTSIZE*0] + wsptr[DCTSIZE*7];
    tmp7 = wsptr[DCTSIZE*0] - wsptr[DCTSIZE*7];
    tmp1 = wsptr[DCTSIZE*1] + wsptr[DCTSIZE*6];
    tmp6 = wsptr[DCTSIZE*1] - wsptr[DCTSIZE*6];
    tmp2 = wsptr[DCTSIZE*2] + wsptr[DCTSIZE*5];
    tmp5 = wsptr[DCTSIZE*2] - wsptr[DCTSIZE*5];
    tmp3 = wsptr[DCTSIZE*3] + wsptr[DCTSIZE*4];
    tmp4 = wsptr[DCTSIZE*3] - wsptr[DCTSIZE*4];

    /* Even part */
    tmp10 = tmp0 + tmp3;    
    tmp13 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp12 = tmp1 - tmp2;

    wsptr[DCTSIZE*0] = tmp10 + tmp11; 
    wsptr[DCTSIZE*4] = tmp10 - tmp11;

    z1 = (tmp12 + tmp13) * ((FAST_FLOAT) 0.707106781); 
    wsptr[DCTSIZE*2] = tmp13 + z1; 
    wsptr[DCTSIZE*6] = tmp13 - z1;

    /* Odd part */
    tmp10 = tmp4 + tmp5;    
    tmp11 = tmp5 + tmp6;
    tmp12 = tmp6 + tmp7;

    z5 = (tmp10 - tmp12) * ((FAST_FLOAT) 0.382683433); 
    z2 = ((FAST_FLOAT) 0.541196100) * tmp10 + z5; 
    z4 = ((FAST_FLOAT) 1.306562965) * tmp12 + z5; 
    z3 = tmp11 * ((FAST_FLOAT) 0.707106781); 

    z11 = tmp7 + z3;        
    z13 = tmp7 - z3;

    wsptr[DCTSIZE*5] = z13 + z2; 
    wsptr[DCTSIZE*3] = z13 - z2;
    wsptr[DCTSIZE*1] = z11 + z4;
    wsptr[DCTSIZE*7] = z11 - z4;

    barrier(CLK_LOCAL_MEM_FENCE);
    /* store out */
    wsptr = workspace;
    wsptr += ctr * DCTSIZE;
    short8 out;
    quantptr += ctr * DCTSIZE;
    float8 f = (float8)
       (wsptr[0] * quantptr[0],
        wsptr[1] * quantptr[1],
        wsptr[2] * quantptr[2],
        wsptr[3] * quantptr[3],
        wsptr[4] * quantptr[4],
        wsptr[5] * quantptr[5],
        wsptr[6] * quantptr[6],
        wsptr[7] * quantptr[7]);
    out = convert_short8_sat_rte(f);
    *fdct_out = out;
}

void FDCT_16x16(__local JSAMPLE* samp_buffer,
    __local DCT_INT* q_tbl_ptr,
    short8* fdct_out,
    __local DCT_INT* workspace)
{
    int ctr = get_global_id(1);
    int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17;
    __local DCT_INT* quantptr;
    __local DCT_INT* wsptr;
    __local DCT_INT* wsptr2;
    __local JSAMPLE* inptr;
    quantptr = q_tbl_ptr;
  
    inptr = samp_buffer + ctr * DCTSIZE16;
    wsptr = workspace + ctr * DCTSIZE ;
    int i;
    for (i = 0; i < 2; i++)
    {
        inptr += i * DCTSIZE16 * 8;
        wsptr += i * DCTSIZE2;        
        
        tmp0 = inptr[0] + inptr[15];
        tmp1 = inptr[1] + inptr[14];
        tmp2 = inptr[2] + inptr[13];
        tmp3 = inptr[3] + inptr[12];
        tmp4 = inptr[4] + inptr[11];
        tmp5 = inptr[5] + inptr[10];
        tmp6 = inptr[6] + inptr[9];
        tmp7 = inptr[7] + inptr[8];

        tmp10 = tmp0 + tmp7;
        tmp14 = tmp0 - tmp7;
        tmp11 = tmp1 + tmp6;
        tmp15 = tmp1 - tmp6;
        tmp12 = tmp2 + tmp5;
        tmp16 = tmp2 - tmp5;
        tmp13 = tmp3 + tmp4;
        tmp17 = tmp3 - tmp4;

        tmp0 = inptr[0] - inptr[15];
        tmp1 = inptr[1] - inptr[14];
        tmp2 = inptr[2] - inptr[13];
        tmp3 = inptr[3] - inptr[12];
        tmp4 = inptr[4] - inptr[11];
        tmp5 = inptr[5] - inptr[10];
        tmp6 = inptr[6] - inptr[9];
        tmp7 = inptr[7] - inptr[8];

        /* Apply unsigned->signed conversion */
        wsptr[0] = (DCT_INT)
          ((tmp10 + tmp11 + tmp12 + tmp13 - 16 * CENTERJSAMPLE) << PASS1_BITS);
        wsptr[4] = (DCT_INT)
          DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.306562965)) + /* c4[16] = c2[8] */
              MULTIPLY(tmp11 - tmp12, FIX_0_541196100),   /* c12[16] = c6[8] */
              CONST_BITS-PASS1_BITS);

        tmp10 = MULTIPLY(tmp17 - tmp15, FIX(0.275899379)) +   /* c14[16] = c7[8] */
            MULTIPLY(tmp14 - tmp16, FIX(1.387039845));    /* c2[16] = c1[8] */

        wsptr[2] = (DCT_INT)
          DESCALE(tmp10 + MULTIPLY(tmp15, FIX(1.451774982))   /* c6+c14 */
              + MULTIPLY(tmp16, FIX(2.172734804)),        /* c2+c10 */
              CONST_BITS-PASS1_BITS);
        wsptr[6] = (DCT_INT)
          DESCALE(tmp10 - MULTIPLY(tmp14, FIX(0.211164243))   /* c2-c6 */
              - MULTIPLY(tmp17, FIX(1.061594338)),        /* c10+c14 */
              CONST_BITS-PASS1_BITS);

        /* Odd part */

        tmp11 = MULTIPLY(tmp0 + tmp1, FIX(1.353318001)) +         /* c3 */
            MULTIPLY(tmp6 - tmp7, FIX(0.410524528));          /* c13 */
        tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.247225013)) +         /* c5 */
            MULTIPLY(tmp5 + tmp7, FIX(0.666655658));          /* c11 */
        tmp13 = MULTIPLY(tmp0 + tmp3, FIX(1.093201867)) +         /* c7 */
            MULTIPLY(tmp4 - tmp7, FIX(0.897167586));          /* c9 */
        tmp14 = MULTIPLY(tmp1 + tmp2, FIX(0.138617169)) +         /* c15 */
            MULTIPLY(tmp6 - tmp5, FIX(1.407403738));          /* c1 */
        tmp15 = MULTIPLY(tmp1 + tmp3, - FIX(0.666655658)) +       /* -c11 */
            MULTIPLY(tmp4 + tmp6, - FIX(1.247225013));        /* -c5 */
        tmp16 = MULTIPLY(tmp2 + tmp3, - FIX(1.353318001)) +       /* -c3 */
            MULTIPLY(tmp5 - tmp4, FIX(0.410524528));          /* c13 */
        tmp10 = tmp11 + tmp12 + tmp13 -
            MULTIPLY(tmp0, FIX(2.286341144)) +                /* c7+c5+c3-c1 */
            MULTIPLY(tmp7, FIX(0.779653625));                 /* c15+c13-c11+c9 */
        tmp11 += tmp14 + tmp15 + MULTIPLY(tmp1, FIX(0.071888074)) /* c9-c3-c15+c11 */
             - MULTIPLY(tmp6, FIX(1.663905119));              /* c7+c13+c1-c5 */
        tmp12 += tmp14 + tmp16 - MULTIPLY(tmp2, FIX(1.125726048)) /* c7+c5+c15-c3 */
             + MULTIPLY(tmp5, FIX(1.227391138));              /* c9-c11+c1-c13 */
        tmp13 += tmp15 + tmp16 + MULTIPLY(tmp3, FIX(1.065388962)) /* c15+c3+c11-c7 */
             + MULTIPLY(tmp4, FIX(2.167985692));              /* c1+c13+c5-c9 */

        wsptr[1] = (DCT_INT) DESCALE(tmp10, CONST_BITS-PASS1_BITS);
        wsptr[3] = (DCT_INT) DESCALE(tmp11, CONST_BITS-PASS1_BITS);
        wsptr[5] = (DCT_INT) DESCALE(tmp12, CONST_BITS-PASS1_BITS);
        wsptr[7] = (DCT_INT) DESCALE(tmp13, CONST_BITS-PASS1_BITS);

    }
    barrier(CLK_LOCAL_MEM_FENCE);
/*
    wsptr2 = workspace ;
    wsptr = workspace + DCTSIZE2;
    wsptr2 += ctr;
    wsptr += ctr;
*/
    wsptr2 = workspace;
    wsptr2 += ctr;
    wsptr = workspace;
    wsptr += DCTSIZE2 + ctr;

    tmp0 = wsptr2[DCTSIZE*0] + wsptr[DCTSIZE*7];
    tmp1 = wsptr2[DCTSIZE*1] + wsptr[DCTSIZE*6];
    tmp2 = wsptr2[DCTSIZE*2] + wsptr[DCTSIZE*5];
    tmp3 = wsptr2[DCTSIZE*3] + wsptr[DCTSIZE*4];
    tmp4 = wsptr2[DCTSIZE*4] + wsptr[DCTSIZE*3];
    tmp5 = wsptr2[DCTSIZE*5] + wsptr[DCTSIZE*2];
    tmp6 = wsptr2[DCTSIZE*6] + wsptr[DCTSIZE*1];
    tmp7 = wsptr2[DCTSIZE*7] + wsptr[DCTSIZE*0];

    tmp10 = tmp0 + tmp7;
    tmp14 = tmp0 - tmp7;
    tmp11 = tmp1 + tmp6;
    tmp15 = tmp1 - tmp6;
    tmp12 = tmp2 + tmp5;
    tmp16 = tmp2 - tmp5;
    tmp13 = tmp3 + tmp4;
    tmp17 = tmp3 - tmp4;

    tmp0 = wsptr2[DCTSIZE*0] - wsptr[DCTSIZE*7];
    tmp1 = wsptr2[DCTSIZE*1] - wsptr[DCTSIZE*6];
    tmp2 = wsptr2[DCTSIZE*2] - wsptr[DCTSIZE*5];
    tmp3 = wsptr2[DCTSIZE*3] - wsptr[DCTSIZE*4];
    tmp4 = wsptr2[DCTSIZE*4] - wsptr[DCTSIZE*3];
    tmp5 = wsptr2[DCTSIZE*5] - wsptr[DCTSIZE*2];
    tmp6 = wsptr2[DCTSIZE*6] - wsptr[DCTSIZE*1];
    tmp7 = wsptr2[DCTSIZE*7] - wsptr[DCTSIZE*0];

    wsptr2[DCTSIZE*0] = (DCT_INT)
      DESCALE(tmp10 + tmp11 + tmp12 + tmp13, PASS1_BITS+2);
    wsptr2[DCTSIZE*4] = (DCT_INT)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.306562965)) + /* c4[16] = c2[8] */
          MULTIPLY(tmp11 - tmp12, FIX_0_541196100),   /* c12[16] = c6[8] */
          CONST_BITS+PASS1_BITS+2);

    tmp10 = MULTIPLY(tmp17 - tmp15, FIX(0.275899379)) +   /* c14[16] = c7[8] */
        MULTIPLY(tmp14 - tmp16, FIX(1.387039845));    /* c2[16] = c1[8] */

    wsptr2[DCTSIZE*2] = (DCT_INT)
      DESCALE(tmp10 + MULTIPLY(tmp15, FIX(1.451774982))   /* c6+c14 */
          + MULTIPLY(tmp16, FIX(2.172734804)),        /* c2+10 */
          CONST_BITS+PASS1_BITS+2);
    wsptr2[DCTSIZE*6] = (DCT_INT)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(0.211164243))   /* c2-c6 */
          - MULTIPLY(tmp17, FIX(1.061594338)),        /* c10+c14 */
          CONST_BITS+PASS1_BITS+2);

    /* Odd part */

    tmp11 = MULTIPLY(tmp0 + tmp1, FIX(1.353318001)) +         /* c3 */
        MULTIPLY(tmp6 - tmp7, FIX(0.410524528));          /* c13 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.247225013)) +         /* c5 */
        MULTIPLY(tmp5 + tmp7, FIX(0.666655658));          /* c11 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(1.093201867)) +         /* c7 */
        MULTIPLY(tmp4 - tmp7, FIX(0.897167586));          /* c9 */
    tmp14 = MULTIPLY(tmp1 + tmp2, FIX(0.138617169)) +         /* c15 */
        MULTIPLY(tmp6 - tmp5, FIX(1.407403738));          /* c1 */
    tmp15 = MULTIPLY(tmp1 + tmp3, - FIX(0.666655658)) +       /* -c11 */
        MULTIPLY(tmp4 + tmp6, - FIX(1.247225013));        /* -c5 */
    tmp16 = MULTIPLY(tmp2 + tmp3, - FIX(1.353318001)) +       /* -c3 */
        MULTIPLY(tmp5 - tmp4, FIX(0.410524528));          /* c13 */
    tmp10 = tmp11 + tmp12 + tmp13 -
        MULTIPLY(tmp0, FIX(2.286341144)) +                /* c7+c5+c3-c1 */
        MULTIPLY(tmp7, FIX(0.779653625));                 /* c15+c13-c11+c9 */
    tmp11 += tmp14 + tmp15 + MULTIPLY(tmp1, FIX(0.071888074)) /* c9-c3-c15+c11 */
         - MULTIPLY(tmp6, FIX(1.663905119));              /* c7+c13+c1-c5 */
    tmp12 += tmp14 + tmp16 - MULTIPLY(tmp2, FIX(1.125726048)) /* c7+c5+c15-c3 */
         + MULTIPLY(tmp5, FIX(1.227391138));              /* c9-c11+c1-c13 */
    tmp13 += tmp15 + tmp16 + MULTIPLY(tmp3, FIX(1.065388962)) /* c15+c3+c11-c7 */
         + MULTIPLY(tmp4, FIX(2.167985692));              /* c1+c13+c5-c9 */

    wsptr2[DCTSIZE*1] = (DCT_INT) DESCALE(tmp10, CONST_BITS+PASS1_BITS+2);
    wsptr2[DCTSIZE*3] = (DCT_INT) DESCALE(tmp11, CONST_BITS+PASS1_BITS+2);
    wsptr2[DCTSIZE*5] = (DCT_INT) DESCALE(tmp12, CONST_BITS+PASS1_BITS+2);
    wsptr2[DCTSIZE*7] = (DCT_INT) DESCALE(tmp13, CONST_BITS+PASS1_BITS+2);    

    barrier(CLK_LOCAL_MEM_FENCE);

    wsptr = workspace;
    wsptr += ctr * DCTSIZE;
    quantptr += ctr * DCTSIZE;
    float out[8];
    int j;
    for (j = 0;j < 8;j++)
    {
        if (abs(wsptr[j]) < quantptr[j])
            out[j] = 0;
        else
            out[j] = wsptr[j] / quantptr[j];
    }
    float8 f = (float8)
       (out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    *fdct_out = convert_short8_sat_rte(f);
}

void FDCT_16x8(__local JSAMPLE* samp_buffer,
    __local DCT_INT* q_tbl_ptr,
    short8* fdct_out,
    __local DCT_INT* workspace)
{
    DCT_INT ctr = get_global_id(1);
    DCT_INT tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    DCT_INT tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17;
    DCT_INT z1;
    __local DCT_INT* quantptr;
    __local DCT_INT* wsptr;
    __local JSAMPLE* inptr;
    quantptr = q_tbl_ptr;
    inptr = samp_buffer + ctr * DCTSIZE16;
    wsptr = workspace + ctr * DCTSIZE ;
    
    tmp0 = inptr[0] + inptr[15];
    tmp1 = inptr[1] + inptr[14];
    tmp2 = inptr[2] + inptr[13];
    tmp3 = inptr[3] + inptr[12];
    tmp4 = inptr[4] + inptr[11];
    tmp5 = inptr[5] + inptr[10];
    tmp6 = inptr[6] + inptr[9];
    tmp7 = inptr[7] + inptr[8];

    tmp10 = tmp0 + tmp7;
    tmp14 = tmp0 - tmp7;
    tmp11 = tmp1 + tmp6;
    tmp15 = tmp1 - tmp6;
    tmp12 = tmp2 + tmp5;
    tmp16 = tmp2 - tmp5;
    tmp13 = tmp3 + tmp4;
    tmp17 = tmp3 - tmp4;

    tmp0 = inptr[0] - inptr[15];
    tmp1 = inptr[1] - inptr[14];
    tmp2 = inptr[2] - inptr[13];
    tmp3 = inptr[3] - inptr[12];
    tmp4 = inptr[4] - inptr[11];
    tmp5 = inptr[5] - inptr[10];
    tmp6 = inptr[6] - inptr[9];
    tmp7 = inptr[7] - inptr[8];

    /* Apply unsigned->signed conversion */
    wsptr[0] = (DCT_INT)
      ((tmp10 + tmp11 + tmp12 + tmp13 - 16 * CENTERJSAMPLE) << PASS1_BITS);
    wsptr[4] = (DCT_INT)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.306562965)) + /* c4[16] = c2[8] */
          MULTIPLY(tmp11 - tmp12, FIX_0_541196100),   /* c12[16] = c6[8] */
          CONST_BITS-PASS1_BITS);

    tmp10 = MULTIPLY(tmp17 - tmp15, FIX(0.275899379)) +   /* c14[16] = c7[8] */
        MULTIPLY(tmp14 - tmp16, FIX(1.387039845));    /* c2[16] = c1[8] */

    wsptr[2] = (DCT_INT)
      DESCALE(tmp10 + MULTIPLY(tmp15, FIX(1.451774982))   /* c6+c14 */
          + MULTIPLY(tmp16, FIX(2.172734804)),        /* c2+c10 */
          CONST_BITS-PASS1_BITS);
    wsptr[6] = (DCT_INT)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(0.211164243))   /* c2-c6 */
          - MULTIPLY(tmp17, FIX(1.061594338)),        /* c10+c14 */
          CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp11 = MULTIPLY(tmp0 + tmp1, FIX(1.353318001)) +         /* c3 */
        MULTIPLY(tmp6 - tmp7, FIX(0.410524528));          /* c13 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.247225013)) +         /* c5 */
        MULTIPLY(tmp5 + tmp7, FIX(0.666655658));          /* c11 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(1.093201867)) +         /* c7 */
        MULTIPLY(tmp4 - tmp7, FIX(0.897167586));          /* c9 */
    tmp14 = MULTIPLY(tmp1 + tmp2, FIX(0.138617169)) +         /* c15 */
        MULTIPLY(tmp6 - tmp5, FIX(1.407403738));          /* c1 */
    tmp15 = MULTIPLY(tmp1 + tmp3, - FIX(0.666655658)) +       /* -c11 */
        MULTIPLY(tmp4 + tmp6, - FIX(1.247225013));        /* -c5 */
    tmp16 = MULTIPLY(tmp2 + tmp3, - FIX(1.353318001)) +       /* -c3 */
        MULTIPLY(tmp5 - tmp4, FIX(0.410524528));          /* c13 */
    tmp10 = tmp11 + tmp12 + tmp13 -
        MULTIPLY(tmp0, FIX(2.286341144)) +                /* c7+c5+c3-c1 */
        MULTIPLY(tmp7, FIX(0.779653625));                 /* c15+c13-c11+c9 */
    tmp11 += tmp14 + tmp15 + MULTIPLY(tmp1, FIX(0.071888074)) /* c9-c3-c15+c11 */
         - MULTIPLY(tmp6, FIX(1.663905119));              /* c7+c13+c1-c5 */
    tmp12 += tmp14 + tmp16 - MULTIPLY(tmp2, FIX(1.125726048)) /* c7+c5+c15-c3 */
         + MULTIPLY(tmp5, FIX(1.227391138));              /* c9-c11+c1-c13 */
    tmp13 += tmp15 + tmp16 + MULTIPLY(tmp3, FIX(1.065388962)) /* c15+c3+c11-c7 */
         + MULTIPLY(tmp4, FIX(2.167985692));              /* c1+c13+c5-c9 */

    wsptr[1] = (DCT_INT) DESCALE(tmp10, CONST_BITS-PASS1_BITS);
    wsptr[3] = (DCT_INT) DESCALE(tmp11, CONST_BITS-PASS1_BITS);
    wsptr[5] = (DCT_INT) DESCALE(tmp12, CONST_BITS-PASS1_BITS);
    wsptr[7] = (DCT_INT) DESCALE(tmp13, CONST_BITS-PASS1_BITS);

    barrier(CLK_LOCAL_MEM_FENCE);
    wsptr = workspace;
    wsptr += ctr;

    tmp0 = wsptr[DCTSIZE*0] + wsptr[DCTSIZE*7];
    tmp1 = wsptr[DCTSIZE*1] + wsptr[DCTSIZE*6];
    tmp2 = wsptr[DCTSIZE*2] + wsptr[DCTSIZE*5];
    tmp3 = wsptr[DCTSIZE*3] + wsptr[DCTSIZE*4];

    tmp10 = tmp0 + tmp3;
    tmp12 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp13 = tmp1 - tmp2;

    tmp0 = wsptr[DCTSIZE*0] - wsptr[DCTSIZE*7];
    tmp1 = wsptr[DCTSIZE*1] - wsptr[DCTSIZE*6];
    tmp2 = wsptr[DCTSIZE*2] - wsptr[DCTSIZE*5];
    tmp3 = wsptr[DCTSIZE*3] - wsptr[DCTSIZE*4];

    wsptr[DCTSIZE*0] = (DCT_INT) DESCALE(tmp10 + tmp11, PASS1_BITS+1);
    wsptr[DCTSIZE*4] = (DCT_INT) DESCALE(tmp10 - tmp11, PASS1_BITS+1);

    z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
    wsptr[DCTSIZE*2] = (DCT_INT) DESCALE(z1 + MULTIPLY(tmp12, FIX_0_765366865),
                       CONST_BITS+PASS1_BITS+1);
    wsptr[DCTSIZE*6] = (DCT_INT) DESCALE(z1 - MULTIPLY(tmp13, FIX_1_847759065),
                       CONST_BITS+PASS1_BITS+1);

    /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
     * 8-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/16).
     * i0..i3 in the paper are tmp0..tmp3 here.
     */

    tmp10 = tmp0 + tmp3;
    tmp11 = tmp1 + tmp2;
    tmp12 = tmp0 + tmp2;
    tmp13 = tmp1 + tmp3;
    z1 = MULTIPLY(tmp12 + tmp13, FIX_1_175875602); /*  c3 */

    tmp0  = MULTIPLY(tmp0,    FIX_1_501321110);    /*  c1+c3-c5-c7 */
    tmp1  = MULTIPLY(tmp1,    FIX_3_072711026);    /*  c1+c3+c5-c7 */
    tmp2  = MULTIPLY(tmp2,    FIX_2_053119869);    /*  c1+c3-c5+c7 */
    tmp3  = MULTIPLY(tmp3,    FIX_0_298631336);    /* -c1+c3+c5-c7 */
    tmp10 = MULTIPLY(tmp10, - FIX_0_899976223);    /*  c7-c3 */
    tmp11 = MULTIPLY(tmp11, - FIX_2_562915447);    /* -c1-c3 */
    tmp12 = MULTIPLY(tmp12, - FIX_0_390180644);    /*  c5-c3 */
    tmp13 = MULTIPLY(tmp13, - FIX_1_961570560);    /* -c3-c5 */

    tmp12 += z1;
    tmp13 += z1;

    wsptr[DCTSIZE*1] = (DCT_INT) DESCALE(tmp0 + tmp10 + tmp12,
                       CONST_BITS+PASS1_BITS+1);
    wsptr[DCTSIZE*3] = (DCT_INT) DESCALE(tmp1 + tmp11 + tmp13,
                       CONST_BITS+PASS1_BITS+1);
    wsptr[DCTSIZE*5] = (DCT_INT) DESCALE(tmp2 + tmp11 + tmp12,
                       CONST_BITS+PASS1_BITS+1);
    wsptr[DCTSIZE*7] = (DCT_INT) DESCALE(tmp3 + tmp10 + tmp13,
                       CONST_BITS+PASS1_BITS+1);

    barrier(CLK_LOCAL_MEM_FENCE);
    wsptr = workspace;
    wsptr += ctr * DCTSIZE;
    quantptr += ctr * DCTSIZE;

    float out[8];
    int j;
    for (j = 0;j < 8;j++)
    {
        if (abs(wsptr[j]) < quantptr[j])
            out[j] = 0;
        else
            out[j] = wsptr[j] / quantptr[j];
    }
    float8 f = (float8)
       (out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    *fdct_out = convert_short8_sat_rte(f);
}


void FDCT_8x16(__local JSAMPLE* samp_buffer,
    __local DCT_INT* q_tbl_ptr,
    short8* fdct_out,
    __local DCT_INT* workspace)
{
    DCT_INT ctr = get_global_id(1);
    DCT_INT tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    DCT_INT tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17;
    DCT_INT z1;
    __local DCT_INT* quantptr;
    __local DCT_INT* wsptr;
    __local DCT_INT* wsptr2;
    __local JSAMPLE* inptr;
    quantptr = q_tbl_ptr;

    inptr = samp_buffer + ctr * DCTSIZE;
    wsptr = workspace + ctr * DCTSIZE ;
    int i;
    for (i = 0; i < 2; i++)
    {
        inptr += i * DCTSIZE2;
        wsptr += i * DCTSIZE2;        

        tmp0 = inptr[0] + inptr[7];
        tmp1 = inptr[1] + inptr[6];
        tmp2 = inptr[2] + inptr[5];
        tmp3 = inptr[3] + inptr[4];

        tmp10 = tmp0 + tmp3;
        tmp12 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp13 = tmp1 - tmp2;

        tmp0 = inptr[0] - inptr[7];
        tmp1 = inptr[1] - inptr[6];
        tmp2 = inptr[2] - inptr[5];
        tmp3 = inptr[3] - inptr[4];

        /* Apply unsigned->signed conversion */
        wsptr[0] = (DCT_INT) ((tmp10 + tmp11 - 8 * CENTERJSAMPLE) << PASS1_BITS);
        wsptr[4] = (DCT_INT) ((tmp10 - tmp11) << PASS1_BITS);

        z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
        wsptr[2] = (DCT_INT) DESCALE(z1 + MULTIPLY(tmp12, FIX_0_765366865),
                       CONST_BITS-PASS1_BITS);
        wsptr[6] = (DCT_INT) DESCALE(z1 - MULTIPLY(tmp13, FIX_1_847759065),
                       CONST_BITS-PASS1_BITS);

        tmp10 = tmp0 + tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp0 + tmp2;
        tmp13 = tmp1 + tmp3;
        z1 = MULTIPLY(tmp12 + tmp13, FIX_1_175875602); /*  c3 */

        tmp0  = MULTIPLY(tmp0,    FIX_1_501321110);    /*  c1+c3-c5-c7 */
        tmp1  = MULTIPLY(tmp1,    FIX_3_072711026);    /*  c1+c3+c5-c7 */
        tmp2  = MULTIPLY(tmp2,    FIX_2_053119869);    /*  c1+c3-c5+c7 */
        tmp3  = MULTIPLY(tmp3,    FIX_0_298631336);    /* -c1+c3+c5-c7 */
        tmp10 = MULTIPLY(tmp10, - FIX_0_899976223);    /*  c7-c3 */
        tmp11 = MULTIPLY(tmp11, - FIX_2_562915447);    /* -c1-c3 */
        tmp12 = MULTIPLY(tmp12, - FIX_0_390180644);    /*  c5-c3 */
        tmp13 = MULTIPLY(tmp13, - FIX_1_961570560);    /* -c3-c5 */

        tmp12 += z1;
        tmp13 += z1;

        wsptr[1] = (DCT_INT) DESCALE(tmp0 + tmp10 + tmp12, CONST_BITS-PASS1_BITS);
        wsptr[3] = (DCT_INT) DESCALE(tmp1 + tmp11 + tmp13, CONST_BITS-PASS1_BITS);
        wsptr[5] = (DCT_INT) DESCALE(tmp2 + tmp11 + tmp12, CONST_BITS-PASS1_BITS);
        wsptr[7] = (DCT_INT) DESCALE(tmp3 + tmp10 + tmp13, CONST_BITS-PASS1_BITS);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    wsptr = workspace + DCTSIZE2 + ctr;
    wsptr2 = workspace + ctr;
 
    tmp0 = wsptr2[DCTSIZE*0] + wsptr[DCTSIZE*7];
    tmp1 = wsptr2[DCTSIZE*1] + wsptr[DCTSIZE*6];
    tmp2 = wsptr2[DCTSIZE*2] + wsptr[DCTSIZE*5];
    tmp3 = wsptr2[DCTSIZE*3] + wsptr[DCTSIZE*4];
    tmp4 = wsptr2[DCTSIZE*4] + wsptr[DCTSIZE*3];
    tmp5 = wsptr2[DCTSIZE*5] + wsptr[DCTSIZE*2];
    tmp6 = wsptr2[DCTSIZE*6] + wsptr[DCTSIZE*1];
    tmp7 = wsptr2[DCTSIZE*7] + wsptr[DCTSIZE*0];

    tmp10 = tmp0 + tmp7;
    tmp14 = tmp0 - tmp7;
    tmp11 = tmp1 + tmp6;
    tmp15 = tmp1 - tmp6;
    tmp12 = tmp2 + tmp5;
    tmp16 = tmp2 - tmp5;
    tmp13 = tmp3 + tmp4;
    tmp17 = tmp3 - tmp4;

    tmp0 = wsptr2[DCTSIZE*0] - wsptr[DCTSIZE*7];
    tmp1 = wsptr2[DCTSIZE*1] - wsptr[DCTSIZE*6];
    tmp2 = wsptr2[DCTSIZE*2] - wsptr[DCTSIZE*5];
    tmp3 = wsptr2[DCTSIZE*3] - wsptr[DCTSIZE*4];
    tmp4 = wsptr2[DCTSIZE*4] - wsptr[DCTSIZE*3];
    tmp5 = wsptr2[DCTSIZE*5] - wsptr[DCTSIZE*2];
    tmp6 = wsptr2[DCTSIZE*6] - wsptr[DCTSIZE*1];
    tmp7 = wsptr2[DCTSIZE*7] - wsptr[DCTSIZE*0];

    wsptr2[DCTSIZE*0] = (DCT_INT)
      DESCALE(tmp10 + tmp11 + tmp12 + tmp13, PASS1_BITS+1);
    wsptr2[DCTSIZE*4] = (DCT_INT)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.306562965)) + /* c4[16] = c2[8] */
          MULTIPLY(tmp11 - tmp12, FIX_0_541196100),   /* c12[16] = c6[8] */
          CONST_BITS+PASS1_BITS+1);

    tmp10 = MULTIPLY(tmp17 - tmp15, FIX(0.275899379)) +   /* c14[16] = c7[8] */
        MULTIPLY(tmp14 - tmp16, FIX(1.387039845));    /* c2[16] = c1[8] */

    wsptr2[DCTSIZE*2] = (DCT_INT)
      DESCALE(tmp10 + MULTIPLY(tmp15, FIX(1.451774982))   /* c6+c14 */
          + MULTIPLY(tmp16, FIX(2.172734804)),        /* c2+c10 */
          CONST_BITS+PASS1_BITS+1);
    wsptr2[DCTSIZE*6] = (DCT_INT)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(0.211164243))   /* c2-c6 */
          - MULTIPLY(tmp17, FIX(1.061594338)),        /* c10+c14 */
          CONST_BITS+PASS1_BITS+1);

    tmp11 = MULTIPLY(tmp0 + tmp1, FIX(1.353318001)) +         /* c3 */
        MULTIPLY(tmp6 - tmp7, FIX(0.410524528));          /* c13 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.247225013)) +         /* c5 */
        MULTIPLY(tmp5 + tmp7, FIX(0.666655658));          /* c11 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(1.093201867)) +         /* c7 */
        MULTIPLY(tmp4 - tmp7, FIX(0.897167586));          /* c9 */
    tmp14 = MULTIPLY(tmp1 + tmp2, FIX(0.138617169)) +         /* c15 */
        MULTIPLY(tmp6 - tmp5, FIX(1.407403738));          /* c1 */
    tmp15 = MULTIPLY(tmp1 + tmp3, - FIX(0.666655658)) +       /* -c11 */
        MULTIPLY(tmp4 + tmp6, - FIX(1.247225013));        /* -c5 */
    tmp16 = MULTIPLY(tmp2 + tmp3, - FIX(1.353318001)) +       /* -c3 */
        MULTIPLY(tmp5 - tmp4, FIX(0.410524528));          /* c13 */
    tmp10 = tmp11 + tmp12 + tmp13 -
        MULTIPLY(tmp0, FIX(2.286341144)) +                /* c7+c5+c3-c1 */
        MULTIPLY(tmp7, FIX(0.779653625));                 /* c15+c13-c11+c9 */
    tmp11 += tmp14 + tmp15 + MULTIPLY(tmp1, FIX(0.071888074)) /* c9-c3-c15+c11 */
         - MULTIPLY(tmp6, FIX(1.663905119));              /* c7+c13+c1-c5 */
    tmp12 += tmp14 + tmp16 - MULTIPLY(tmp2, FIX(1.125726048)) /* c7+c5+c15-c3 */
         + MULTIPLY(tmp5, FIX(1.227391138));              /* c9-c11+c1-c13 */
    tmp13 += tmp15 + tmp16 + MULTIPLY(tmp3, FIX(1.065388962)) /* c15+c3+c11-c7 */
         + MULTIPLY(tmp4, FIX(2.167985692));              /* c1+c13+c5-c9 */

    wsptr2[DCTSIZE*1] = (DCT_INT) DESCALE(tmp10, CONST_BITS+PASS1_BITS+1);
    wsptr2[DCTSIZE*3] = (DCT_INT) DESCALE(tmp11, CONST_BITS+PASS1_BITS+1);
    wsptr2[DCTSIZE*5] = (DCT_INT) DESCALE(tmp12, CONST_BITS+PASS1_BITS+1);
    wsptr2[DCTSIZE*7] = (DCT_INT) DESCALE(tmp13, CONST_BITS+PASS1_BITS+1);
    barrier(CLK_LOCAL_MEM_FENCE);

    wsptr = workspace;
    wsptr += ctr * DCTSIZE;
    quantptr += ctr * DCTSIZE;
    float out[8];
    int j;
    for (j = 0;j < 8;j++)
    {
        if (abs(wsptr[j]) < quantptr[j])
            out[j] = 0;
        else
            out[j] = wsptr[j] / quantptr[j];
    }
    float8 f = (float8)
       (out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    *fdct_out = convert_short8_sat_rte(f);
}


__kernel void fdct16x16_aan(__global struct QuantiTable* quant_table,
    __global JSAMPLE* samp_in,
    __global JCOEF* coef_out,
    int ci,
    int blks_in_row)
{
    int GblockIndex = get_global_id(0);//0~sum of blk_num_round8
    int lineIndex = get_local_id(1);
    int LblkIndex = get_local_id(0);

    /* Load the fdct_int_table from global compInfo to local q_table */
    __global DCT_INT* q_tbl_ptr;
    __local DCT_INT q_table[DCTSIZE2];
    event_t e;
    q_tbl_ptr = quant_table->fdct_table_i[ci];
    e = async_work_group_copy(q_table, q_tbl_ptr, DCTSIZE2, (event_t)0);

    /* Load the 16x16 input blocks to local input_buffer 
       each work_item get 2 line of sample, each line has 16 samples;
    */
    int rowIndex, blk_infront;

    float tmp = (float)GblockIndex / (float)blks_in_row;
    rowIndex = floor(tmp);
    blk_infront = GblockIndex - rowIndex * blks_in_row;

    __global JSAMPLE* samp_in_ptr1;
    __global JSAMPLE* samp_in_ptr2;
    samp_in_ptr1 = samp_in + 
        rowIndex * blks_in_row * DCTSIZE16_16 +
        lineIndex * blks_in_row * DCTSIZE16 * 2 +
        blk_infront * DCTSIZE16;
    samp_in_ptr2 = samp_in_ptr1 + blks_in_row * DCTSIZE16;

    __local JSAMPLE sampspace[DCTSIZE16_16 * 8];
    __local uchar16* sampspace_ptr1;
    __local uchar16* sampspace_ptr2;
    sampspace_ptr1 = (__local uchar16*)(sampspace + LblkIndex * DCTSIZE16_16 + lineIndex * DCTSIZE16 * 2);
    sampspace_ptr2 = (__local uchar16*)
        (sampspace + LblkIndex * DCTSIZE16_16 + lineIndex * DCTSIZE16 * 2 + DCTSIZE16);

    *sampspace_ptr1 = vload16(0, samp_in_ptr1);
    *sampspace_ptr2 = vload16(0, samp_in_ptr2);

    __local JSAMPLE* s_ptr;
    s_ptr = sampspace + LblkIndex * DCTSIZE16_16 ;

    /* Require the Local workspace ,each workspace SIZE is DCTSIZE2*2 */
    __local DCT_INT workspace[DCTSIZE2 * 2 * 8];
    __local DCT_INT* wspc_ptr;
    wspc_ptr = workspace + LblkIndex * DCTSIZE2 * 2;

    /* Determine the out_put_ptr, We put the 8x8 blk one by one*/
    __global JCOEF* coef_out_ptr;
    coef_out_ptr = coef_out + GblockIndex * DCTSIZE2 + lineIndex * DCTSIZE;

    /* wait for q_table loading */
    wait_group_events(1,&e);
    short8 p_data;
    FDCT_16x16(
        s_ptr,
        q_table,
        &p_data,
        wspc_ptr);

    vstore8(p_data, 0, coef_out_ptr); 

}


__kernel void fdct8x8_aan(__global struct QuantiTable* quant_table,
    __global JSAMPLE* samp_in,
    __global JCOEF* coef_out,
    int ci,
    int blks_in_row)
{
    int GblockIndex = get_global_id(0);//0~sum of blk_num_round8
    int lineIndex = get_local_id(1);
    int LblkIndex = get_local_id(0);
    
    /* Load the fdct_table from global compInfo to local q_table */
    __global DCT_FLOAT* q_tbl_ptr;
    __local DCT_FLOAT q_table[DCTSIZE2];
    event_t e;
    q_tbl_ptr = quant_table->fdct_table_f[ci];
    e = async_work_group_copy(q_table, q_tbl_ptr, DCTSIZE2, (event_t)0);

    /* Load the input sample that has been padded to 8x8 block in resize_kernel to Local buffer*/
    int rowIndex, blk_infront;

    float tmp = (float)GblockIndex / (float)blks_in_row;
    rowIndex = floor(tmp);
    blk_infront = GblockIndex - rowIndex * blks_in_row;

    __global JSAMPLE* samp_in_ptr;
    samp_in_ptr = samp_in + 
        rowIndex * blks_in_row * DCTSIZE2 +
        lineIndex * blks_in_row * DCTSIZE +
        blk_infront * DCTSIZE;
    __local JSAMPLE sampspace[DCTSIZE2 * 8];
    __local uchar8* sampspace_ptr;
    sampspace_ptr = (__local uchar8*)(sampspace + LblkIndex * DCTSIZE2 + lineIndex * DCTSIZE);
    *sampspace_ptr = vload8(0, samp_in_ptr);

    /* Require the Local workspace */
    __local FAST_FLOAT workspace[DCTSIZE2 * 8];
    __local FAST_FLOAT* wspc_ptr;
    wspc_ptr = workspace + LblkIndex * DCTSIZE2;

    /* Determine the out_put_ptr, We put the 8x8 blk one by one*/
    __global JCOEF* coef_out_ptr;
    coef_out_ptr = coef_out + GblockIndex * DCTSIZE2 + lineIndex * DCTSIZE;

    /* wait for q_table loading */
    wait_group_events(1,&e);
    short8 p_data;
    __local JSAMPLE* s_ptr;
    s_ptr = sampspace + LblkIndex * DCTSIZE2;
    FDCT_8x8(
        s_ptr,
        q_table,
        &p_data,
        wspc_ptr);

    vstore8(p_data, 0, coef_out_ptr); 
}

/*
 * Perform the forward DCT on a 16x8 sample block.
 *
 * 16-point FDCT in pass 1 (rows), 8-point in pass 2 (columns).
 */
__kernel void fdct16x8_aan(__global struct QuantiTable* quant_table,
    __global JSAMPLE* samp_in,
    __global JCOEF* coef_out,
    int ci,
    int blks_in_row)
{
    int GblockIndex = get_global_id(0);//0~sum of blk_num_round8
    int lineIndex = get_local_id(1);
    int LblkIndex = get_local_id(0);

    /* Load the fdct_int_table from global compInfo to local q_table */
    __global DCT_INT* q_tbl_ptr;
    __local DCT_INT q_table[DCTSIZE2];
    event_t e;
    q_tbl_ptr = quant_table->fdct_table_i[ci];
    e = async_work_group_copy(q_table, q_tbl_ptr, DCTSIZE2, (event_t)0);

    /* Load the 16x8 input blocks to local input_buffer 
       each work_item get 1 line of sample, each line has 16 samples;
    */
    int rowIndex, blk_infront;

    float tmp = (float)GblockIndex / (float)blks_in_row;
    rowIndex = floor(tmp);
    blk_infront = GblockIndex - rowIndex * blks_in_row;

    __global JSAMPLE* samp_in_ptr;
    samp_in_ptr = samp_in + 
        rowIndex * blks_in_row * DCTSIZE16_8 +
        lineIndex * blks_in_row * DCTSIZE16 +
        blk_infront * DCTSIZE16;

    __local JSAMPLE sampspace[DCTSIZE16_8 * 8];
    __local uchar16* sampspace_ptr;
    sampspace_ptr = (__local uchar16*)(sampspace + LblkIndex * DCTSIZE16_8 + lineIndex * DCTSIZE16);

    *sampspace_ptr = vload16(0, samp_in_ptr);

    __local JSAMPLE* s_ptr;
    s_ptr = sampspace + LblkIndex * DCTSIZE16_8;

    /* Require the Local workspace ,each workspace SIZE is DCTSIZE2*2 */
    __local DCT_INT workspace[DCTSIZE2 * 8];
    __local DCT_INT* wspc_ptr;
    wspc_ptr = workspace + LblkIndex * DCTSIZE2;

    /* Determine the out_put_ptr, We put the 8x8 blk one by one*/
    __global JCOEF* coef_out_ptr;
    coef_out_ptr = coef_out + GblockIndex * DCTSIZE2 + lineIndex * DCTSIZE;

    /* wait for q_table loading */
    wait_group_events(1,&e);
    short8 p_data;
    FDCT_16x8(
        s_ptr,
        q_table,
        &p_data,
        wspc_ptr);

    vstore8(p_data, 0, coef_out_ptr); 
}

/*
 * Perform the forward DCT on an 8x16 sample block.
 *
 * 8-point FDCT in pass 1 (rows), 16-point in pass 2 (columns).
 */
__kernel void fdct8x16_aan(__global struct QuantiTable* quant_table,
    __global JSAMPLE* samp_in,
    __global JCOEF* coef_out,
    int ci,
    int blks_in_row)
{
    int GblockIndex = get_global_id(0);//0~sum of blk_num_round8
    int lineIndex = get_local_id(1);
    int LblkIndex = get_local_id(0);

    /* Load the fdct_int_table from global compInfo to local q_table */
    __global DCT_INT* q_tbl_ptr;
    __local DCT_INT q_table[DCTSIZE2];
    event_t e;
    q_tbl_ptr = quant_table->fdct_table_i[ci];
    e = async_work_group_copy(q_table, q_tbl_ptr, DCTSIZE2, (event_t)0);

    /* Load the 8x16 input blocks to local input_buffer 
       each work_item get 2 line of sample, each line has 8 samples;
    */
    int rowIndex, blk_infront;

    float tmp = (float)GblockIndex / (float)blks_in_row;
    rowIndex = floor(tmp);
    blk_infront = GblockIndex - rowIndex * blks_in_row;

    __global JSAMPLE* samp_in_ptr1;
    __global JSAMPLE* samp_in_ptr2;
    samp_in_ptr1 = samp_in + 
        rowIndex * blks_in_row * DCTSIZE16_8 +
        lineIndex * blks_in_row * DCTSIZE * 2 +
        blk_infront * DCTSIZE;
    samp_in_ptr2 = samp_in_ptr1 + blks_in_row * DCTSIZE;

    __local JSAMPLE sampspace[DCTSIZE16_8 * 8];
    __local uchar8* sampspace_ptr1;
    __local uchar8* sampspace_ptr2;
    sampspace_ptr1 = (__local uchar8*)(sampspace + LblkIndex * DCTSIZE16_8 + lineIndex * DCTSIZE * 2);
    sampspace_ptr2 = (__local uchar8*)
        (sampspace + LblkIndex * DCTSIZE16_8 + lineIndex * DCTSIZE * 2 + DCTSIZE);
    *sampspace_ptr1 = vload8(0, samp_in_ptr1);
    *sampspace_ptr2 = vload8(0, samp_in_ptr2);

    __local JSAMPLE* s_ptr;
    s_ptr = sampspace + LblkIndex * DCTSIZE16_8;

    /* Require the Local workspace ,each workspace SIZE is DCTSIZE2*2 */
    __local DCT_INT workspace[DCTSIZE2 * 2 * 8];
    __local DCT_INT* wspc_ptr;
    wspc_ptr = workspace + LblkIndex * DCTSIZE2 * 2;

    /* Determine the out_put_ptr, We put the 8x8 blk one by one*/
    __global JCOEF* coef_out_ptr;
    coef_out_ptr = coef_out + GblockIndex * DCTSIZE2 + lineIndex * DCTSIZE;

    /* wait for q_table loading */
    wait_group_events(1,&e);
    short8 p_data;
    FDCT_8x16(
        s_ptr,
        q_table,
        &p_data,
        wspc_ptr);

    vstore8(p_data, 0, coef_out_ptr); 
}

