/*
 * decode_idct.cl
 * author: xiaoE
 * describe:
 *   This file contains the coefficient IDCT kernel.
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

#define DEQUANTIZE_FLOAT(coef,quantval)  (((FAST_FLOAT) (coef)) * (quantval))
#define DEQUANTIZE(coef,quantval)  (((int) (coef)) * (quantval))
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
    float idct_table_f[MAX_COMPONENT_COUNT][64];
    float fdct_table_f[MAX_COMPONENT_COUNT][64];
    int idct_table_i[MAX_COMPONENT_COUNT][64];
    int fdct_table_i[MAX_COMPONENT_COUNT][64];
};



void iDCT_8x8(__local JCOEF* coef_buffer,
    __local DCT_FLOAT* q_tbl_ptr,
    uchar8* idct_out,
    __local FAST_FLOAT* workspace)
{
    FAST_FLOAT tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    FAST_FLOAT tmp10, tmp11, tmp12, tmp13;
    FAST_FLOAT z5, z10, z11, z12, z13;
    __local DCT_FLOAT* quantptr;
    __local FAST_FLOAT* wsptr;
    __local JCOEF* inptr;
    inptr = coef_buffer;
    int ctr = get_global_id(1);
    wsptr = workspace;
    quantptr = q_tbl_ptr;
  
    inptr += ctr;
    quantptr += ctr;
    wsptr +=ctr;
    if (inptr[DCTSIZE*1] == 0 && inptr[DCTSIZE*2] == 0 &&
       inptr[DCTSIZE*3] == 0 && inptr[DCTSIZE*4] == 0 &&
       inptr[DCTSIZE*5] == 0 && inptr[DCTSIZE*6] == 0 &&
       inptr[DCTSIZE*7] == 0) {
   /* AC terms all zero */
        FAST_FLOAT dcval = DEQUANTIZE_FLOAT(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]);
   
        wsptr[DCTSIZE*0] = dcval;
        wsptr[DCTSIZE*1] = dcval;
        wsptr[DCTSIZE*2] = dcval;
        wsptr[DCTSIZE*3] = dcval;
        wsptr[DCTSIZE*4] = dcval;
        wsptr[DCTSIZE*5] = dcval;
        wsptr[DCTSIZE*6] = dcval;
        wsptr[DCTSIZE*7] = dcval;
    }
    else{
     /* Even part */
        tmp0 = DEQUANTIZE_FLOAT(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]);
        tmp1 = DEQUANTIZE_FLOAT(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]);
        tmp2 = DEQUANTIZE_FLOAT(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]);
        tmp3 = DEQUANTIZE_FLOAT(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]);

        tmp10 = tmp0 + tmp2;    /* phase 3 */
        tmp11 = tmp0 - tmp2;

        tmp13 = tmp1 + tmp3;    /* phases 5-3 */
        tmp12 = (tmp1 - tmp3) * ((FAST_FLOAT) 1.414213562) - tmp13; /* 2*c4 */

        tmp0 = tmp10 + tmp13;   /* phase 2 */
        tmp3 = tmp10 - tmp13;
        tmp1 = tmp11 + tmp12;
        tmp2 = tmp11 - tmp12;
        
        /* Odd part */

        tmp4 = DEQUANTIZE_FLOAT(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]);
        tmp5 = DEQUANTIZE_FLOAT(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]);
        tmp6 = DEQUANTIZE_FLOAT(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]);
        tmp7 = DEQUANTIZE_FLOAT(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]);

        z13 = tmp6 + tmp5;
        z10 = tmp6 - tmp5;
        z11 = tmp4 + tmp7;
        z12 = tmp4 - tmp7;

        tmp7 = z11 + z13;       
        tmp11 = (z11 - z13) * ((FAST_FLOAT) 1.414213562); 

        z5 = (z10 + z12) * ((FAST_FLOAT) 1.847759065); 
        tmp10 = ((FAST_FLOAT) 1.082392200) * z12 - z5; 
        tmp12 = ((FAST_FLOAT) -2.613125930) * z10 + z5;

        tmp6 = tmp12 - tmp7;    
        tmp5 = tmp11 - tmp6;
        tmp4 = tmp10 + tmp5;

        wsptr[DCTSIZE*0] = tmp0 + tmp7;
        wsptr[DCTSIZE*7] = tmp0 - tmp7;
        wsptr[DCTSIZE*1] = tmp1 + tmp6;
        wsptr[DCTSIZE*6] = tmp1 - tmp6;
        wsptr[DCTSIZE*2] = tmp2 + tmp5;
        wsptr[DCTSIZE*5] = tmp2 - tmp5;
        wsptr[DCTSIZE*4] = tmp3 + tmp4;
        wsptr[DCTSIZE*3] = tmp3 - tmp4;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  
    wsptr = workspace;
    wsptr += DCTSIZE*ctr;

    tmp10 = wsptr[0] + wsptr[4];
    tmp11 = wsptr[0] - wsptr[4];
    tmp13 = wsptr[2] + wsptr[6];
    tmp12 = (wsptr[2] - wsptr[6]) * ((FAST_FLOAT) 1.414213562) - tmp13;

    tmp0 = tmp10 + tmp13;
    tmp3 = tmp10 - tmp13;
    tmp1 = tmp11 + tmp12;
    tmp2 = tmp11 - tmp12;

    /* Odd part */

    z13 = wsptr[5] + wsptr[3];
    z10 = wsptr[5] - wsptr[3];
    z11 = wsptr[1] + wsptr[7];
    z12 = wsptr[1] - wsptr[7];

    tmp7 = z11 + z13;
    tmp11 = (z11 - z13) * ((FAST_FLOAT) 1.414213562);

    z5 = (z10 + z12) * ((FAST_FLOAT) 1.847759065); 
    tmp10 = ((FAST_FLOAT) 1.082392200) * z12 - z5; 
    tmp12 = ((FAST_FLOAT) -2.613125930) * z10 + z5;

    tmp6 = tmp12 - tmp7;
    tmp5 = tmp11 - tmp6;
    tmp4 = tmp10 + tmp5;

    uchar8 out;

    out.s0 = clamp((int)DESCALE((int)(tmp0 + tmp7),3)+128,0,255);
    out.s7 = clamp((int)DESCALE((int)(tmp0 - tmp7),3)+128,0,255);
    out.s1 = clamp((int)DESCALE((int)(tmp1 + tmp6),3)+128,0,255);
    out.s6 = clamp((int)DESCALE((int)(tmp1 - tmp6),3)+128,0,255);
    out.s2 = clamp((int)DESCALE((int)(tmp2 + tmp5),3)+128,0,255);
    out.s5 = clamp((int)DESCALE((int)(tmp2 - tmp5),3)+128,0,255);
    out.s4 = clamp((int)DESCALE((int)(tmp3 + tmp4),3)+128,0,255);
    out.s3 = clamp((int)DESCALE((int)(tmp3 - tmp4),3)+128,0,255);
/*

    out.s0 = (uchar)clamp(convert_int((tmp0 + tmp7)*C_norm+128),0,255);
    out.s7 = (uchar)clamp(convert_int((tmp0 - tmp7)*C_norm+128),0,255);
    out.s1 = (uchar)clamp(convert_int((tmp1 + tmp6)*C_norm+128),0,255);
    out.s6 = (uchar)clamp(convert_int((tmp1 - tmp6)*C_norm+128),0,255);
    out.s2 = (uchar)clamp(convert_int((tmp2 + tmp5)*C_norm+128),0,255);
    out.s5 = (uchar)clamp(convert_int((tmp2 - tmp5)*C_norm+128),0,255);
    out.s4 = (uchar)clamp(convert_int((tmp3 + tmp4)*C_norm+128),0,255);
    out.s3 = (uchar)clamp(convert_int((tmp3 - tmp4)*C_norm+128),0,255);
*/
    *idct_out = out;

}

void iDCT_16x16(__global JCOEF* coef_buffer,
    __local int* q_tbl_ptr,
    uchar16* idct_out1,
    uchar16* idct_out2,
    __local int* workspace)

{
    INT32 tmp0, tmp1, tmp2, tmp3, tmp10, tmp11, tmp12, tmp13;
    INT32 tmp20, tmp21, tmp22, tmp23, tmp24, tmp25, tmp26, tmp27;
    INT32 z1, z2, z3, z4;
    __local int* quantptr;
    __local int* wsptr;
    __global JCOEF* inptr;
    inptr = coef_buffer;
    int ctr = get_global_id(1);
    wsptr = workspace;
    quantptr = q_tbl_ptr;

    inptr += ctr;
    quantptr += ctr;
    wsptr +=ctr;

    tmp0 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]);
    tmp0 <<= CONST_BITS;
    tmp0 += 1 << (CONST_BITS-PASS1_BITS-1);

    z1 = DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]);
    tmp1 = MULTIPLY(z1, FIX(1.306562965));      
    tmp2 = MULTIPLY(z1, FIX_0_541196100);       

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;
    tmp12 = tmp0 + tmp2;
    tmp13 = tmp0 - tmp2;

    z1 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]);
    z2 = DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]);
    z3 = z1 - z2;
    z4 = MULTIPLY(z3, FIX(0.275899379));        
    z3 = MULTIPLY(z3, FIX(1.387039845));        

    tmp0 = z3 + MULTIPLY(z2, FIX_2_562915447);  
    tmp1 = z4 + MULTIPLY(z1, FIX_0_899976223);  
    tmp2 = z3 - MULTIPLY(z1, FIX(0.601344887)); 
    tmp3 = z4 - MULTIPLY(z2, FIX(0.509795579)); 

    tmp20 = tmp10 + tmp0;
    tmp27 = tmp10 - tmp0;
    tmp21 = tmp12 + tmp1;
    tmp26 = tmp12 - tmp1;
    tmp22 = tmp13 + tmp2;
    tmp25 = tmp13 - tmp2;
    tmp23 = tmp11 + tmp3;
    tmp24 = tmp11 - tmp3;

        z1 = DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]);
    z2 = DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]);
    z3 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]);
    z4 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]);

    tmp11 = z1 + z3;

    tmp1  = MULTIPLY(z1 + z2, FIX(1.353318001));   
    tmp2  = MULTIPLY(tmp11,   FIX(1.247225013));   
    tmp3  = MULTIPLY(z1 + z4, FIX(1.093201867));   
    tmp10 = MULTIPLY(z1 - z4, FIX(0.897167586));   
    tmp11 = MULTIPLY(tmp11,   FIX(0.666655658));   
    tmp12 = MULTIPLY(z1 - z2, FIX(0.410524528));   
    tmp0  = tmp1 + tmp2 + tmp3 -
            MULTIPLY(z1, FIX(2.286341144));       
    tmp13 = tmp10 + tmp11 + tmp12 -
            MULTIPLY(z1, FIX(1.835730603));       
    z1    = MULTIPLY(z2 + z3, FIX(0.138617169));  
    tmp1  += z1 + MULTIPLY(z2, FIX(0.071888074)); 
    tmp2  += z1 - MULTIPLY(z3, FIX(1.125726048)); 
    z1    = MULTIPLY(z3 - z2, FIX(1.407403738));  
    tmp11 += z1 - MULTIPLY(z3, FIX(0.766367282)); 
    tmp12 += z1 + MULTIPLY(z2, FIX(1.971951411)); 
    z2    += z4;
    z1    = MULTIPLY(z2, - FIX(0.666655658));     
    tmp1  += z1;
    tmp3  += z1 + MULTIPLY(z4, FIX(1.065388962)); 
    z2    = MULTIPLY(z2, - FIX(1.247225013));     
    tmp10 += z2 + MULTIPLY(z4, FIX(3.141271809)); 
    tmp12 += z2;
    z2    = MULTIPLY(z3 + z4, - FIX(1.353318001)); 
    tmp2  += z2;
    tmp3  += z2;
    z2    = MULTIPLY(z4 - z3, FIX(0.410524528));   
    tmp10 += z2;
    tmp11 += z2;

    wsptr[8*0]  = (int) RIGHT_SHIFT(tmp20 + tmp0,  CONST_BITS-PASS1_BITS);
    wsptr[8*15] = (int) RIGHT_SHIFT(tmp20 - tmp0,  CONST_BITS-PASS1_BITS);
    wsptr[8*1]  = (int) RIGHT_SHIFT(tmp21 + tmp1,  CONST_BITS-PASS1_BITS);
    wsptr[8*14] = (int) RIGHT_SHIFT(tmp21 - tmp1,  CONST_BITS-PASS1_BITS);
    wsptr[8*2]  = (int) RIGHT_SHIFT(tmp22 + tmp2,  CONST_BITS-PASS1_BITS);
    wsptr[8*13] = (int) RIGHT_SHIFT(tmp22 - tmp2,  CONST_BITS-PASS1_BITS);
    wsptr[8*3]  = (int) RIGHT_SHIFT(tmp23 + tmp3,  CONST_BITS-PASS1_BITS);
    wsptr[8*12] = (int) RIGHT_SHIFT(tmp23 - tmp3,  CONST_BITS-PASS1_BITS);
    wsptr[8*4]  = (int) RIGHT_SHIFT(tmp24 + tmp10, CONST_BITS-PASS1_BITS);
    wsptr[8*11] = (int) RIGHT_SHIFT(tmp24 - tmp10, CONST_BITS-PASS1_BITS);
    wsptr[8*5]  = (int) RIGHT_SHIFT(tmp25 + tmp11, CONST_BITS-PASS1_BITS);
    wsptr[8*10] = (int) RIGHT_SHIFT(tmp25 - tmp11, CONST_BITS-PASS1_BITS);
    wsptr[8*6]  = (int) RIGHT_SHIFT(tmp26 + tmp12, CONST_BITS-PASS1_BITS);
    wsptr[8*9]  = (int) RIGHT_SHIFT(tmp26 - tmp12, CONST_BITS-PASS1_BITS);
    wsptr[8*7]  = (int) RIGHT_SHIFT(tmp27 + tmp13, CONST_BITS-PASS1_BITS);
    wsptr[8*8]  = (int) RIGHT_SHIFT(tmp27 - tmp13, CONST_BITS-PASS1_BITS);

    barrier(CLK_LOCAL_MEM_FENCE);

    wsptr = workspace;
    wsptr += DCTSIZE* 2 *ctr;
    __local int *wsptr_mid;
    wsptr_mid = workspace;
    wsptr_mid += DCTSIZE*(2*ctr + 1);

    tmp0 = (INT32) wsptr[0] + (ONE << (PASS1_BITS+2));
    tmp0 <<= CONST_BITS;

    z1 = (INT32) wsptr[4];
    tmp1 = MULTIPLY(z1, FIX(1.306562965));      
    tmp2 = MULTIPLY(z1, FIX_0_541196100);      

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;
    tmp12 = tmp0 + tmp2;
    tmp13 = tmp0 - tmp2;

    z1 = (INT32) wsptr[2];
    z2 = (INT32) wsptr[6];
    z3 = z1 - z2;
    z4 = MULTIPLY(z3, FIX(0.275899379));        
    z3 = MULTIPLY(z3, FIX(1.387039845));        

    tmp0 = z3 + MULTIPLY(z2, FIX_2_562915447); 
    tmp1 = z4 + MULTIPLY(z1, FIX_0_899976223); 
    tmp2 = z3 - MULTIPLY(z1, FIX(0.601344887));
    tmp3 = z4 - MULTIPLY(z2, FIX(0.509795579));

    tmp20 = tmp10 + tmp0;
    tmp27 = tmp10 - tmp0;
    tmp21 = tmp12 + tmp1;
    tmp26 = tmp12 - tmp1;
    tmp22 = tmp13 + tmp2;
    tmp25 = tmp13 - tmp2;
    tmp23 = tmp11 + tmp3;
    tmp24 = tmp11 - tmp3;

    z1 = (INT32) wsptr[1];
    z2 = (INT32) wsptr[3];
    z3 = (INT32) wsptr[5];
    z4 = (INT32) wsptr[7];

    tmp11 = z1 + z3;

    tmp1  = MULTIPLY(z1 + z2, FIX(1.353318001));   
    tmp2  = MULTIPLY(tmp11,   FIX(1.247225013));   
    tmp3  = MULTIPLY(z1 + z4, FIX(1.093201867));   
    tmp10 = MULTIPLY(z1 - z4, FIX(0.897167586));   
    tmp11 = MULTIPLY(tmp11,   FIX(0.666655658));   
    tmp12 = MULTIPLY(z1 - z2, FIX(0.410524528));   
    tmp0  = tmp1 + tmp2 + tmp3 -
            MULTIPLY(z1, FIX(2.286341144));      
    tmp13 = tmp10 + tmp11 + tmp12 -
            MULTIPLY(z1, FIX(1.835730603));      
    z1    = MULTIPLY(z2 + z3, FIX(0.138617169)); 
    tmp1  += z1 + MULTIPLY(z2, FIX(0.071888074));
    tmp2  += z1 - MULTIPLY(z3, FIX(1.125726048));
    z1    = MULTIPLY(z3 - z2, FIX(1.407403738)); 
    tmp11 += z1 - MULTIPLY(z3, FIX(0.766367282));
    tmp12 += z1 + MULTIPLY(z2, FIX(1.971951411));
    z2    += z4;
    z1    = MULTIPLY(z2, - FIX(0.666655658));    
    tmp1  += z1;
    tmp3  += z1 + MULTIPLY(z4, FIX(1.065388962));
    z2    = MULTIPLY(z2, - FIX(1.247225013));     
    tmp10 += z2 + MULTIPLY(z4, FIX(3.141271809)); 
    tmp12 += z2;
    z2    = MULTIPLY(z3 + z4, - FIX(1.353318001)); 
    tmp2  += z2;
    tmp3  += z2;
    z2    = MULTIPLY(z4 - z3, FIX(0.410524528));   
    tmp10 += z2;
    tmp11 += z2;


    uchar16 out_data1;
    out_data1.s0 = clamp((int) RIGHT_SHIFT(tmp20 + tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255); 
    out_data1.sF = clamp((int) RIGHT_SHIFT(tmp20 - tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s1 = clamp((int) RIGHT_SHIFT(tmp21 + tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.sE = clamp((int) RIGHT_SHIFT(tmp21 - tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s2 = clamp((int) RIGHT_SHIFT(tmp22 + tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.sD = clamp((int) RIGHT_SHIFT(tmp22 - tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s3 = clamp((int) RIGHT_SHIFT(tmp23 + tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.sC = clamp((int) RIGHT_SHIFT(tmp23 - tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s4 = clamp((int) RIGHT_SHIFT(tmp24 + tmp10,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.sB = clamp((int) RIGHT_SHIFT(tmp24 - tmp10,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s5 = clamp((int) RIGHT_SHIFT(tmp25 + tmp11,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.sA = clamp((int) RIGHT_SHIFT(tmp25 - tmp11,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s6 = clamp((int) RIGHT_SHIFT(tmp26 + tmp12,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s9 = clamp((int) RIGHT_SHIFT(tmp26 - tmp12,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s7 = clamp((int) RIGHT_SHIFT(tmp27 + tmp13,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data1.s8 = clamp((int) RIGHT_SHIFT(tmp27 - tmp13,CONST_BITS+PASS1_BITS+3)+128,0,255);

    tmp0 = (INT32) wsptr_mid[0] + (ONE << (PASS1_BITS+2));
    tmp0 <<= CONST_BITS;

    z1 = (INT32) wsptr_mid[4];
    tmp1 = MULTIPLY(z1, FIX(1.306562965));      
    tmp2 = MULTIPLY(z1, FIX_0_541196100);       

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;
    tmp12 = tmp0 + tmp2;
    tmp13 = tmp0 - tmp2;

    z1 = (INT32) wsptr_mid[2];
    z2 = (INT32) wsptr_mid[6];
    z3 = z1 - z2;
    z4 = MULTIPLY(z3, FIX(0.275899379));        
    z3 = MULTIPLY(z3, FIX(1.387039845));        

    tmp0 = z3 + MULTIPLY(z2, FIX_2_562915447);  
    tmp1 = z4 + MULTIPLY(z1, FIX_0_899976223);  
    tmp2 = z3 - MULTIPLY(z1, FIX(0.601344887)); 
    tmp3 = z4 - MULTIPLY(z2, FIX(0.509795579)); 

    tmp20 = tmp10 + tmp0;
    tmp27 = tmp10 - tmp0;
    tmp21 = tmp12 + tmp1;
    tmp26 = tmp12 - tmp1;
    tmp22 = tmp13 + tmp2;
    tmp25 = tmp13 - tmp2;
    tmp23 = tmp11 + tmp3;
    tmp24 = tmp11 - tmp3;


    z1 = (INT32) wsptr_mid[1];
    z2 = (INT32) wsptr_mid[3];
    z3 = (INT32) wsptr_mid[5];
    z4 = (INT32) wsptr_mid[7];

    tmp11 = z1 + z3;

    tmp1  = MULTIPLY(z1 + z2, FIX(1.353318001));   
    tmp2  = MULTIPLY(tmp11,   FIX(1.247225013));   
    tmp3  = MULTIPLY(z1 + z4, FIX(1.093201867));   
    tmp10 = MULTIPLY(z1 - z4, FIX(0.897167586));   
    tmp11 = MULTIPLY(tmp11,   FIX(0.666655658));   
    tmp12 = MULTIPLY(z1 - z2, FIX(0.410524528));   
    tmp0  = tmp1 + tmp2 + tmp3 -
            MULTIPLY(z1, FIX(2.286341144));       
    tmp13 = tmp10 + tmp11 + tmp12 -
            MULTIPLY(z1, FIX(1.835730603));       
    z1    = MULTIPLY(z2 + z3, FIX(0.138617169));  
    tmp1  += z1 + MULTIPLY(z2, FIX(0.071888074)); 
    tmp2  += z1 - MULTIPLY(z3, FIX(1.125726048)); 
    z1    = MULTIPLY(z3 - z2, FIX(1.407403738));  
    tmp11 += z1 - MULTIPLY(z3, FIX(0.766367282)); 
    tmp12 += z1 + MULTIPLY(z2, FIX(1.971951411)); 
    z2    += z4;
    z1    = MULTIPLY(z2, - FIX(0.666655658));     
    tmp1  += z1;
    tmp3  += z1 + MULTIPLY(z4, FIX(1.065388962)); 
    z2    = MULTIPLY(z2, - FIX(1.247225013));     
    tmp10 += z2 + MULTIPLY(z4, FIX(3.141271809)); 
    tmp12 += z2;
    z2    = MULTIPLY(z3 + z4, - FIX(1.353318001)); 
    tmp2  += z2;
    tmp3  += z2;
    z2    = MULTIPLY(z4 - z3, FIX(0.410524528));   
    tmp10 += z2;
    tmp11 += z2;

    uchar16 out_data2;   
    out_data2.s0 = clamp((int) RIGHT_SHIFT(tmp20 + tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255); 
    out_data2.sF = clamp((int) RIGHT_SHIFT(tmp20 - tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s1 = clamp((int) RIGHT_SHIFT(tmp21 + tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.sE = clamp((int) RIGHT_SHIFT(tmp21 - tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s2 = clamp((int) RIGHT_SHIFT(tmp22 + tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.sD = clamp((int) RIGHT_SHIFT(tmp22 - tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s3 = clamp((int) RIGHT_SHIFT(tmp23 + tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.sC = clamp((int) RIGHT_SHIFT(tmp23 - tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s4 = clamp((int) RIGHT_SHIFT(tmp24 + tmp10,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.sB = clamp((int) RIGHT_SHIFT(tmp24 - tmp10,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s5 = clamp((int) RIGHT_SHIFT(tmp25 + tmp11,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.sA = clamp((int) RIGHT_SHIFT(tmp25 - tmp11,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s6 = clamp((int) RIGHT_SHIFT(tmp26 + tmp12,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s9 = clamp((int) RIGHT_SHIFT(tmp26 - tmp12,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s7 = clamp((int) RIGHT_SHIFT(tmp27 + tmp13,CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data2.s8 = clamp((int) RIGHT_SHIFT(tmp27 - tmp13,CONST_BITS+PASS1_BITS+3)+128,0,255);

    *idct_out1 = out_data1;
    *idct_out2 = out_data2;
}
    

void iDCT_16x8(__global JCOEF* coef_buffer,
    __local int* q_tbl_ptr,
    uchar16* idct_out,
    __local int* workspace)
{
    __local int* quantptr;
    __local int* wsptr;
    __global JCOEF* inptr;
    inptr = coef_buffer;
    int ctr = get_global_id(1);
    wsptr = workspace;
    quantptr = q_tbl_ptr;

    inptr += ctr;
    quantptr += ctr;
    wsptr +=ctr;

    INT32 tmp0, tmp1, tmp2, tmp3, tmp10, tmp11, tmp12, tmp13;
    INT32 tmp20, tmp21, tmp22, tmp23, tmp24, tmp25, tmp26, tmp27;
    INT32 z1, z2, z3, z4;

    if (inptr[DCTSIZE*1] == 0 && inptr[DCTSIZE*2] == 0 &&
       inptr[DCTSIZE*3] == 0 && inptr[DCTSIZE*4] == 0 &&
       inptr[DCTSIZE*5] == 0 && inptr[DCTSIZE*6] == 0 &&
       inptr[DCTSIZE*7] == 0) {

        int dcval = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) << PASS1_BITS;

        wsptr[DCTSIZE*0] = dcval;
        wsptr[DCTSIZE*1] = dcval;
        wsptr[DCTSIZE*2] = dcval;
        wsptr[DCTSIZE*3] = dcval;
        wsptr[DCTSIZE*4] = dcval;
        wsptr[DCTSIZE*5] = dcval;
        wsptr[DCTSIZE*6] = dcval;
        wsptr[DCTSIZE*7] = dcval;
    }else{
        /* Even part: reverse the even part of the forward DCT. */
        /* The rotator is sqrt(2)*c(-6). */

        z2 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]);
        z3 = DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]);

        z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
        tmp2 = z1 + MULTIPLY(z2, FIX_0_765366865);
        tmp3 = z1 - MULTIPLY(z3, FIX_1_847759065);

        z2 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]);
        z3 = DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]);
        z2 <<= CONST_BITS;
        z3 <<= CONST_BITS;
        /* Add fudge factor here for final descale. */
        z2 += ONE << (CONST_BITS-PASS1_BITS-1);

        tmp0 = z2 + z3;
        tmp1 = z2 - z3;

        tmp10 = tmp0 + tmp2;
        tmp13 = tmp0 - tmp2;
        tmp11 = tmp1 + tmp3;
        tmp12 = tmp1 - tmp3;

        /* Odd part per figure 8; the matrix is unitary and hence its
         * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
         */

        tmp0 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]);
        tmp1 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]);
        tmp2 = DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]);
        tmp3 = DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]);

        z2 = tmp0 + tmp2;
        z3 = tmp1 + tmp3;

        z1 = MULTIPLY(z2 + z3, FIX_1_175875602); /* sqrt(2) * c3 */
        z2 = MULTIPLY(z2, - FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
        z3 = MULTIPLY(z3, - FIX_0_390180644); /* sqrt(2) * (c5-c3) */
        z2 += z1;
        z3 += z1;

        z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223); /* sqrt(2) * (c7-c3) */
        tmp0 = MULTIPLY(tmp0, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
        tmp3 = MULTIPLY(tmp3, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
        tmp0 += z1 + z2;
        tmp3 += z1 + z3;

        z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
        tmp1 = MULTIPLY(tmp1, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
        tmp2 = MULTIPLY(tmp2, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
        tmp1 += z1 + z3;
        tmp2 += z1 + z2;

        /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

        wsptr[DCTSIZE*0] = (int) RIGHT_SHIFT(tmp10 + tmp3, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE*7] = (int) RIGHT_SHIFT(tmp10 - tmp3, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE*1] = (int) RIGHT_SHIFT(tmp11 + tmp2, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE*6] = (int) RIGHT_SHIFT(tmp11 - tmp2, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE*2] = (int) RIGHT_SHIFT(tmp12 + tmp1, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE*5] = (int) RIGHT_SHIFT(tmp12 - tmp1, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE*3] = (int) RIGHT_SHIFT(tmp13 + tmp0, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE*4] = (int) RIGHT_SHIFT(tmp13 - tmp0, CONST_BITS-PASS1_BITS);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    wsptr = workspace;
    wsptr += DCTSIZE * ctr;

    tmp0 = (INT32) wsptr[0] + (ONE << (PASS1_BITS+2));
    tmp0 <<= CONST_BITS;

    z1 = (INT32) wsptr[4];
    tmp1 = MULTIPLY(z1, FIX(1.306562965));      /* c4[16] = c2[8] */
    tmp2 = MULTIPLY(z1, FIX_0_541196100);       /* c12[16] = c6[8] */

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;
    tmp12 = tmp0 + tmp2;
    tmp13 = tmp0 - tmp2;

    z1 = (INT32) wsptr[2];
    z2 = (INT32) wsptr[6];
    z3 = z1 - z2;
    z4 = MULTIPLY(z3, FIX(0.275899379));        /* c14[16] = c7[8] */
    z3 = MULTIPLY(z3, FIX(1.387039845));        /* c2[16] = c1[8] */

    tmp0 = z3 + MULTIPLY(z2, FIX_2_562915447);  /* (c6+c2)[16] = (c3+c1)[8] */
    tmp1 = z4 + MULTIPLY(z1, FIX_0_899976223);  /* (c6-c14)[16] = (c3-c7)[8] */
    tmp2 = z3 - MULTIPLY(z1, FIX(0.601344887)); /* (c2-c10)[16] = (c1-c5)[8] */
    tmp3 = z4 - MULTIPLY(z2, FIX(0.509795579)); /* (c10-c14)[16] = (c5-c7)[8] */

    tmp20 = tmp10 + tmp0;
    tmp27 = tmp10 - tmp0;
    tmp21 = tmp12 + tmp1;
    tmp26 = tmp12 - tmp1;
    tmp22 = tmp13 + tmp2;
    tmp25 = tmp13 - tmp2;
    tmp23 = tmp11 + tmp3;
    tmp24 = tmp11 - tmp3;

    z1 = (INT32) wsptr[1];
    z2 = (INT32) wsptr[3];
    z3 = (INT32) wsptr[5];
    z4 = (INT32) wsptr[7];

    tmp11 = z1 + z3;

    tmp1  = MULTIPLY(z1 + z2, FIX(1.353318001));   /* c3 */
    tmp2  = MULTIPLY(tmp11,   FIX(1.247225013));   /* c5 */
    tmp3  = MULTIPLY(z1 + z4, FIX(1.093201867));   /* c7 */
    tmp10 = MULTIPLY(z1 - z4, FIX(0.897167586));   /* c9 */
    tmp11 = MULTIPLY(tmp11,   FIX(0.666655658));   /* c11 */
    tmp12 = MULTIPLY(z1 - z2, FIX(0.410524528));   /* c13 */
    tmp0  = tmp1 + tmp2 + tmp3 -
        MULTIPLY(z1, FIX(2.286341144));        /* c7+c5+c3-c1 */
    tmp13 = tmp10 + tmp11 + tmp12 -
        MULTIPLY(z1, FIX(1.835730603));        /* c9+c11+c13-c15 */
    z1    = MULTIPLY(z2 + z3, FIX(0.138617169));   /* c15 */
    tmp1  += z1 + MULTIPLY(z2, FIX(0.071888074));  /* c9+c11-c3-c15 */
    tmp2  += z1 - MULTIPLY(z3, FIX(1.125726048));  /* c5+c7+c15-c3 */
    z1    = MULTIPLY(z3 - z2, FIX(1.407403738));   /* c1 */
    tmp11 += z1 - MULTIPLY(z3, FIX(0.766367282));  /* c1+c11-c9-c13 */
    tmp12 += z1 + MULTIPLY(z2, FIX(1.971951411));  /* c1+c5+c13-c7 */
    z2    += z4;
    z1    = MULTIPLY(z2, - FIX(0.666655658));      /* -c11 */
    tmp1  += z1;
    tmp3  += z1 + MULTIPLY(z4, FIX(1.065388962));  /* c3+c11+c15-c7 */
    z2    = MULTIPLY(z2, - FIX(1.247225013));      /* -c5 */
    tmp10 += z2 + MULTIPLY(z4, FIX(3.141271809));  /* c1+c5+c9-c13 */
    tmp12 += z2;
    z2    = MULTIPLY(z3 + z4, - FIX(1.353318001)); /* -c3 */
    tmp2  += z2;
    tmp3  += z2;
    z2    = MULTIPLY(z4 - z3, FIX(0.410524528));   /* c13 */
    tmp10 += z2;
    tmp11 += z2;

    uchar16 out_data;
    out_data.s0 = clamp((int) RIGHT_SHIFT(tmp20 + tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.sF = clamp((int) RIGHT_SHIFT(tmp20 - tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s1 = clamp((int) RIGHT_SHIFT(tmp21 + tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.sE = clamp((int) RIGHT_SHIFT(tmp21 - tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s2 = clamp((int) RIGHT_SHIFT(tmp22 + tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.sD = clamp((int) RIGHT_SHIFT(tmp22 - tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s3 = clamp((int) RIGHT_SHIFT(tmp23 + tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.sC = clamp((int) RIGHT_SHIFT(tmp23 - tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s4 = clamp((int) RIGHT_SHIFT(tmp24 + tmp10, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.sB = clamp((int) RIGHT_SHIFT(tmp24 - tmp10, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s5 = clamp((int) RIGHT_SHIFT(tmp25 + tmp11, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.sA = clamp((int) RIGHT_SHIFT(tmp25 - tmp11, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s6 = clamp((int) RIGHT_SHIFT(tmp26 + tmp12, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s9 = clamp((int) RIGHT_SHIFT(tmp26 - tmp12, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s7 = clamp((int) RIGHT_SHIFT(tmp27 + tmp13, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s8 = clamp((int) RIGHT_SHIFT(tmp27 - tmp13, CONST_BITS+PASS1_BITS+3)+128,0,255);

    *idct_out = out_data;
}

void iDCT_8x16(__global JCOEF* coef_buffer,
    __local int* q_tbl_ptr,
    uchar8* idct_out1,
    uchar8* idct_out2,
    __local int* workspace)
{
    __local int* quantptr;
    __local int* wsptr;
    __global JCOEF* inptr;
    inptr = coef_buffer;
    int ctr = get_global_id(1);
    wsptr = workspace;
    quantptr = q_tbl_ptr;

    inptr += ctr;
    quantptr += ctr;
    wsptr +=ctr;

    INT32 tmp0, tmp1, tmp2, tmp3, tmp10, tmp11, tmp12, tmp13;
    INT32 tmp20, tmp21, tmp22, tmp23, tmp24, tmp25, tmp26, tmp27;
    INT32 z1, z2, z3, z4;

    /* Even part */
    tmp0 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]);
    tmp0 <<= CONST_BITS;
    /* Add fudge factor here for final descale. */
    tmp0 += ONE << (CONST_BITS-PASS1_BITS-1);

    z1 = DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]);
    tmp1 = MULTIPLY(z1, FIX(1.306562965));      /* c4[16] = c2[8] */
    tmp2 = MULTIPLY(z1, FIX_0_541196100);       /* c12[16] = c6[8] */

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;
    tmp12 = tmp0 + tmp2;
    tmp13 = tmp0 - tmp2;

    z1 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]);
    z2 = DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]);
    z3 = z1 - z2;
    z4 = MULTIPLY(z3, FIX(0.275899379));        /* c14[16] = c7[8] */
    z3 = MULTIPLY(z3, FIX(1.387039845));        /* c2[16] = c1[8] */

    tmp0 = z3 + MULTIPLY(z2, FIX_2_562915447);  /* (c6+c2)[16] = (c3+c1)[8] */
    tmp1 = z4 + MULTIPLY(z1, FIX_0_899976223);  /* (c6-c14)[16] = (c3-c7)[8] */
    tmp2 = z3 - MULTIPLY(z1, FIX(0.601344887)); /* (c2-c10)[16] = (c1-c5)[8] */
    tmp3 = z4 - MULTIPLY(z2, FIX(0.509795579)); /* (c10-c14)[16] = (c5-c7)[8] */

    tmp20 = tmp10 + tmp0;
    tmp27 = tmp10 - tmp0;
    tmp21 = tmp12 + tmp1;
    tmp26 = tmp12 - tmp1;
    tmp22 = tmp13 + tmp2;
    tmp25 = tmp13 - tmp2;
    tmp23 = tmp11 + tmp3;
    tmp24 = tmp11 - tmp3;

    /* Odd part */
    z1 = DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]);
    z2 = DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]);
    z3 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]);
    z4 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]);

    tmp11 = z1 + z3;

    tmp1  = MULTIPLY(z1 + z2, FIX(1.353318001));   /* c3 */
    tmp2  = MULTIPLY(tmp11,   FIX(1.247225013));   /* c5 */
    tmp3  = MULTIPLY(z1 + z4, FIX(1.093201867));   /* c7 */
    tmp10 = MULTIPLY(z1 - z4, FIX(0.897167586));   /* c9 */
    tmp11 = MULTIPLY(tmp11,   FIX(0.666655658));   /* c11 */
    tmp12 = MULTIPLY(z1 - z2, FIX(0.410524528));   /* c13 */
    tmp0  = tmp1 + tmp2 + tmp3 -
        MULTIPLY(z1, FIX(2.286341144));        /* c7+c5+c3-c1 */
    tmp13 = tmp10 + tmp11 + tmp12 -
        MULTIPLY(z1, FIX(1.835730603));        /* c9+c11+c13-c15 */
    z1    = MULTIPLY(z2 + z3, FIX(0.138617169));   /* c15 */
    tmp1  += z1 + MULTIPLY(z2, FIX(0.071888074));  /* c9+c11-c3-c15 */
    tmp2  += z1 - MULTIPLY(z3, FIX(1.125726048));  /* c5+c7+c15-c3 */
    z1    = MULTIPLY(z3 - z2, FIX(1.407403738));   /* c1 */
    tmp11 += z1 - MULTIPLY(z3, FIX(0.766367282));  /* c1+c11-c9-c13 */
    tmp12 += z1 + MULTIPLY(z2, FIX(1.971951411));  /* c1+c5+c13-c7 */
    z2    += z4;
    z1    = MULTIPLY(z2, - FIX(0.666655658));      /* -c11 */
    tmp1  += z1;
    tmp3  += z1 + MULTIPLY(z4, FIX(1.065388962));  /* c3+c11+c15-c7 */
    z2    = MULTIPLY(z2, - FIX(1.247225013));      /* -c5 */
    tmp10 += z2 + MULTIPLY(z4, FIX(3.141271809));  /* c1+c5+c9-c13 */
    tmp12 += z2;
    z2    = MULTIPLY(z3 + z4, - FIX(1.353318001)); /* -c3 */
    tmp2  += z2;
    tmp3  += z2;
    z2    = MULTIPLY(z4 - z3, FIX(0.410524528));   /* c13 */
    tmp10 += z2;
    tmp11 += z2;

    wsptr[8*0]  = (int) RIGHT_SHIFT(tmp20 + tmp0,  CONST_BITS-PASS1_BITS);
    wsptr[8*15] = (int) RIGHT_SHIFT(tmp20 - tmp0,  CONST_BITS-PASS1_BITS);
    wsptr[8*1]  = (int) RIGHT_SHIFT(tmp21 + tmp1,  CONST_BITS-PASS1_BITS);
    wsptr[8*14] = (int) RIGHT_SHIFT(tmp21 - tmp1,  CONST_BITS-PASS1_BITS);
    wsptr[8*2]  = (int) RIGHT_SHIFT(tmp22 + tmp2,  CONST_BITS-PASS1_BITS);
    wsptr[8*13] = (int) RIGHT_SHIFT(tmp22 - tmp2,  CONST_BITS-PASS1_BITS);
    wsptr[8*3]  = (int) RIGHT_SHIFT(tmp23 + tmp3,  CONST_BITS-PASS1_BITS);
    wsptr[8*12] = (int) RIGHT_SHIFT(tmp23 - tmp3,  CONST_BITS-PASS1_BITS);
    wsptr[8*4]  = (int) RIGHT_SHIFT(tmp24 + tmp10, CONST_BITS-PASS1_BITS);
    wsptr[8*11] = (int) RIGHT_SHIFT(tmp24 - tmp10, CONST_BITS-PASS1_BITS);
    wsptr[8*5]  = (int) RIGHT_SHIFT(tmp25 + tmp11, CONST_BITS-PASS1_BITS);
    wsptr[8*10] = (int) RIGHT_SHIFT(tmp25 - tmp11, CONST_BITS-PASS1_BITS);
    wsptr[8*6]  = (int) RIGHT_SHIFT(tmp26 + tmp12, CONST_BITS-PASS1_BITS);
    wsptr[8*9]  = (int) RIGHT_SHIFT(tmp26 - tmp12, CONST_BITS-PASS1_BITS);
    wsptr[8*7]  = (int) RIGHT_SHIFT(tmp27 + tmp13, CONST_BITS-PASS1_BITS);
    wsptr[8*8]  = (int) RIGHT_SHIFT(tmp27 - tmp13, CONST_BITS-PASS1_BITS);

    barrier(CLK_LOCAL_MEM_FENCE);
    wsptr = workspace;
    wsptr += DCTSIZE * 2 *ctr;
    __local int *wsptr_mid;
    wsptr_mid = workspace;
    wsptr_mid += DCTSIZE *(2 * ctr + 1);

    z2 = (INT32) wsptr[2];
    z3 = (INT32) wsptr[6];

    z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
    tmp2 = z1 + MULTIPLY(z2, FIX_0_765366865);
    tmp3 = z1 - MULTIPLY(z3, FIX_1_847759065);

    /* Add fudge factor here for final descale. */
    z2 = (INT32) wsptr[0] + (ONE << (PASS1_BITS+2));
    z3 = (INT32) wsptr[4];

    tmp0 = (z2 + z3) << CONST_BITS;
    tmp1 = (z2 - z3) << CONST_BITS;

    tmp10 = tmp0 + tmp2;
    tmp13 = tmp0 - tmp2;
    tmp11 = tmp1 + tmp3;
    tmp12 = tmp1 - tmp3;

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    tmp0 = (INT32) wsptr[7];
    tmp1 = (INT32) wsptr[5];
    tmp2 = (INT32) wsptr[3];
    tmp3 = (INT32) wsptr[1];

    z2 = tmp0 + tmp2;
    z3 = tmp1 + tmp3;

    z1 = MULTIPLY(z2 + z3, FIX_1_175875602); /* sqrt(2) * c3 */
    z2 = MULTIPLY(z2, - FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
    z3 = MULTIPLY(z3, - FIX_0_390180644); /* sqrt(2) * (c5-c3) */
    z2 += z1;
    z3 += z1;

    z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223); /* sqrt(2) * (c7-c3) */
    tmp0 = MULTIPLY(tmp0, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
    tmp3 = MULTIPLY(tmp3, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
    tmp0 += z1 + z2;
    tmp3 += z1 + z3;

    z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
    tmp1 = MULTIPLY(tmp1, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
    tmp2 = MULTIPLY(tmp2, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
    tmp1 += z1 + z3;
    tmp2 += z1 + z2;

    /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

    uchar8 out_data;
    out_data.s0 = clamp((int) RIGHT_SHIFT(tmp10 + tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s7 = clamp((int) RIGHT_SHIFT(tmp10 - tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s1 = clamp((int) RIGHT_SHIFT(tmp11 + tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s6 = clamp((int) RIGHT_SHIFT(tmp11 - tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s2 = clamp((int) RIGHT_SHIFT(tmp12 + tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s5 = clamp((int) RIGHT_SHIFT(tmp12 - tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s3 = clamp((int) RIGHT_SHIFT(tmp13 + tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s4 = clamp((int) RIGHT_SHIFT(tmp13 - tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255);
    *idct_out1 = out_data;

    z2 = (INT32) wsptr_mid[2];
    z3 = (INT32) wsptr_mid[6];

    z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
    tmp2 = z1 + MULTIPLY(z2, FIX_0_765366865);
    tmp3 = z1 - MULTIPLY(z3, FIX_1_847759065);

    /* Add fudge factor here for final descale. */
    z2 = (INT32) wsptr_mid[0] + (ONE << (PASS1_BITS+2));
    z3 = (INT32) wsptr_mid[4];

    tmp0 = (z2 + z3) << CONST_BITS;
    tmp1 = (z2 - z3) << CONST_BITS;

    tmp10 = tmp0 + tmp2;
    tmp13 = tmp0 - tmp2;
    tmp11 = tmp1 + tmp3;
    tmp12 = tmp1 - tmp3;

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    tmp0 = (INT32) wsptr_mid[7];
    tmp1 = (INT32) wsptr_mid[5];
    tmp2 = (INT32) wsptr_mid[3];
    tmp3 = (INT32) wsptr_mid[1];

    z2 = tmp0 + tmp2;
    z3 = tmp1 + tmp3;

    z1 = MULTIPLY(z2 + z3, FIX_1_175875602); /* sqrt(2) * c3 */
    z2 = MULTIPLY(z2, - FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
    z3 = MULTIPLY(z3, - FIX_0_390180644); /* sqrt(2) * (c5-c3) */
    z2 += z1;
    z3 += z1;

    z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223); /* sqrt(2) * (c7-c3) */
    tmp0 = MULTIPLY(tmp0, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
    tmp3 = MULTIPLY(tmp3, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
    tmp0 += z1 + z2;
    tmp3 += z1 + z3;

    z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
    tmp1 = MULTIPLY(tmp1, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
    tmp2 = MULTIPLY(tmp2, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
    tmp1 += z1 + z3;
    tmp2 += z1 + z2;

    /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */
    out_data.s0 = clamp((int) RIGHT_SHIFT(tmp10 + tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s7 = clamp((int) RIGHT_SHIFT(tmp10 - tmp3, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s1 = clamp((int) RIGHT_SHIFT(tmp11 + tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s6 = clamp((int) RIGHT_SHIFT(tmp11 - tmp2, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s2 = clamp((int) RIGHT_SHIFT(tmp12 + tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s5 = clamp((int) RIGHT_SHIFT(tmp12 - tmp1, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s3 = clamp((int) RIGHT_SHIFT(tmp13 + tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255);
    out_data.s4 = clamp((int) RIGHT_SHIFT(tmp13 - tmp0, CONST_BITS+PASS1_BITS+3)+128,0,255);
    *idct_out2 = out_data;
}

#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.  
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.  
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.  
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.  
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.  
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.  
#define C_norm 0.3535533905932737f // 1 / (8^0.5)

void IDCTvector(__local JCOEF *Vect0, int Step)
{
    __local JCOEF *Vect1 = Vect0 + Step;
    __local JCOEF *Vect2 = Vect1 + Step;
    __local JCOEF *Vect3 = Vect2 + Step;
    __local JCOEF *Vect4 = Vect3 + Step;
    __local JCOEF *Vect5 = Vect4 + Step;
    __local JCOEF *Vect6 = Vect5 + Step;
    __local JCOEF *Vect7 = Vect6 + Step;

     float Y04P   = ((*Vect0) + (*Vect4))*1.0;
     float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

     float Y04P2b6ePP = Y04P + Y2b6eP;
     float Y04P2b6ePM = Y04P - Y2b6eP;
     float Y7f1aP3c5dPP = C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
     float Y7a1fM3d5cMP = C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

     float Y04M   = (*Vect0) - (*Vect4);
     float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

     float Y04M2e6bMP = Y04M + Y2e6bM;
     float Y04M2e6bMM = Y04M - Y2e6bM;
     float Y1c7dM3f5aPM = C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
     float Y1d7cP3a5fMM = C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

    (*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    (*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    (*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    (*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    (*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    (*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    (*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    (*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

__kernel void idct8x8_aan(__global struct QuantiTable* quant_table,
    __global JCOEF* coef_in,
    __global JSAMPLE* samp_out,
    int ci,
    int blks_in_row,
    int coef_offset)
{
    int GblockIndex = get_global_id(0);//0~sum of blk_num_round8
    int LblkIndex = get_local_id(0);
    int lineIndex = get_local_id(1);

    int rowIndex;/* rowIndex from the beginning of self-channel blk */
    int blk_infront; /* blk num behind me in single row */
    /* Get coef_in BLOCK_ptr */
    __global JCOEF* sCurrentBlock;
    sCurrentBlock = coef_in + coef_offset * DCTSIZE2 + GblockIndex * DCTSIZE2;
    
    /* Get local wspc_ptr */
    __local FAST_FLOAT workspace[DCTSIZE2 * 8];
    __local FAST_FLOAT* wspc_ptr;
    wspc_ptr = workspace + LblkIndex * DCTSIZE2;

    /* Get the quantitize_table ptr */
    __global DCT_FLOAT* q_tbl_ptr;    
    q_tbl_ptr = quant_table->idct_table_f[ci];

    __local DCT_FLOAT q_table[DCTSIZE2];
    event_t e;
    e = async_work_group_copy(q_table, q_tbl_ptr, DCTSIZE2, (event_t)0);

    float tmp = (float)(GblockIndex) / (float)blks_in_row;
    rowIndex = floor(tmp);
    blk_infront = (GblockIndex) - rowIndex * blks_in_row;

    /* Get samp_out_ptr */

    __global JSAMPLE* samp_out_ptr;
    samp_out_ptr = samp_out + coef_offset * DCTSIZE2 +
        rowIndex * blks_in_row * DCTSIZE2 + 
        lineIndex * blks_in_row * DCTSIZE + 
        blk_infront * DCTSIZE;

    wait_group_events(1, &e);

    __local JCOEF coef[DCTSIZE2 * 8];
    __local JCOEF* coef_ptr;
    coef_ptr = (coef + LblkIndex * DCTSIZE2 + lineIndex * DCTSIZE);
    *(__local short8*)coef_ptr = vload8(0, sCurrentBlock + lineIndex * DCTSIZE);
    coef_ptr = coef + LblkIndex * DCTSIZE2;

    uchar8 p_data;
    iDCT_8x8(coef_ptr,
        q_table,
        &p_data,
        wspc_ptr);
    vstore8(p_data, 0, samp_out_ptr); 

}    


__kernel void idct16x16_aan(__global struct QuantiTable * quant_table,
    __global JCOEF* coef_in,
    __global JSAMPLE* samp_out,
    int ci,
    int blks_in_row,
    int coef_offset)
{
    int GblockIndex = get_global_id(0);//0~sum of blk_num_round8
    int LblkIndex = get_local_id(0);
    int lineIndex = get_local_id(1);

    int rowIndex;/* rowIndex from the beginning of self-channel blk */
    int blk_infront;
    /* Get coef_in BLOCK_ptr */
    __global JCOEF* sCurrentBlock;
    sCurrentBlock = coef_in + coef_offset * DCTSIZE2 + GblockIndex * DCTSIZE2;
    
    /* Get local wspc_ptr */
    __local int workspace[8 * 16 * 8];//every 8-item need a 8*16 workspace,so every group need 8*16*8 
    __local int* wspc_ptr;
    wspc_ptr = workspace + LblkIndex * 8 * 16;

    /* Get the quantitize_table ptr */
    __global int* q_tbl_ptr;    

    __local int q_table[DCTSIZE2];
    event_t e;
    q_tbl_ptr = quant_table->idct_table_i[ci];
    e = async_work_group_copy(q_table, q_tbl_ptr, DCTSIZE2, (event_t)0);

    float tmp = (float)(GblockIndex) / (float)blks_in_row;
    rowIndex = floor(tmp);
    blk_infront = (GblockIndex) - rowIndex * blks_in_row;

    /* Get samp_out_ptr */

    __global JSAMPLE* samp_out_ptr1;
    __global JSAMPLE* samp_out_ptr2;

    samp_out_ptr1 = samp_out + coef_offset * DCTSIZE2 * 4 + 
        rowIndex * blks_in_row * DCTSIZE2 * 4 + 
        lineIndex * blks_in_row * DCTSIZE * 4 + 
        blk_infront * DCTSIZE * 2;
    samp_out_ptr2 = samp_out_ptr1 + blks_in_row * DCTSIZE * 2;

//    samp_out_ptr = samp_out + GblockIndex * DCTSIZE2 + lineIndex * DCTSIZE;   
    wait_group_events(1, &e);
    uchar16 p_data1,p_data2;
    iDCT_16x16((__global JCOEF*)sCurrentBlock,
        q_table,
        &p_data1,
        &p_data2,
        wspc_ptr);
    vstore16(p_data1, 0, samp_out_ptr1); 
    vstore16(p_data2, 0, samp_out_ptr2); 
}    

/*
 * Perform dequantization and inverse DCT on one block of coefficients,
 * producing a 16x8 output block.
 *
 * 8-point IDCT in pass 1 (columns), 16-point in pass 2 (rows).
 */
__kernel void idct16x8_aan(__global struct QuantiTable * quant_table,
    __global JCOEF* coef_in,
    __global JSAMPLE* samp_out,
    int ci,
    int blks_in_row,
    int coef_offset)
{
    int GblockIndex = get_global_id(0);//0~sum of blk_num_round8
    int LblkIndex = get_local_id(0);
    int lineIndex = get_local_id(1);

    int rowIndex;/* rowIndex from the beginning of self-channel blk */
    int blk_infront;
    /* Get coef_in BLOCK_ptr */
    __global JCOEF* sCurrentBlock;
    sCurrentBlock = coef_in + coef_offset * DCTSIZE2 + GblockIndex * DCTSIZE2;
    
    /* Get local wspc_ptr */
    __local int workspace[DCTSIZE2 * 8];//every 8-item need a 8*8 workspace,so every group need 64*8 
    __local int* wspc_ptr;
    wspc_ptr = workspace + LblkIndex * DCTSIZE2;

    /* Get the quantitize_table ptr */
    __global int* q_tbl_ptr;    

    __local int q_table[DCTSIZE2];
    event_t e;
    q_tbl_ptr = quant_table->idct_table_i[ci];
    e = async_work_group_copy(q_table, q_tbl_ptr, DCTSIZE2, (event_t)0);

    float tmp = (float)(GblockIndex) / (float)blks_in_row;
    rowIndex = floor(tmp);
    blk_infront = (GblockIndex) - rowIndex * blks_in_row;

    /* Get samp_out_ptr */

    __global JSAMPLE* samp_out_ptr;

    samp_out_ptr = samp_out + coef_offset * DCTSIZE16_8 + 
        rowIndex * blks_in_row * DCTSIZE16_8 + 
        lineIndex * blks_in_row * DCTSIZE16  + 
        blk_infront * DCTSIZE16;
    wait_group_events(1, &e);
    uchar16 p_data;
    iDCT_16x8((__global JCOEF*)sCurrentBlock,
        q_table,
        &p_data,
        wspc_ptr);
    vstore16(p_data, 0, samp_out_ptr); 
}

/*
 * Perform dequantization and inverse DCT on one block of coefficients,
 * producing a 8x16 output block.
 *
 * 16-point IDCT in pass 1 (columns), 8-point in pass 2 (rows).
 */
__kernel void idct8x16_aan(__global struct QuantiTable * quant_table,
    __global JCOEF* coef_in,
    __global JSAMPLE* samp_out,
    int ci,
    int blks_in_row,
    int coef_offset)
{
    int GblockIndex = get_global_id(0);//0~sum of blk_num_round8
    int LblkIndex = get_local_id(0);
    int lineIndex = get_local_id(1);

    int rowIndex;/* rowIndex from the beginning of self-channel blk */
    int blk_infront;
    /* Get coef_in BLOCK_ptr */
    __global JCOEF* sCurrentBlock;
    sCurrentBlock = coef_in + coef_offset * DCTSIZE2 + GblockIndex * DCTSIZE2;
    
    /* Get local wspc_ptr */
    __local int workspace[DCTSIZE16_8 * 8];//every 8-item need a 16*8 workspace,so every group need 128*8 
    __local int* wspc_ptr;
    wspc_ptr = workspace + LblkIndex * DCTSIZE16_8;

    /* Get the quantitize_table ptr */
    __global int* q_tbl_ptr;    

    __local int q_table[DCTSIZE2];
    event_t e;
    q_tbl_ptr = quant_table->idct_table_i[ci];
    e = async_work_group_copy(q_table, q_tbl_ptr, DCTSIZE2, (event_t)0);

    float tmp = (float)(GblockIndex) / (float)blks_in_row;
    rowIndex = floor(tmp);
    blk_infront = (GblockIndex) - rowIndex * blks_in_row;

    /* Get samp_out_ptr */

    __global JSAMPLE* samp_out_ptr1;
    __global JSAMPLE* samp_out_ptr2;

    samp_out_ptr1 = samp_out + coef_offset * DCTSIZE16_8 + 
        rowIndex * blks_in_row * DCTSIZE16_8 + 
        lineIndex * blks_in_row * DCTSIZE * 2  + 
        blk_infront * DCTSIZE;
    samp_out_ptr2 = samp_out_ptr1 + blks_in_row * DCTSIZE;
    wait_group_events(1, &e);
    uchar8 p_data1;
    uchar8 p_data2;
    iDCT_8x16((__global JCOEF*)sCurrentBlock,
        q_table,
        &p_data1,
        &p_data2,
        wspc_ptr);
    vstore8(p_data1, 0, samp_out_ptr1); 
    vstore8(p_data2, 0, samp_out_ptr2); 
}
