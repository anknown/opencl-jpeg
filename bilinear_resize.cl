#define DCTSIZE 8
#define DCTSIZE2 64
typedef unsigned char JSAMPLE;
typedef float DCT_FLOAT;

__kernel void bilinear_resize(int src_width,
    int src_height,
    int resize_blks_in_row,
    int resize_blks_out_row,
    int resize_blks_out_height,
    __global JSAMPLE* samp_in,
    __global JSAMPLE* samp_out,
    int add_it)
{

    int dstXindex = get_global_id(0);
    int dstYindex = get_global_id(1);
    int dstWidth = get_global_size(0);
    int dstHeight = get_global_size(1);
    
    float ratioX = src_width * 1.0 / dstWidth;
    float ratioY = src_height * 1.0 / dstHeight;
    float newX, newY;
    float t1, t2, t3, t4;
    int src_x1, src_x2, src_y1, src_y2;
    newY = dstYindex * ratioY;       
    src_y1 = (int)newY;
    src_y2 = (newY + 1) > (src_height - 1) ? (src_height - 1) : (src_y1 + 1);    
    newX = dstXindex * ratioX;
    src_x1 = (int)newX;
    src_x2 = (newX + 1) > (src_width - 1) ? (src_width - 1) : (src_x1 + 1);    

    t1 = fabs(src_x2 - newX);
    t2 = fabs(newX - src_x1);
    t3 = fabs(src_y2 - newY);
    t4 = fabs(newY - src_y1);

/*
    t1 = src_x2 - newX;
    t2 = newX - src_x1;
    t3 = src_y2 - newY;
    t4 = newY - src_y1;
*/
    /* Compute the in&out ptr */
    __global JSAMPLE *out_ptr;

    /* Reserved pad samp 
     * Org: dstWidth * dstHeight 
     * Paded:  resize_blks_in_row * resize_blks_in_height 
     */
    int pad_width, pad_height;

    pad_width = resize_blks_out_row * DCTSIZE;
    pad_height = resize_blks_out_height * DCTSIZE;

    out_ptr = samp_out + dstYindex * pad_width + dstXindex;
    /* Compute the input samp_ptr */
    __global JSAMPLE *in_ptrA, *in_ptrB, *in_ptrC, *in_ptrD;
    in_ptrA = samp_in + src_y1 * resize_blks_in_row * DCTSIZE + src_x1;
    in_ptrB = samp_in + src_y1 * resize_blks_in_row * DCTSIZE + src_x2;
    in_ptrC = samp_in + src_y2 * resize_blks_in_row * DCTSIZE + src_x1;
    in_ptrD = samp_in + src_y2 * resize_blks_in_row * DCTSIZE + src_x2;

    JSAMPLE color_A, color_B, color_C, color_D;
    color_A = *in_ptrA; 
    color_B = *in_ptrB;  
    color_C = *in_ptrC; 
    color_D = *in_ptrD; 
    JSAMPLE result;
    result = t3 * t1 * color_A + t4 * t1 * color_B + t3 * t2 * color_C + t4 * t2 * color_D;
//    result = result == 0 ? (color_A + color_B + color_C + color_D)/4 : result;
    result = (result == 0 ? color_A : result);
    *out_ptr = 0;
    *out_ptr = result;
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (add_it)
    {
    /* expand most right edge */
    int add_width, add_height;
    add_width = pad_width - dstWidth;
    add_height = pad_height - dstHeight;
    if (dstXindex == (dstWidth -1))
    {
        if (add_width)
        {
            int i;
            for(i = 1; i < add_width+1;i++)
            {
                *(out_ptr+i) = result;
            }
        }
    }
    if (dstYindex >= (dstHeight - add_height))
    {
        if (add_height)
        {
            *(out_ptr + pad_width * add_height) = result;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (dstYindex >= (dstHeight-add_height) && dstXindex >= (dstWidth-add_width))
    {
        if (add_height !=0 && add_width != 0 )
        {
            *(out_ptr + pad_width*add_height+add_width) = result;
        }
    }
       
/*       
    if (dstYindex >= (dstHeight - add_height) && dstXindex == 0)
    {
        if (add_height)
        {
            int i;
            for(i = 0;i < pad_width;i++)
            {
                *(out_ptr + pad_width * add_height + i) = *(out_ptr + i);
            }
        }
    }
*/
    }
}
