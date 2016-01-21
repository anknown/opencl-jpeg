/*
 * jdtrans.c
 *
 * Copyright (C) 1995-1997, Thomas G. Lane.
 * Modified 2000-2009 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains library routines for transcoding decompression,
 * that is, reading raw DCT coefficient arrays from an input JPEG file.
 * The routines in jdapimin.c will also be needed by a transcoder.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"

/* Forward declarations */
LOCAL(void) transdecode_master_selection JPP((j_decompress_ptr cinfo));


/*
 * Read the coefficient arrays from a JPEG file.
 * jpeg_read_header must be completed before calling this.
 *
 * The entire image is read into a set of virtual coefficient-block arrays,
 * one per component.  The return value is a pointer to the array of
 * virtual-array descriptors.  These can be manipulated directly via the
 * JPEG memory manager, or handed off to jpeg_write_coefficients().
 * To release the memory occupied by the virtual arrays, call
 * jpeg_finish_decompress() when done with the data.
 *
 * An alternative usage is to simply obtain access to the coefficient arrays
 * during a buffered-image-mode decompression operation.  This is allowed
 * after any jpeg_finish_output() call.  The arrays can be accessed until
 * jpeg_finish_decompress() is called.  (Note that any call to the library
 * may reposition the arrays, so don't rely on access_virt_barray() results
 * to stay valid across library calls.)
 *
 * Returns NULL if suspended.  This case need be checked only if
 * a suspending data source is used.
 */

#ifdef OPENCL_SUPPORTED
#ifdef OCL_RESIZE
typedef struct {
    JBLOCKARRAY mem_buffer;   /* => the in-memory buffer */
    JDIMENSION rows_in_array; /* total virtual array height */
    JDIMENSION blocksperrow;  /* width of array (and of memory buffer) */
}my_barray_control;
typedef my_barray_control* my_barray_ptr;

typedef float DCT_FLOAT;
LOCAL(void)
send_qtable(j_decompress_ptr srcinfo,
    j_compress_ptr dstinfo,
    ocl_pri_env_struct* pri_env)
{

    struct QuantiTable q_table;
    int i, row, col, ci, qtblno, tblno;
    jpeg_component_info* src_compptr;
    jpeg_component_info* dst_compptr;
    JQUANT_TBL** qtbl_ptr;
    JQUANT_TBL* qtbl;
    DCT_FLOAT *idtbl, *fdtbl;
    int *idtbl_int;
    int *fdtbl_int;
    static const double aanscalefactor[DCTSIZE] = {
      1.0, 1.387039845, 1.306562965, 1.175875602,
      1.0, 0.785694958, 0.541196100, 0.275899379
    };
    /* alloc dstinfo quant_table and set to defaults(2 quant_tbl) */
    dstinfo->input_components = srcinfo->num_components;
    jpeg_set_defaults(dstinfo);
    /* check srcinfo if not equal copy it */
    for (tblno = 0; tblno < NUM_QUANT_TBLS; tblno++)
    {
        if (srcinfo->quant_tbl_ptrs[tblno] != NULL)
        {
            qtbl_ptr = &dstinfo->quant_tbl_ptrs[tblno];
            if (*qtbl_ptr == NULL)
                *qtbl_ptr = jpeg_alloc_quant_table((j_common_ptr)dstinfo);
            MEMCOPY((*qtbl_ptr)->quantval, 
                srcinfo->quant_tbl_ptrs[tblno]->quantval,
                SIZEOF((*qtbl_ptr)->quantval));
            (*qtbl_ptr)->sent_table = FALSE;
        }
    }
    ocl_set_quality_ratings(dstinfo, &dstinfo->input_quality, TRUE);
    i = 0;
    for(ci = 0; ci < srcinfo->comps_in_scan; ci++)
    {
        i = 0;
        src_compptr = &srcinfo->comp_info[ci];
        dst_compptr = &dstinfo->comp_info[ci];
        qtblno = src_compptr->quant_tbl_no;
        qtbl = dstinfo->quant_tbl_ptrs[qtblno];
        idtbl = &(q_table.idct_table_f[ci]);
        fdtbl = &(q_table.fdct_table_f[ci]);
        idtbl_int = &(q_table.idct_table_i[ci]);
        fdtbl_int = &(q_table.fdct_table_i[ci]);
        for (row = 0; row < DCTSIZE; row++) {
            for (col = 0; col < DCTSIZE; col++) {
                idtbl[i] = (DCT_FLOAT)
                    ((double) src_compptr->quant_table->quantval[i] *
                    aanscalefactor[row] * aanscalefactor[col]);
                fdtbl[i] = (DCT_FLOAT)
                    (1.0 / (((double) qtbl->quantval[i] *
                    aanscalefactor[row] * aanscalefactor[col] * 8.0)));
                idtbl_int[i] = (int)(src_compptr->quant_table->quantval[i]);
                fdtbl_int[i] = (int)((qtbl->quantval[i])<<3);
            i++;
            }
        }
    }
   // cl_event event;
    cl_int error_code;
    error_code = clEnqueueWriteBuffer(pri_env->queue_transfer,
        pri_env->quanti_tbl_mem,
        CL_TRUE,
        0,
        sizeof(struct QuantiTable),
        &q_table,
        0,NULL,NULL);
    CHECK_OCL_ERROR(error_code, "Couldn't Write quanti_tbl to GPU");    
  //  PROFILE_EVENT("Write_QuantiTable to GPU:", event, pri_env->queue_transfer);
}


LOCAL(void)
init_channel_paramer(j_decompress_ptr srcinfo)
{
    int ci;
    my_barray_ptr src_coef[MAX_COMPONENT_COUNT];
    int width_in_iMCUs, height_in_iMCUs;
    int width_in_blocks[MAX_COMPONENT_COUNT], height_in_blocks[MAX_COMPONENT_COUNT];
    int max_resize_in_width = 0;
    int max_resize_in_height = 0;
    int max_resize_out_width = 0;
    int max_resize_out_height = 0;
    int h_samp_factor, v_samp_factor;
    /* compute the dct_table for inverse&forword DCT kernel */
    jpeg_component_info* compptr;
    struct OCL_parameter* channel_param;
    /* compute what the dct kernel need */

#if 1
    if (srcinfo->num_components != 1)
    {
    width_in_iMCUs = (JDIMENSION)
      jdiv_round_up((long) srcinfo->resize_output_width,
            (long) srcinfo->max_h_samp_factor * srcinfo->min_DCT_h_scaled_size);
    height_in_iMCUs = (JDIMENSION)
      jdiv_round_up((long) srcinfo->resize_output_height,
            (long) srcinfo->max_v_samp_factor * srcinfo->min_DCT_v_scaled_size);
    }
    else {
    width_in_iMCUs = (JDIMENSION)
      jdiv_round_up((long) srcinfo->resize_output_width,8);
    height_in_iMCUs = (JDIMENSION)
      jdiv_round_up((long) srcinfo->resize_output_height,8);
    }
#else
    width_in_iMCUs = (JDIMENSION)
      jdiv_round_up((long) srcinfo->resize_output_width,
            (long) srcinfo->max_h_samp_factor * srcinfo->min_DCT_h_scaled_size);
    height_in_iMCUs = (JDIMENSION)
      jdiv_round_up((long) srcinfo->resize_output_height,
            (long) srcinfo->max_v_samp_factor * srcinfo->min_DCT_v_scaled_size);

#endif
//    printf("width_in_iMCUs = %d\theight_in_iMCUs = %d\n",width_in_iMCUs,height_in_iMCUs);
    for (ci = 0, compptr = srcinfo->comp_info; ci < srcinfo->num_components; ci++, compptr++)
    {
        src_coef[ci] = (my_barray_ptr) srcinfo->coef->coef_arrays[ci];
        /* compute what the fdct kernel need */
#if 1
        if (srcinfo->num_components != 1)
	{
        h_samp_factor = compptr->h_samp_factor;
        v_samp_factor = compptr->v_samp_factor;
	}else{
	h_samp_factor = 1;
	v_samp_factor = 1;
	}
#else
        h_samp_factor = compptr->h_samp_factor;
        v_samp_factor = compptr->v_samp_factor;

#endif
        width_in_blocks[ci] = width_in_iMCUs * h_samp_factor;
        height_in_blocks[ci] = height_in_iMCUs * v_samp_factor;
        max_resize_in_width =
            max_resize_in_width > src_coef[ci]->blocksperrow ?
            max_resize_in_width : src_coef[ci]->blocksperrow;
        max_resize_in_height =
            max_resize_in_height > src_coef[ci]->rows_in_array ?
            max_resize_in_height : src_coef[ci]->rows_in_array;
        max_resize_out_width =
            max_resize_out_width > width_in_blocks[ci] ?
            max_resize_out_width : width_in_blocks[ci];
        max_resize_out_height =
            max_resize_out_height > height_in_blocks[ci] ?
            max_resize_out_height : height_in_blocks[ci];
    }
    for (ci = 0, compptr = srcinfo->comp_info; ci < srcinfo->num_components; ci++, compptr++)
    {
        channel_param = &(compptr->channel.ocl_param);

        channel_param->fdct_blk_num =
            jround_up(width_in_blocks[ci] * height_in_blocks[ci], 8);
//        printf("Comp[%d]:Fdct_blk_num = %d\twidth_in_blocks = %d\theight_in_blocks =%d\n",ci,channel_param->fdct_blk_num, width_in_blocks[ci],height_in_blocks[ci]);
        channel_param->fdct_coef_out_size = width_in_blocks[ci] * height_in_blocks[ci] * DCTSIZE2 * sizeof(JCOEF);
        /* This is what the idct kernel need */
        channel_param->idct_blks_in_row = src_coef[ci]->blocksperrow;

        /* This is what the resize kernel need */
        channel_param->src_width = compptr->downsampled_width;
        channel_param->src_height = compptr->downsampled_height;
        channel_param->resize_blks_in_row = max_resize_in_width;
//        channel_param->resize_blks_in_height = max_resize_in_height;
        channel_param->resize_blks_out_row = max_resize_out_width;
        channel_param->resize_blks_out_height = max_resize_out_height;

        /* This is what the dct kernel need */
        channel_param->dst_blks_in_row = width_in_blocks[ci];
//        channel_param->dst_blks_in_height = height_in_blocks[ci];
    }

}


GLOBAL(jvirt_barray_ptr *)
ocl_jpeg_read_coefficients (j_decompress_ptr cinfo, 
    j_compress_ptr dstinfo,
    ocl_pri_env_struct* pri_env)
{
  if (cinfo->global_state == DSTATE_READY) {
    /* First call: initialize active modules */
    transdecode_master_selection(cinfo);
    cinfo->global_state = DSTATE_RDCOEFS;
  }

  send_qtable(cinfo, dstinfo, pri_env);
  init_channel_paramer(cinfo);

  if (cinfo->global_state == DSTATE_RDCOEFS) {
    /* Absorb whole file into the coef buffer */
    for (;;) {
      int retcode;
      /* Call progress monitor hook if present */
      if (cinfo->progress != NULL)
    (*cinfo->progress->progress_monitor) ((j_common_ptr) cinfo);
      /* Absorb some more input */
      retcode = (*cinfo->inputctl->consume_input) (cinfo);
      if (retcode == JPEG_SUSPENDED)
    return NULL;
      if (retcode == JPEG_REACHED_EOI)
    break;
      /* Advance progress counter if appropriate */
      if (cinfo->progress != NULL &&
      (retcode == JPEG_ROW_COMPLETED || retcode == JPEG_REACHED_SOS)) {
    if (++cinfo->progress->pass_counter >= cinfo->progress->pass_limit) {
      /* startup underestimated number of scans; ratchet up one scan */
      cinfo->progress->pass_limit += (long) cinfo->total_iMCU_rows;
    }
      }
    }
    /* Set state so that jpeg_finish_decompress does the right thing */
    cinfo->global_state = DSTATE_STOPPING;
  }
  /* At this point we should be in state DSTATE_STOPPING if being used
 *    * standalone, or in state DSTATE_BUFIMAGE if being invoked to get access
 *       * to the coefficients during a full buffered-image-mode decompression.
 *          */
  if ((cinfo->global_state == DSTATE_STOPPING ||
       cinfo->global_state == DSTATE_BUFIMAGE) && cinfo->buffered_image) {
    return cinfo->coef->coef_arrays;
  }
  /* Oops, improper usage */
  ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);
  return NULL;          /* keep compiler happy */
}
#endif
#endif


GLOBAL(jvirt_barray_ptr *)
jpeg_read_coefficients (j_decompress_ptr cinfo)
{
  if (cinfo->global_state == DSTATE_READY) {
    /* First call: initialize active modules */
    transdecode_master_selection(cinfo);
    cinfo->global_state = DSTATE_RDCOEFS;
  }
  if (cinfo->global_state == DSTATE_RDCOEFS) {
    /* Absorb whole file into the coef buffer */
    for (;;) {
      int retcode;
      /* Call progress monitor hook if present */
      if (cinfo->progress != NULL)
    (*cinfo->progress->progress_monitor) ((j_common_ptr) cinfo);
      /* Absorb some more input */
      retcode = (*cinfo->inputctl->consume_input) (cinfo);
      if (retcode == JPEG_SUSPENDED)
    return NULL;
      if (retcode == JPEG_REACHED_EOI)
    break;
      /* Advance progress counter if appropriate */
      if (cinfo->progress != NULL &&
      (retcode == JPEG_ROW_COMPLETED || retcode == JPEG_REACHED_SOS)) {
    if (++cinfo->progress->pass_counter >= cinfo->progress->pass_limit) {
      /* startup underestimated number of scans; ratchet up one scan */
      cinfo->progress->pass_limit += (long) cinfo->total_iMCU_rows;
    }
      }
    }
    /* Set state so that jpeg_finish_decompress does the right thing */
    cinfo->global_state = DSTATE_STOPPING;
  }
  /* At this point we should be in state DSTATE_STOPPING if being used
   * standalone, or in state DSTATE_BUFIMAGE if being invoked to get access
   * to the coefficients during a full buffered-image-mode decompression.
   */
  if ((cinfo->global_state == DSTATE_STOPPING ||
       cinfo->global_state == DSTATE_BUFIMAGE) && cinfo->buffered_image) {
    return cinfo->coef->coef_arrays;
  }
  /* Oops, improper usage */
  ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);
  return NULL;          /* keep compiler happy */
}


/*
 * Master selection of decompression modules for transcoding.
 * This substitutes for jdmaster.c's initialization of the full decompressor.
 */

LOCAL(void)
transdecode_master_selection (j_decompress_ptr cinfo)
{
  /* This is effectively a buffered-image operation. */
  cinfo->buffered_image = TRUE;

  /* Compute output image dimensions and related values. */
  jpeg_core_output_dimensions(cinfo);

  /* Entropy decoding: either Huffman or arithmetic coding. */
  if (cinfo->arith_code)
    jinit_arith_decoder(cinfo);
  else {
    jinit_huff_decoder(cinfo);
  }

  /* Always get a full-image coefficient buffer. */
  jinit_d_coef_controller(cinfo, TRUE);

  /* We can now tell the memory manager to allocate virtual arrays. */
  (*cinfo->mem->realize_virt_arrays) ((j_common_ptr) cinfo);

  /* Initialize input side of decompressor to consume first scan. */
  (*cinfo->inputctl->start_input_pass) (cinfo);

  /* Initialize progress monitoring. */
  if (cinfo->progress != NULL) {
    int nscans;
    /* Estimate number of scans to set pass_limit. */
    if (cinfo->progressive_mode) {
      /* Arbitrarily estimate 2 interleaved DC scans + 3 AC scans/component. */
      nscans = 2 + 3 * cinfo->num_components;
    } else if (cinfo->inputctl->has_multiple_scans) {
      /* For a nonprogressive multiscan file, estimate 1 scan per component. */
      nscans = cinfo->num_components;
    } else {
      nscans = 1;
    }
    cinfo->progress->pass_counter = 0L;
    cinfo->progress->pass_limit = (long) cinfo->total_iMCU_rows * nscans;
    cinfo->progress->completed_passes = 0;
    cinfo->progress->total_passes = 1;
  }
}


#ifdef OPENCL_SUPPORTED
GLOBAL(boolean)
ocl_set_quality_ratings (j_compress_ptr cinfo, int *arg, boolean force_baseline)
/* Process a quality-ratings parameter string, of the form
 *  *     N[,N,...]
 *   * If there are more q-table slots than parameters, the last value is replicated.
 *    */
{
  int val = 75;         /* default value */
  int tblno;
  char ch;
  if (*arg != 0)
    val = *arg;  

//  for (tblno = 0; tblno < NUM_QUANT_TBLS; tblno++) {
//      /* Convert user 0-100 rating to percentage scaling */
//      cinfo->q_scale_factor[tblno] = jpeg_quality_scaling(val);
//  }
      cinfo->q_scale_factor[0] = jpeg_quality_scaling(val);

  zy_jpeg_set_qtables(cinfo, force_baseline);
  return TRUE;
}
#endif
