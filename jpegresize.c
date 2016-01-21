/*
 * jpegresize.c
 */

#include "jpeglib.h"
#include "ocl_transupp.h"

#ifdef OPENCL_SUPPORTED
GLOBAL(int)
jpeg_resize (unsigned char* srcdata, 
    long insize, 
    unsigned char* dstdata, 
    int dst_width,
    int dst_height,
    int quality,
    ocl_pub_env_struct* pub_env, 
    ocl_pri_env_struct* pri_env)
{
  GET_TIME_F(proc_begin);
  struct jpeg_decompress_struct srcinfo;
  struct jpeg_compress_struct dstinfo;
  struct jpeg_error_mgr jsrcerr, jdsterr;
  jpeg_transform_info transformoption; /* image transformation options */
  //copyoption = JCOPYOPT_COMMENTS;
  transformoption.transform = JXFORM_NONE;
  transformoption.perfect = FALSE;
  transformoption.trim = FALSE;
  transformoption.force_grayscale = FALSE;
  transformoption.crop = FALSE;
#ifdef PROGRESS_REPORT
  struct cdjpeg_progress_mgr progress;
#endif
  jvirt_barray_ptr * src_coef_arrays;
  jvirt_barray_ptr * dst_coef_arrays;
  /* Initialize the JPEG decompression object with default error handling. */
  srcinfo.err = jpeg_std_error(&jsrcerr);
  jpeg_create_decompress(&srcinfo);
  /* Initialize the JPEG compression object with default error handling. */
  dstinfo.err = jpeg_std_error(&jdsterr);
  jpeg_create_compress(&dstinfo);
  srcinfo.use_gpu = 1;
  dstinfo.use_gpu = 1;
  /* Now safe to enable signal catcher.
   * Note: we assume only the decompression object will have virtual arrays.
   */
#ifdef NEED_SIGNAL_CATCHER
  enable_signal_catcher((j_common_ptr) &srcinfo);
#endif

  /* Scan command line to find file names.
   * It is convenient to use just one switch-parsing routine, but the switch
   * values read here are mostly ignored; we will rescan the switches after
   * opening the input file.  Also note that most of the switches affect the
   * destination JPEG object, so we parse into that and then copy over what
   * needs to affects the source too.
   */

  jsrcerr.trace_level = jdsterr.trace_level;
  srcinfo.mem->max_memory_to_use = dstinfo.mem->max_memory_to_use;

#ifdef PROGRESS_REPORT
  start_progress_monitor((j_common_ptr) &dstinfo, &progress);
#endif
  srcinfo.resize_output_width = dst_width;
  srcinfo.resize_output_height = dst_height;
  dstinfo.input_quality = quality;
  /* Specify data source for decompression */
  jpeg_mem_src(&srcinfo, srcdata, insize);
  /* Enable saving of extra markers that we want to copy */
  //jcopy_markers_setup(&srcinfo, copyoption);
  /* Read file header */
  (void) jpeg_read_header(&srcinfo, TRUE);
  /* Any space needed by a transform option must be requested before
   * jpeg_read_coefficients so that memory allocation will be done right.
   */
#if TRANSFORMS_SUPPORTED
  /* Fail right away if -perfect is given and transformation is not perfect.
   */
  if (!jtransform_request_workspace(&srcinfo, &transformoption)) {
    fprintf(stderr, "transformation is not perfect\n");
    exit(EXIT_FAILURE);
  }
#endif

  /* Read source file as DCT coefficients */
#ifdef OCL_RESIZE
  ocl_init_channel(&srcinfo, &transformoption, pri_env);
  GET_TIME(srcinfo._dehuff_start);
  src_coef_arrays = ocl_jpeg_read_coefficients(&srcinfo, &dstinfo, pri_env);
  GET_TIME(srcinfo._dehuff_end);
#else
  src_coef_arrays = jpeg_read_coefficients(&srcinfo);
#endif  

  /* Initialize destination compression parameters from source values */
  jpeg_copy_critical_parameters(&srcinfo, &dstinfo);
  /* Adjust destination parameters if required by transform options;
   * also find out which set of coefficient arrays will hold the output.
   */
#if TRANSFORMS_SUPPORTED
  dst_coef_arrays = jtransform_adjust_parameters(&srcinfo, &dstinfo,
                         src_coef_arrays,
                         &transformoption);
#else
  dst_coef_arrays = src_coef_arrays;
#endif

#if 0
  unsigned char* dst=NULL;
  int iLen = 0;
  jpeg_mem_dest(&dstinfo, &dst, &iLen);
#else
  int iLen = 1024*1024*10;
  jpeg_mem_dest(&dstinfo, &dstdata, &iLen);
#endif
  /* Start compressor (note no image data is actually written here) */
  jpeg_write_coefficients(&dstinfo, dst_coef_arrays);
  /* Copy to the output file any extra markers that we want to preserve */
//  jcopy_markers_execute(&srcinfo, &dstinfo, copyoption);
#ifdef OCL_RESIZE
  GET_TIME(srcinfo._GPU_start);
  jtransform_execute_resize(&srcinfo, &dstinfo, src_coef_arrays, &transformoption, pub_env, pri_env);
  GET_TIME(srcinfo._GPU_end);
#endif
  /* Finish compression and release memory */
  GET_TIME_F(cleanup_start);
  jpeg_finish_compress(&dstinfo);
  (void) jpeg_finish_decompress(&srcinfo);
  GET_TIME_F(cleanup_end);
  PRINT_TIME("The Whole process last:",proc_begin,cleanup_end);
  PRINT_TIME("\tThe CPU dehuff last:",srcinfo._dehuff_start,srcinfo._dehuff_end);
  PRINT_TIME("\tThe GPU last:",srcinfo._GPU_start, srcinfo._GPU_end);
  PRINT_TIME("\t\tThe GPU_side proc_info:",srcinfo._init_decode_start, srcinfo._init_decode_end);

  PRINT_LAST("\t\tThe GPU_side Trans_in:",srcinfo._trans_coefin_last);
  PRINT_LAST("\t\tThe GPU_side IDCT:",srcinfo._idct_last);
  PRINT_LAST("\t\tThe GPU_side Resize:",srcinfo._resize_last);
  PRINT_LAST("\t\tThe GPU_side FDCT:",srcinfo._fdct_last);
  PRINT_LAST("\t\tThe GPU_side Trans_out:",srcinfo._trans_coefout_last);

  PRINT_TIME("\tThe CPU enhuff last[include I/O]:",cleanup_start,cleanup_end);
  jpeg_destroy_compress(&dstinfo);
  jpeg_destroy_decompress(&srcinfo);

#ifdef PROGRESS_REPORT
  end_progress_monitor((j_common_ptr) &dstinfo);
#endif
  /* All done. */
  //exit(jsrcerr.num_warnings + jdsterr.num_warnings ?EXIT_WARNING:EXIT_SUCCESS);
  return iLen;         /* suppress no-return-value warnings */
}
#endif

LOCAL(int)
keymatch (char * arg, const char * keyword, int minchars)
/* Case-insensitive matching of (possibly abbreviated) keyword switches. */
/* keyword is the constant keyword (must be lower case already), */
/* minchars is length of minimum legal abbreviation. */
{
  register int ca, ck;
  register int nmatched = 0;

  while ((ca = *arg++) != '\0') {
    if ((ck = *keyword++) == '\0')
      return 0;         /* arg longer than keyword, no good */
    if (isupper(ca))        /* force arg to lcase (assume ck is already) */
      ca = tolower(ca);
    if (ca != ck)
      return 0;         /* no good */
    nmatched++;         /* count matched characters */
  }
  /* reached end of argument; fail if it's too short for unique abbrev */
  if (nmatched < minchars)
    return 0;
  return 1;         /* A-OK */
}

LOCAL(void)
usage(void)
{
	fprintf(stderr, "usage: jpegresize [switches] inputfile\n");
	fprintf(stderr, "Switches:\n");
	fprintf(stderr, " -w width      Assign output file width\n");
	fprintf(stderr, " -h height     Assign output file height\n");
	fprintf(stderr, " -q quality    Assign output file quality\n");
	fprintf(stderr, " -outfile file Assign output file\n");
}

/*
 * The main program.
*/
int 
main(int argc, char **argv) 
{
	int i = 0;
	char * input_file  = NULL;
	char * output_file = NULL;
	int width = 0, height = 0, quality = 0;

    if(2 > argc){
		usage();
        return 1;
    }

	for(i = 1; i < argc;){
		if(keymatch(argv[i], "-w", 2)){
			width = atoi(argv[i + 1]);
			i += 2;
		}else if(keymatch(argv[i], "-h", 2)){
			height = atoi(argv[i + 1]);
			i += 2;
		}else if(keymatch(argv[i], "-q", 2)){
			quality = atoi(argv[i + 1]);
			i += 2;
		}else if(keymatch(argv[i], "-outfile", 8)){
			output_file = argv[i + 1];
			i += 2;
		}else{
			if(*argv[i] != '-'){
				input_file = argv[i];
			}
			++i;
		}
	}

#ifdef DEBUG
	printf("w:%d, h:%d, q:%d, input_file:%s, output_file:%s\n", 
					width, height, quality, input_file, output_file);
#endif

	if(NULL == input_file){
		fprintf(stderr, "Input file not found\n");
		usage();
		return 0;
	}
	if(NULL == output_file){
		fprintf(stderr, "Output file not found\n");
		usage();
		return 0;
	}
	if(width <= 0 && width > MAX_IMAGE_WIDTH){
		fprintf(stderr, "Invalid output pic width\n");
		usage();
		return 0;
	}
	if(height <= 0 && height > MAX_IMAGE_HEIGHT){
		fprintf(stderr, "Invalid output pic height\n");
		usage();
		return 0;
	}
	if(quality <= 0 && quality > 100){
		fprintf(stderr, "Invalid output pic quality\n");
		usage();
		return 0;
	}

    /* pub_env should be global variable */
    ocl_pub_env_struct* pub_env;
    pub_env = (ocl_pub_env_struct*)malloc(sizeof(ocl_pub_env_struct));
    ocl_pub_env_init(pub_env);
    /* pri_env should be thread own */
    ocl_pri_env_struct* pri_env;
    pri_env = (ocl_pri_env_struct*)malloc(sizeof(ocl_pri_env_struct));
    ocl_pri_env_init(pub_env, pri_env);

    unsigned char * input_data = NULL;
    unsigned char * output_data = NULL;

	int input_size  = 0;
	int output_size = 0;

    FILE * input_fd;
	FILE * output_fd;

    input_fd = fopen(input_file,"r");
    if (input_fd == NULL){
        printf("Couldn't open input file: %s\n", input_file);
		return 1;
    }
    fseek(input_fd, 0, SEEK_END);
    input_size = ftell(input_fd);
    input_data = (unsigned char*)malloc(input_size + 1);
    fseek(input_fd, 0, SEEK_SET);
    fread(input_data, input_size + 1, 1, input_fd);
	fclose(input_fd);
	
    /* should malloc dstdata as big as enough */
    output_data = (unsigned char*)malloc(2 * input_size + 1);
    output_size = jpeg_resize(input_data,
		input_size, output_data, width, height, quality, pub_env, pri_env);
#ifdef DEBUG
    printf("output_size = %d\n",output_size);
#endif
    if (output_size <= 0 && output_data == NULL){
		fprintf(stderr, "Oops! resize failed\n");
        return 1;
	}
    output_fd = fopen(output_file, "w");
	if(NULL == output_fd){
        printf("Couldn't open output file: %s\n", output_file);
		return 1;
	}
    fwrite(output_data, output_size, 1, output_fd);
    fclose(output_fd);

    /* 
	 * input_data/output_data/pri_env/pub_env  
	 * should be free by application 
	 * */
    free(input_data);
	input_data = NULL;
    free(output_data);
	output_data = NULL;

    ocl_pri_env_release(pri_env);
    free(pri_env);
	pri_env = NULL;

    ocl_pub_env_release(pub_env);
    free(pub_env);        
	pub_env = NULL;

    return 0;
}

