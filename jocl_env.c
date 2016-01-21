#include <stdio.h>
#include "jpeglib.h"
cl_int create_with_file_name(ocl_pub_env_struct* env, cl_program* prog, const char* file_name)
{
    cl_int error_code;
    FILE *program_handle;
    char *program_buffer;
    size_t program_size;
    program_handle = fopen(file_name, "r");

    if(program_handle == NULL){
        printf("Couldnn't open progfile.\n");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size+1);
    program_buffer[program_size]='\0';
    fread(program_buffer, sizeof(char), program_size,program_handle);
    fclose(program_handle);
    /*
    *prog = clCreateProgramWithBinary(env->context,
                                       1,
                                       &env->device_id,
                                       &program_size,
                                       &program_buffer,
                                       NULL,
                                       &error_code);
    */
    *prog = clCreateProgramWithSource(env->context,
                                       1,
                                       //&env->device_id,
                                       //&program_size,
                                       &program_buffer,
                                       NULL,
                                       &error_code);
    free(program_buffer);
    CHECK_OCL_ERROR(error_code,"Create Program fail");
    if(error_code == CL_SUCCESS)
    {
	    error_code = clBuildProgram(*prog,1,&env->device_id,NULL,NULL,NULL);
    }
    if(error_code != CL_SUCCESS){
	    /*
	    // Determine the size of the log
	    size_t log_size;
	    clGetProgramBuildInfo(*prog, &env->device_id, 
		CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	    // Allocate memory for the log
	    char *log = (char *) malloc(log_size);

	    // Get the log
	    clGetProgramBuildInfo(*prog, &env->device_id, 
		CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

	    // Print the log
	    */
	    printf("what's the fuck: %s err:%d\n",file_name, error_code);
    }
    return error_code;
}

cl_int ocl_mem_init(ocl_mem* cl_mem_member, ocl_pub_env_struct* pub_env)
{
    cl_int error_code;
    /* Create src_coef_mem to hold dehuff's resule
     * Type : short */
    cl_mem_member->src_coef_mem = clCreateBuffer(pub_env->context,
        CL_MEM_READ_ONLY,
        MAX_CHANNEL_COEF_IN_MEM,
        NULL,
        &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create src_coef_mem buf.\n");
    }

    /* Create dst_samp_mem to hold IDCT's resule 
     * Type : uchar */
    cl_mem_member->idct_samp_out_mem = clCreateBuffer(pub_env->context,
        CL_MEM_READ_WRITE,
        MAX_CHANNEL_SAMP_MEM,
        NULL,
        &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create dst_coef_mem buf.\n");
    }
    cl_mem_member->resize_samp_out_mem = clCreateBuffer(pub_env->context,
        CL_MEM_READ_WRITE,
        MAX_CHANNEL_SAMP_MEM,
        NULL,
        &error_code);
    CHECK_OCL_ERROR(error_code, "Couldn't Create resize_samp_out_mem buffer");
    /* Create dst_coef_mem to hold IDCT's resule 
     * Type : short */
    cl_mem_member->dst_coef_mem = clCreateBuffer(pub_env->context,
        CL_MEM_WRITE_ONLY,
        MAX_CHANNEL_COEF_OUT_MEM,
        NULL,
        &error_code);
    CHECK_OCL_ERROR(error_code, "Couldn't Create dst_coef_mem buffer");
    return error_code;     
}

void ocl_pub_env_init(ocl_pub_env_struct* env)
{
    cl_int error_code;
    cl_uint num_devices;
    /* Create platform*/
    if(CL_SUCCESS != (error_code = clGetPlatformIDs(1,&env->platform_id,NULL)) )
    {
        perror("Couldn't find platform\n");
    }

    /* Create device*/
#if 1
    clGetDeviceIDs(env->platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    env->device_list = (cl_device_id*)
        malloc(sizeof(cl_device_id) * num_devices);
    error_code = clGetDeviceIDs(env->platform_id,CL_DEVICE_TYPE_GPU,num_devices,env->device_list,NULL);
	cl_uint max_cu;
	clGetDeviceInfo(env->device_list[0],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&max_cu,NULL);
    env->device_id = env->device_list[0];

#else
    error_code = clGetDeviceIDs(env->platform_id,CL_DEVICE_TYPE_CPU,1,&env->device_id,NULL);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't find GPU devices\n");
    }
#endif
    /* Create context */
    env->context = clCreateContext(NULL,1,&env->device_id,NULL,NULL,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create context\n");
    }
    /* Create queue*/
#if 0
    #ifdef OCL_PROFILE
    env->queue_compute = clCreateCommandQueue(env->context,
            env->device_id,CL_QUEUE_PROFILING_ENABLE,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create Command queue\n");
    }
    env->queue_transfer = clCreateCommandQueue(env->context,
            env->device_id,CL_QUEUE_PROFILING_ENABLE,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create Command queue\n");
    }
    #else
    env->queue_compute = clCreateCommandQueue(env->context,
            env->device_id,(cl_command_queue_properties)NULL,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create Command queue\n");
    }
    env->queue_transfer = clCreateCommandQueue(env->context,
            env->device_id,(cl_command_queue_properties)NULL,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create Command queue\n");
    }
    #endif
#endif
}

void ocl_pri_env_init(ocl_pub_env_struct* pub_env, ocl_pri_env_struct* pri_env)
{
    cl_int error_code;
    int ci;
#if 1
    #ifdef OCL_PROFILE
    pri_env->queue_compute = clCreateCommandQueue(pub_env->context,
            pub_env->device_id,CL_QUEUE_PROFILING_ENABLE,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create Command queue\n");
    }
    pri_env->queue_transfer = clCreateCommandQueue(pub_env->context,
            pub_env->device_id,CL_QUEUE_PROFILING_ENABLE,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create Command queue\n");
    }
    #else
    pri_env->queue_compute = clCreateCommandQueue(pub_env->context,
            pub_env->device_id,(cl_command_queue_properties)NULL,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create Command queue\n");
    }
    pri_env->queue_transfer = clCreateCommandQueue(pub_env->context,
            pub_env->device_id,(cl_command_queue_properties)NULL,&error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create Command queue\n");
    }
    #endif
#else
    pri_env->queue_transfer = pub_env->queue_transfer;
    pri_env->queue_compute = pub_env->queue_compute;
#endif
    /* Create decode_info to hold what we need when we do_idct in GPU 
     * Type : struct DecodeInfo */
    pri_env->quanti_tbl_mem = clCreateBuffer(pub_env->context,
        CL_MEM_READ_ONLY,
        sizeof(struct QuantiTable),
        NULL,
        &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create decode_info buf.\n");
    }

    for (ci = 0;ci < MAX_COMPONENT_COUNT;ci++)
    {
        /* Create Gmem that needed during the whole process */
        if(CL_SUCCESS != ocl_mem_init(&(pri_env->cl_mem_member[ci]), pub_env))
        {
            perror("Init ocl_mem Fail\n");
        }
    }
    /* create IDCT program & build it */
    error_code = create_with_file_name(pub_env, &pri_env->idct_program, IDCT_CLC);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create decode_idct program\n");
    }
    pri_env->idct8x8_kernel = clCreateKernel(pri_env->idct_program, "idct8x8_aan", &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create idct kernel\n");
    }
    pri_env->idct16x16_kernel = clCreateKernel(pri_env->idct_program, "idct16x16_aan", &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create idct kernel\n");
    }
    pri_env->idct16x8_kernel = clCreateKernel(pri_env->idct_program, "idct16x8_aan", &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create idct kernel\n");
    }
    pri_env->idct8x16_kernel = clCreateKernel(pri_env->idct_program, "idct8x16_aan", &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create idct kernel\n");
    }

    /* create Resize program & build it */
    error_code = create_with_file_name(pub_env, &pri_env->resize_program, RESIZE_CLC);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create bilinear_resize program\n");
    }
    pri_env->resize_kernel = clCreateKernel(pri_env->resize_program, "bilinear_resize", &error_code);
    if(error_code != CL_SUCCESS)
    {
        printf("Error_code:%d\n",error_code);
        perror("Couldn't Create resize kernel:%d\n");
    }

    /* create FDCT program & build it */
    error_code = create_with_file_name(pub_env, &pri_env->dct_program, FDCT_CLC);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create bilinear_resize program\n");
    }
    pri_env->fdct8x8_kernel = clCreateKernel(pri_env->dct_program, "fdct8x8_aan", &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create fdct kernel\n");
    }
    pri_env->fdct16x16_kernel = clCreateKernel(pri_env->dct_program, "fdct16x16_aan", &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create fdct kernel\n");
    }
    pri_env->fdct16x8_kernel = clCreateKernel(pri_env->dct_program, "fdct16x8_aan", &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create fdct kernel\n");
    }
    pri_env->fdct8x16_kernel = clCreateKernel(pri_env->dct_program, "fdct8x16_aan", &error_code);
    if(error_code != CL_SUCCESS)
    {
        perror("Couldn't Create fdct kernel\n");
    }
}

void ocl_release_mem(ocl_pri_env_struct* env)
{
    int ci;
    for (ci = 0; ci < MAX_COMPONENT_COUNT;ci++)
    {
        clReleaseMemObject(env->cl_mem_member[ci].src_coef_mem);
        clReleaseMemObject(env->cl_mem_member[ci].idct_samp_out_mem);
        clReleaseMemObject(env->cl_mem_member[ci].resize_samp_out_mem);
        clReleaseMemObject(env->cl_mem_member[ci].dst_coef_mem);
    }
    clReleaseMemObject(env->quanti_tbl_mem);
}
void ocl_pub_env_release(ocl_pub_env_struct* pub_env)
{
    clReleaseContext(pub_env->context);
    clReleaseDevice(pub_env->device_id);
}
void ocl_pri_env_release(ocl_pri_env_struct* pri_env)
{
    ocl_release_mem(pri_env);
    clReleaseKernel(pri_env->idct8x8_kernel);
    clReleaseKernel(pri_env->idct16x16_kernel);
    clReleaseKernel(pri_env->idct8x16_kernel);
    clReleaseKernel(pri_env->idct16x8_kernel);
    clReleaseKernel(pri_env->resize_kernel);
    clReleaseKernel(pri_env->fdct8x8_kernel);
    clReleaseKernel(pri_env->fdct16x16_kernel);
    clReleaseKernel(pri_env->fdct16x8_kernel);
    clReleaseKernel(pri_env->fdct8x16_kernel);
    clReleaseProgram(pri_env->idct_program);
    clReleaseProgram(pri_env->resize_program);
    clReleaseProgram(pri_env->dct_program);
}
