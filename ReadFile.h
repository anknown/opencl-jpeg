//#pragma once

#include <stddef.h>

size_t read_all_bytes(const char * fileName,void ** mem);
void free_all_bytes(void * mem);
