#ifndef __PIPE_WRAPPER_H__
#define __PIPE_WRAPPER_H__
#include "strcat_varargs.h"
int
pipe_wrapper(int xp[2], char* prefix_path, char* middle_path, char* suffix_path);
#endif/* __PIPE_WRAPPER_H__ */
