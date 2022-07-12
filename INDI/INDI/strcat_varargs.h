#ifndef __STRCAT_VARARGS_H__
#define __STRCAT_VARARGS_H__

# define NULLC "<<<nullptr>>>"
# define PCTS(S) S ? S : NULLC

char*
str_deref_envvar(char* input_ptr);
char*
vstrcat_varargs(char* dest_ptr0, size_t dest_length, va_list ap);
char*
strcat_varargs(char* dest_ptr0, size_t dest_length, ...);
#endif/*__STRCAT_VARARGS_H__*/
