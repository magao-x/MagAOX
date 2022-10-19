#ifdef __TEST_MAIN
#include <stdio.h>
#endif /* __TEST_MAIN */
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "strcat_varargs.h"

////////////////////////////////////////////////////////////////////////
// De-reference possible char* ptr to environment variable (envvar) name
// - Returns pointer to either initial char* or to dereferenced envvar
//
// Protocol:  use leading asterisk to indicate presence of envvar name
// e.g.
//
// - *ENVNAME*default_deref\0
//   - Envvar with a default string
//   => return pointer to envvar value if envvar ENVNAME exists
//   OR
//   => return pointer to default_deref if envvar ENVNAME does not exist
//
// - *ENVNAME\0
//   - Envvar without default
//   => return pointer to envvar value if envvar ENVNAME exists
//   OR
//   => use null pointer if envvar ENVNAME does not exist
//
// - **the_rest\0
//   - Non-envvar string that starts with an asterisk
//   - double-asterisk "encodes" single leading asterisk in a non-envvar
//   - if first and second chars of target are *; then return a pointer
//     to *the_rest i.e. to the second *
////////////////////////////////////////////////////////////////////////
#define __TMP_MAX_ENVVAR_LEN 256
char*
str_deref_envvar(char* input_ptr) {

    char envvar_name[__TMP_MAX_ENVVAR_LEN];
    char* default_ptr;
    char* output_ptr;
    size_t L;

    if (!input_ptr || '*' != *input_ptr) { return input_ptr; }
    if ('*' == *(++input_ptr)) { return input_ptr; }
    default_ptr = strchr(input_ptr,'*');
    if (default_ptr) { L = (default_ptr++) - input_ptr; }
    else             { L = strlen(input_ptr); }
    if (__TMP_MAX_ENVVAR_LEN <= L) { return NULL; }
    strncpy(envvar_name, input_ptr, L);
    envvar_name[L] = '\0';
    output_ptr = getenv(envvar_name);
    return output_ptr ? output_ptr : default_ptr;
}
#undef __TMP_MAX_ENVVAR_LEN

////////////////////////////////////////////////////////////////////////
// Multi-string concatenation, with va_list and envvar dereferencing
//
// Roughly equivalent to
//
//   vsnprintf( output_ptr0, dest_length, "%s%s%s...", ap)
//
//   dest_ptr0 - the starting address to which to write the strings
// dest_length - the maximum number of characters to write, including
//               a null terminator
//          ap - va_list arguments from caller, initialized via va_start
//               in caller, all args are character pointers (char*),
//               terminated by NULL
////////////////////////////////////////////////////////////////////////
char*
vstrcat_varargs(char* dest_ptr0, size_t dest_length, va_list ap) {

    char* nextarg_ptr;
    char* dest_ptr = dest_ptr0;
    char* dest_ptr_last = dest_ptr + dest_length;

    while (dest_ptr < dest_ptr_last) {
        nextarg_ptr = va_arg(ap, char*);
        if (!nextarg_ptr) break;
        nextarg_ptr = str_deref_envvar(nextarg_ptr);
        if (!nextarg_ptr) { dest_ptr = dest_ptr_last; break; }
        strncpy(dest_ptr, nextarg_ptr, dest_ptr_last-dest_ptr);
        dest_ptr += strlen(nextarg_ptr);
    }

    if (dest_ptr >= dest_ptr_last) { *dest_ptr0 = '\0'; return NULL; }

    *dest_ptr = '\0';
    return dest_ptr0;
}

////////////////////////////////////////////////////////////////////////
// Multi-string concatenation, with varargs and envvar dereferencing
// - wrapper for vstrcat_varargs above, but with varargs (, ...) here
// - See documentation for vstrcat_varargs above
//
// Roughly equivalent to
//
//   snprintf( output_ptr0, dest_length, "%s%s...", cptr0, cptr1, ...)
////////////////////////////////////////////////////////////////////////
char*
strcat_varargs(char* dest_ptr0, size_t dest_length, ...) {

    char* cptr;
    va_list ap;

    va_start(ap,dest_length);
    cptr = vstrcat_varargs(dest_ptr0, dest_length, ap);
    va_end(ap);
    return cptr;
}

////////////////////////////////////////////////////////////////////////
// Test code
// Build:  cc  -D__TEST_MAIN   strcat_varargs.c   -o strcat_varargs
// Usage:  ./strcat_varargs [--verbose] [--quiet]
//  Exit:  code is 0 if all tests pass
////////////////////////////////////////////////////////////////////////
#ifdef __TEST_MAIN
int
main(int argc, char** argv) {
    char* result;
    char dest[2048];
    char tmp[2048];
    int f = 0;
    int s = 0;
    char* sORf;
    static char* success = { "Success" };
    static char* failure = { "Failure" };
    char* verbose = NULL;
    char* quiet = NULL;
    int iargc;

    for (iargc=0; iargc < argc && !verbose; ++iargc) {
        verbose = !strcmp("--verbose",argv[iargc]) ? argv[iargc] : verbose;
        quiet = !strcmp("--quiet",argv[iargc]) ? argv[iargc] : quiet;
    }

#   define EXPECT_RESULT(EXPECT,ARGS) \
    result = strcat_varargs ARGS ; \
    if (!result || strcmp(dest,EXPECT)) { sORf = failure; ++f; } \
    else { sORf = success ; ++s; } \
    if (verbose || strcmp(sORf,success)) { \
        fprintf(stdout,"  Status:  %s\n  Expect:  '%s'\n     Got:  '%s'\n", sORf, EXPECT, PCTS(result)); \
    }

#   define EXPECT_NULL(ARGS) \
    result = strcat_varargs ARGS ; \
    if (result) { sORf = failure; ++f; } \
    else { sORf = success ; ++s; } \
    if (verbose || strcmp(sORf,success)) { \
        fprintf(stdout,"  Status:  %s\n  Expect:  '%s'\n     Got:  '%s'\n", sORf, NULLC, PCTS(result)); \
    }

    EXPECT_RESULT("a/b/c.d",(dest, 2048, "a/", "b", "/c", ".d", NULL))
    EXPECT_NULL((dest, 7, "a/", "b", "/c", ".d", NULL))

    // Many of the following tests end with the following arguments
    // => Implement the following macro
#   define ABCDN "/a/", "b", "/c", ".d", NULL

    result = getenv("HOME");
    sprintf(tmp,"/.%s/a/b/c.d", result);
    // Envvar without alternate, destination too short by one character
    EXPECT_NULL(
        (dest, strlen(tmp), "/.", "*HOME", ABCDN)
    )
    // Envvar without alternate, adequate destination
    EXPECT_RESULT(tmp,
        (dest, strlen(tmp)+1, "/.", "*HOME", ABCDN)
    )
    // Double-asterisked non-envvar
    EXPECT_RESULT("/./*HOME**home*/a/b/c.d"
        ,(dest, 2048, "/./", "**HOME**home*", ABCDN)
    )
    // Non-existent envvar without alternate
    EXPECT_NULL(
        (dest, 2048, "/./", "*HOME ", ABCDN)
    )
    // Non-existent envvar with alternate
    EXPECT_RESULT("/./*home*/a/b/c.d"
        ,(dest, 2048, "/./", "*HOME **home*", ABCDN)
    )
    // Premature NULL argument
    EXPECT_RESULT("/./a/b"
        ,(dest, 2048, "/./", "a/", "b", NULL, "/c", ".d", ABCDN)
    )

    result = getenv("HOME");
    sprintf(tmp,"/.%s/a/b%s/c.d", result, result);
    // Two envvars
    EXPECT_RESULT(tmp
        ,(dest, strlen(tmp)+1, "/.", "*HOME", "/a/", "b"
                             , "*HOME*home", "/c", ".d", NULL)
    )

    if (!quiet || verbose) {
        fprintf(stdout,"Successes/Failures:  %d/%d\n", s, f);
    }

    return f;
}
#endif /* __TEST_MAIN */
