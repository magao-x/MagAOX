#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NULLC "<<<nullptr>>>"
#define PCTS(S) ((S) ? (S) : NULLC)

////////////////////////////////////////////////////////////////////////
// Test code
// Build:  cc  -D__TEST_MAIN   strcat_varargs.c   -o strcat_varargs
// Usage:  ./strcat_varargs [--verbose] [--quiet]
//  Exit:  code is 0 if all tests pass
////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv) {
    char* strcat_varargs(char*, size_t, ...);
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
