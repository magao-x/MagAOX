
#ifdef __cplusplus
extern "C"
{
#endif
   
/* encode */
extern int to64frombits(unsigned char *out, const unsigned char *in,
    int inlen);

/* decode */
extern int from64tobits(char *out, const char *in);


#ifdef __cplusplus
} //extern "C"
#endif
   
   
