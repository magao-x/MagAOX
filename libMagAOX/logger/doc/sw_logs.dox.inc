The message parameters are:
 - <b>file</b> which should be the __FILE__ macro
 - <b>line</b> which should be the __LINE__ macro
 - <b>code_errno</b> which should be the errno value, or 0 if not relevant 
 - <b>code_other</b> which should a non-errno code if relevant, 0 otherwise.
 - <b>expl which</b> is an explanation of the error.  Can be empty.
 
software_log::messageT constructors are provided with combinations of the parameters.  Some examples:
\code
log<software_error>({__FILE__, __LINE__, errno, retval, "an error occurred"}); //log an error with full details.
\endcode
\code
log<software_error>({__FILE__, __LINE__, errno); //log an error with just errno.  strerror can be used to get the explanation so we don't log it.
\endcode
\code
log<software_info>({__FILE__, __LINE__, "a directory was created"}); //an informational message
\endcode
\code
log<software_debug>({__FILE__, __LINE__}); //a trace 
\endcode
