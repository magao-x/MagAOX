/** \file exceptions.hpp
 * \brief Exception helpers
 */

#ifndef common_exceptions_hpp
#define common_exceptions_hpp

#include <exception>
#include <source_location>
#include <format>

namespace MagAOX
{

/** \defgroup exceptions
 * \ingroup common
 *
 * @{
 */

/// Augments an exception with the source file and line
/**
 * \tparam baseexcept is the base class exception which takes a string as constructor argument
 */
class xwcException : public std::exception
{
  protected:
    std::string m_what; ///< The full what message (message + file information).

    std::string m_message; ///< The explanatory message

    std::source_location m_location;

    int m_code{ 0 }; ///< An error code (optional)

  public:
    xwcException() = delete;

    /// Constructor
    /**
     * The what() message becomes "msg (file line)".
     */
    explicit
    xwcException( const std::string         &msg, /**<[in] the error descriptionat message) */
                  const std::source_location loc = std::source_location::current() )
        : m_what{ std::format( "{} ({} {})", msg, loc.file_name(), loc.line() ) }, m_message{ msg }, m_location{ loc }
    {
    }

    /// Constructor with code
    /**
     * The what() message becomes "msg (file line)".
     */
    xwcException( const std::string         &msg,  /**<[in] the error description (what message) */
                  int                        code, /**<[in] a descriptive error code */
                  const std::source_location loc =
                      std::source_location::current() /**< [in] [optional] the location of this call */ )
        : m_what{ std::format( "{} code: {} ({} {})", msg, code, loc.file_name(), loc.line() ) }, m_message{ msg },
          m_location{ loc }, m_code{ code }
    {
    }

    /// Get the what string
    /** \returns the value of m_what.c_str()
     *
     */
    virtual const char *what() const noexcept
    {
        return m_what.c_str();
    }

    /// Get the message
    /** \returns the value of m_message
     *
     */
    const std::string &message() const
    {
        return m_message;
    }

    /// Get the source file
    /** \returns the value of m_location.file_name()
     *
     */
    const std::string file_name() const
    {
        return m_location.file_name();
    }

    /// Get the source line
    /** \returns the value of m_location.line()
     *
     */
    int line() const
    {
        return m_location.line();
    }

    /// Get the error code
    /** \returns the value of m_code
     *
     */
    int code() const
    {
        return m_code;
    }
};

} // namespace MagAOX

///@}

#endif // common_exceptions_hpp
