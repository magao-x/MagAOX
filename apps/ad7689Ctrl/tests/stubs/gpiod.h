/** \file gpiod.h
 * \brief Test-only libgpiod v2 API declarations; no real GPIO implementation.
 * \cond AD7689_TEST_STUBS
 */
#ifndef ad7689Ctrl_test_gpiod_h
#define ad7689Ctrl_test_gpiod_h
#include <cstddef>

struct gpiod_chip
{
};
struct gpiod_line_request
{
};
struct gpiod_line_settings
{
};
struct gpiod_line_config
{
};
struct gpiod_request_config
{
};
enum gpiod_line_direction
{
    GPIOD_LINE_DIRECTION_OUTPUT = 3
};
enum gpiod_line_drive
{
    GPIOD_LINE_DRIVE_PUSH_PULL = 1
};
enum gpiod_line_value
{
    GPIOD_LINE_VALUE_INACTIVE = 0,
    GPIOD_LINE_VALUE_ACTIVE   = 1
};

/// Open a fake GPIO chip.
gpiod_chip *gpiod_chip_open( const char *path /**< [in] requested chip */ );
/// Close a fake GPIO chip.
void gpiod_chip_close( gpiod_chip *chip /**< [in] owned chip */ );
/// Allocate fake line settings.
gpiod_line_settings *gpiod_line_settings_new();
/// Free fake line settings.
void gpiod_line_settings_free( gpiod_line_settings *settings /**< [in] owned settings */ );
/// Set fake output direction.
int gpiod_line_settings_set_direction( gpiod_line_settings *settings /**< [in] settings */,
                                       gpiod_line_direction direction /**< [in] requested direction */ );
/// Set fake push-pull drive.
int gpiod_line_settings_set_drive( gpiod_line_settings *settings /**< [in] settings */,
                                   gpiod_line_drive     drive /**< [in] requested drive */ );
/// Set physical output polarity.
void gpiod_line_settings_set_active_low( gpiod_line_settings *settings /**< [in] settings */,
                                         bool                 activeLow /**< [in] requested inversion */ );
/// Set the initial fake output value.
int gpiod_line_settings_set_output_value( gpiod_line_settings *settings /**< [in] settings */,
                                          gpiod_line_value     value /**< [in] initial value */ );
/// Allocate a fake line configuration.
gpiod_line_config *gpiod_line_config_new();
/// Free a fake line configuration.
void gpiod_line_config_free( gpiod_line_config *config /**< [in] owned configuration */ );
/// Add one fake output to the request.
int gpiod_line_config_add_line_settings( gpiod_line_config   *config /**< [in] config */,
                                         const unsigned      *offsets /**< [in] offsets */,
                                         size_t               count /**< [in] number of offsets */,
                                         gpiod_line_settings *settings /**< [in] settings */ );
/// Allocate a fake request configuration.
gpiod_request_config *gpiod_request_config_new();
/// Free a fake request configuration.
void gpiod_request_config_free( gpiod_request_config *config /**< [in] owned config */ );
/// Set the fake request consumer name.
void gpiod_request_config_set_consumer( gpiod_request_config *config /**< [in] config */,
                                        const char           *consumer /**< [in] name */ );
/// Claim the fake line exclusively.
gpiod_line_request *gpiod_chip_request_lines( gpiod_chip           *chip /**< [in] chip */,
                                              gpiod_request_config *request /**< [in] consumer config */,
                                              gpiod_line_config    *config /**< [in] line config */ );
/// Set one fake output and simulate converter EOC when appropriate.
int gpiod_line_request_set_value( gpiod_line_request *request /**< [in] request */,
                                  unsigned            offset /**< [in] chip-local offset */,
                                  gpiod_line_value    value /**< [in] physical level */ );
/// Release the fake request.
void gpiod_line_request_release( gpiod_line_request *request /**< [in] owned request */ );
#endif
/** \endcond */
