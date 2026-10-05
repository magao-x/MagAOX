/** \file logMeta.cpp
 * \brief Declares and defines the logMeta class and related classes.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_files
 *
 */

#include "logMeta.hpp"

// #include "generated/logTypes.hpp"

#include "generated/logMemberAccessor.hpp"

namespace MagAOX
{
namespace logger
{

/*
logMetaDetail logMemberAccessor( flatlogs::eventCodeT ec,
                          const std::string & memberName
                        )
{
   switch(ec)
   {
      case telem_stdcam::eventCode:
         return telem_stdcam::getAccessor(memberName);
      case telem_telcat::eventCode:
         return telem_telcat::getAccessor(memberName);
      case telem_teldata::eventCode:
         return telem_teldata::getAccessor(memberName);
      case telem_telpos::eventCode:
         return telem_telpos::getAccessor(memberName);
      case telem_stage::eventCode:
         return telem_stage::getAccessor(memberName);
      case telem_zaber::eventCode:
         return telem_zaber::getAccessor(memberName);
      case telem_dmspeck::eventCode:
         return telem_dmspeck::getAccessor(memberName);
      case telem_observer::eventCode:
         return telem_observer::getAccessor(memberName);
      case telem_fxngen::eventCode:
         return telem_fxngen::getAccessor(memberName);
      case telem_loopgain::eventCode:
         return telem_loopgain::getAccessor(memberName);
      default:
         std::cerr << "Missing logMemberAccessor case entry for " << ec << ": " << memberName << "\n";
         return logMetaDetail();
   }
}*/

const std::string &logMeta::unavailableValue()
{
    static const std::string value = "NOT AVAILABLE";

    return value;
}

bool verifyLogEntry( flatlogs::eventCodeT ev, char *log )
{
    if( log == nullptr )
    {
        DEBUG_CRUMB( "verifyLogEntry null ev=" + std::to_string( ev ) + " " );
        return false;
    }

    flatlogs::bufferPtrT logBuff( log, []( char * ) {} );
    flatlogs::msgLenT    len = flatlogs::logHeader::msgLen( log );
    bool                 verified{ false };

    switch( ev )
    {
    case eventCodes::GIT_STATE:
        verified = git_state::verify( logBuff, len );
        break;
    case eventCodes::TEXT_LOG:
        verified = text_log::verify( logBuff, len );
        break;
    case eventCodes::USER_LOG:
        verified = user_log::verify( logBuff, len );
        break;
    case eventCodes::STATE_CHANGE:
        verified = state_change::verify( logBuff, len );
        break;
    case eventCodes::SOFTWARE_LOG:
        verified = software_log::verify( logBuff, len );
        break;
    case eventCodes::CONFIG_LOG:
        verified = config_log::verify( logBuff, len );
        break;
    case eventCodes::INDIDRIVER_START:
        verified = indidriver_start::verify( logBuff, len );
        break;
    case eventCodes::INDIDRIVER_STOP:
        verified = indidriver_stop::verify( logBuff, len );
        break;
    case eventCodes::LOOP_CLOSED:
        verified = loop_closed::verify( logBuff, len );
        break;
    case eventCodes::LOOP_PAUSED:
        verified = loop_paused::verify( logBuff, len );
        break;
    case eventCodes::LOOP_OPEN:
        verified = loop_open::verify( logBuff, len );
        break;
    case eventCodes::OBSERVER:
        verified = observer::verify( logBuff, len );
        break;
    case eventCodes::AO_OPERATOR:
        verified = ao_operator::verify( logBuff, len );
        break;
    case eventCodes::PICO_CHANNEL:
        verified = pico_channel::verify( logBuff, len );
        break;
    case eventCodes::OUTLET_STATE:
        verified = outlet_state::verify( logBuff, len );
        break;
    case eventCodes::OUTLET_CHANNEL_STATE:
        verified = outlet_channel_state::verify( logBuff, len );
        break;
    case eventCodes::TELEM_SAVING_STATE:
        verified = telem_saving_state::verify( logBuff, len );
        break;
    case eventCodes::TELEM_FXNGEN:
        verified = telem_fxngen::verify( logBuff, len );
        break;
    case eventCodes::TTMMOD_PARAMS:
        verified = ttmmod_params::verify( logBuff, len );
        break;
    case eventCodes::OCAM_TEMPS:
        verified = ocam_temps::verify( logBuff, len );
        break;
    case eventCodes::CRED2_TEMPS:
        verified = cred2_temps::verify( logBuff, len );
        break;
    case eventCodes::SAVING_START:
        verified = saving_start::verify( logBuff, len );
        break;
    case eventCodes::SAVING_STOP:
        verified = saving_stop::verify( logBuff, len );
        break;
    case eventCodes::TELEM_SAVING:
        verified = telem_saving::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TELPOS:
        verified = telem_telpos::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TELDATA:
        verified = telem_teldata::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TELVANE:
        verified = telem_telvane::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TELENV:
        verified = telem_telenv::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TELCAT:
        verified = telem_telcat::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TELSEE:
        verified = telem_telsee::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TCSI_TIPTILT:
        verified = telem_tcsi_tiptilt::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TCSI_FOCUS:
        verified = telem_tcsi_focus::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TCSI_LABMODE:
        verified = telem_tcsi_labmode::verify( logBuff, len );
        break;
    case eventCodes::TELEM_STAGE:
        verified = telem_stage::verify( logBuff, len );
        break;
    case eventCodes::TELEM_ZABER:
        verified = telem_zaber::verify( logBuff, len );
        break;
    case eventCodes::TELEM_PICO:
        verified = telem_pico::verify( logBuff, len );
        break;
    case eventCodes::TELEM_POSITION:
        verified = telem_position::verify( logBuff, len );
        break;
    case eventCodes::TELEM_PSFACQ:
        verified = telem_psfacq::verify( logBuff, len );
        break;
    case eventCodes::TELEM_POKECENTER:
        verified = telem_pokecenter::verify( logBuff, len );
        break;
    case eventCodes::TELEM_POKELOOP:
        verified = telem_pokeloop::verify( logBuff, len );
        break;
    case eventCodes::TELEM_OBSERVER:
        verified = telem_observer::verify( logBuff, len );
        break;
    case eventCodes::TELEM_RHUSB:
        verified = telem_rhusb::verify( logBuff, len );
        break;
    case eventCodes::TELEM_TEMPS:
        verified = telem_temps::verify( logBuff, len );
        break;
    case eventCodes::TELEM_STDCAM:
        verified = telem_stdcam::verify( logBuff, len );
        break;
    case eventCodes::TELEM_CORETEMPS:
        verified = telem_coretemps::verify( logBuff, len );
        break;
    case eventCodes::TELEM_CORELOADS:
        verified = telem_coreloads::verify( logBuff, len );
        break;
    case eventCodes::TELEM_DRIVETEMPS:
        verified = telem_drivetemps::verify( logBuff, len );
        break;
    case eventCodes::TELEM_USAGE:
        verified = telem_usage::verify( logBuff, len );
        break;
    case eventCodes::TELEM_COOLER:
        verified = telem_cooler::verify( logBuff, len );
        break;
    case eventCodes::TELEM_CHRONY_STATUS:
        verified = telem_chrony_status::verify( logBuff, len );
        break;
    case eventCodes::TELEM_CHRONY_STATS:
        verified = telem_chrony_stats::verify( logBuff, len );
        break;
    case eventCodes::TELEM_DMSPECK:
        verified = telem_dmspeck::verify( logBuff, len );
        break;
    case eventCodes::TELEM_FGTIMINGS:
        verified = telem_fgtimings::verify( logBuff, len );
        break;
    case eventCodes::TELEM_DMMODES:
        verified = telem_dmmodes::verify( logBuff, len );
        break;
    case eventCodes::TELEM_LOOPGAIN:
        verified = telem_loopgain::verify( logBuff, len );
        break;
    case eventCodes::TELEM_MODALGAINOPT:
        verified = telem_modalgainopt::verify( logBuff, len );
        break;
    case eventCodes::TELEM_BLOCKGAINS:
        verified = telem_blockgains::verify( logBuff, len );
        break;
    case eventCodes::TELEM_OFFLOADING:
        verified = telem_offloading::verify( logBuff, len );
        break;
    case eventCodes::TELEM_W2TCSOFFLOADER:
        verified = telem_w2tcsoffloader::verify( logBuff, len );
        break;
    case eventCodes::TELEM_FLOWRPM:
        verified = telem_flowrpm::verify( logBuff, len );
        break;
    case eventCodes::TELEM_PI335:
        verified = telem_pi335::verify( logBuff, len );
        break;
    case eventCodes::TELEM_SPARKLECLOCK:
        verified = telem_sparkleclock::verify( logBuff, len );
        break;
    case eventCodes::TELEM_POLTRACK:
        verified = telem_poltrack::verify( logBuff, len );
        break;
    case eventCodes::TELEM_ADCTRACK:
        verified = telem_adctrack::verify( logBuff, len );
        break;
    default:
        DEBUG_CRUMB( "verifyLogEntry unknown ev=" + std::to_string( ev ) +
                     " actualEv=" + std::to_string( flatlogs::logHeader::eventCode( log ) ) + " logTs=" +
                     logMapDebugTime( flatlogs::logHeader::timespec( log ) ) + " msgLen=" + std::to_string( len ) +
                     " totalSize=" + std::to_string( flatlogs::logHeader::totalSize( log ) ) + " " );
        return false;
    }

    if( !verified )
    {
        DEBUG_CRUMB( "verifyLogEntry failed ev=" + std::to_string( ev ) +
                     " actualEv=" + std::to_string( flatlogs::logHeader::eventCode( log ) ) + " logTs=" +
                     logMapDebugTime( flatlogs::logHeader::timespec( log ) ) + " msgLen=" + std::to_string( len ) +
                     " totalSize=" + std::to_string( flatlogs::logHeader::totalSize( log ) ) + " " );
    }

    return verified;
}

std::string logMeta::fitsKeyword() const
{
    std::string keyw;
    if( m_detail.hierarch )
    {
        // Add spaces to make sure hierarch is invoked
        keyw = m_spec.device + " " + m_spec.keyword;
        if( keyw.size() < 9 )
        {
            keyw += std::string( 9 - keyw.size(), ' ' );
        }
    }
    else
    {
        keyw = m_spec.keyword;
    }

    return keyw;
}

logMeta::logMeta( const logMetaSpec &lms )
{
    setLog( lms );
}

const std::string &logMeta::device()
{
    return m_spec.device;
}

const std::string &logMeta::keyword()
{
    return m_spec.keyword;
}

const std::string &logMeta::comment()
{
    return m_spec.comment;
}

const std::string &logMeta::unavailableReason() const
{
    return m_unavailableReason;
}

int logMeta::setLog( const logMetaSpec &lms )
{
    m_spec   = lms;
    m_detail = logMemberAccessor( m_spec.eventCode, m_spec.member );

    if( m_spec.keyword == "" )
        m_spec.keyword = m_detail.keyword;
    if( m_spec.format == "" )
        m_spec.format = m_detail.format;
    if( m_spec.format == "" )
    {
        switch( m_detail.valType )
        {
        case valTypes::String:
            m_spec.format = "%s";
            break;
        case valTypes::Bool:
            m_spec.format = "%d";
            break;
        case valTypes::Char:
            m_spec.format = "%d";
            break;
        case valTypes::UChar:
            m_spec.format = "%u";
            break;
        case valTypes::Short:
            m_spec.format = "%d";
            break;
        case valTypes::UShort:
            m_spec.format = "%u";
            break;
        case valTypes::Int:
            m_spec.format = "%d";
            break;
        case valTypes::UInt:
            m_spec.format = "%u";
            break;
        case valTypes::Long:
            m_spec.format = "%ld";
            break;
        case valTypes::ULong:
            m_spec.format = "%lu";
            break;
        case valTypes::Float:
            m_spec.format = "%G";
            break;
        case valTypes::Double:
            m_spec.format = "%G";
            break;
        case valTypes::Vector_Bool:
            m_spec.format = "%d";
            break;
        case valTypes::Vector_Float:
            m_spec.format = "%G";
            break;
        default:
            std::cerr << "Unrecognised value type for " + m_spec.device + " " + m_spec.keyword +
                             ".  Using format %d/\n";
            m_spec.format = "%d";
        }
    }

    if( m_spec.comment == "" )
        m_spec.comment = m_detail.comment;

    return 0;
}

std::string logMeta::value( logMap<verboseT>          &lm,
                            const flatlogs::timespecX &stime,
                            const flatlogs::timespecX &atime,
                            double                     maxGap )
{
    m_unavailableReason.clear();

    if( m_detail.accessor == nullptr )
    {
        m_unavailableReason = "because no metadata accessor is available";
        return m_invalidValue;
    }

    if( m_detail.valType == valTypes::String )
    {
        return valueString( lm, stime, atime, maxGap );
    }
    else
    {
        return valueNumber( lm, stime, atime, maxGap );
    }
}

std::string logMeta::valueNumber( logMap<verboseT>          &lm,
                                  const flatlogs::timespecX &stime,
                                  const flatlogs::timespecX &atime,
                                  double                     maxGap )
{
    char str[64];

    if( m_detail.metaType == metaTypes::State )
    {
        switch( m_detail.valType )
        {
        case valTypes::Bool:
        {
            bool val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<bool ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Char:
        {
            char val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<char ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
            {
                return m_invalidValue;
            }
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::UChar:
        {
            unsigned char val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<unsigned char ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Short:
        {
            short val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<short ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::UShort:
        {
            unsigned short val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<unsigned short ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Int:
        {
            int val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<int ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::UInt:
        {
            unsigned int val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<unsigned int ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Long:
        {
            long val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<long ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::ULong:
        {
            unsigned long val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<unsigned long ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::LongLong:
        {
            long long val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<long long ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::ULongLong:
        {
            unsigned long long val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<unsigned long long ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Float:
        {
            float val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<float ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Double:
        {
            double val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<double ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Vector_Bool:
        {
            std::vector<bool> val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<std::vector<bool> ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;

            if( val.size() == 0 )
                return "";

            std::string res;

            for( size_t n = 0; n < val.size() - 1; ++n )
            {
                snprintf( str, sizeof( str ), m_spec.format.c_str(), (int)val[n] );
                res += str;
                res += ',';
            }

            snprintf( str, sizeof( str ), m_spec.format.c_str(), (int)val.back() );
            res += str;

            return res;
        }
        case valTypes::Vector_Float:
        {
            std::vector<float> val;
            if( getLogStateVal( val,
                                lm,
                                m_spec.device,
                                m_spec.eventCode,
                                stime,
                                atime,
                                reinterpret_cast<std::vector<float> ( * )( void * )>( m_detail.accessor ),
                                &m_hint,
                                maxGap,
                                &m_unavailableReason ) != 0 )
                return m_invalidValue;

            if( val.size() == 0 )
                return "";

            std::string res;

            for( size_t n = 0; n < val.size() - 1; ++n )
            {
                snprintf( str, sizeof( str ), m_spec.format.c_str(), val[n] );
                res += str;
                res += ',';
            }

            snprintf( str, sizeof( str ), m_spec.format.c_str(), val.back() );
            res += str;

            return res;
        }
        default:
            return m_invalidValue;
        }
    }
    else if( m_detail.metaType == metaTypes::Continuous || m_detail.metaType == metaTypes::Continuous_Angle )
    {
        switch( m_detail.valType )
        {
        case valTypes::Bool:
        {
            bool val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<bool ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Char:
        {
            char val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<char ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::UChar:
        {
            unsigned char val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<unsigned char ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Short:
        {
            short val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<short ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::UShort:
        {
            unsigned short val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<unsigned short ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Int:
        {
            int val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<int ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::UInt:
        {
            unsigned int val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<unsigned int ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Long:
        {
            long val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<long ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::ULong:
        {
            unsigned long val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<unsigned long ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::LongLong:
        {
            long long val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<long long ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::ULongLong:
        {
            unsigned long long val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<unsigned long long ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Float:
        {
            float val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<float ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason,
                               m_detail.metaType == metaTypes::Continuous_Angle ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        case valTypes::Double:
        {
            double val;
            if( getLogContVal( val,
                               lm,
                               m_spec.device,
                               m_spec.eventCode,
                               stime,
                               atime,
                               reinterpret_cast<double ( * )( void * )>( m_detail.accessor ),
                               &m_hint,
                               maxGap,
                               &m_unavailableReason,
                               m_detail.metaType == metaTypes::Continuous_Angle ) != 0 )
                return m_invalidValue;
            snprintf( str, sizeof( str ), m_spec.format.c_str(), val );
            return std::string( str );
        }
        default:
            return m_invalidValue;
        }
    }

    return m_invalidValue;
}

std::string logMeta::valueString( logMap<verboseT>          &lm,
                                  const flatlogs::timespecX &stime,
                                  const flatlogs::timespecX &atime,
                                  double                     maxGap )
{
    std::string val;
    if( m_detail.metaType == metaTypes::State )
    {
        if( getLogStateVal( val,
                            lm,
                            m_spec.device,
                            m_spec.eventCode,
                            stime,
                            atime,
                            reinterpret_cast<std::string ( * )( void * )>( m_detail.accessor ),
                            &m_hint,
                            maxGap,
                            &m_unavailableReason ) != 0 )
        {
#ifdef HARD_EXIT
            std::cerr << __FILE__ << " " << __LINE__ << "\n";

            exit( -1 );
#endif
            val = m_invalidValue;
        }
    }
    else
    {
        std::cerr << "String type specified as something other than state\n";
    }
    return val;
}

mx::fits::fitsHeaderCard<logMeta::verboseT> logMeta::unavailableCard() const
{
    return mx::fits::fitsHeaderCard<verboseT>( fitsKeyword(), unavailableValue(), m_spec.comment );
}

mx::fits::fitsHeaderCard<logMeta::verboseT>
logMeta::card( logMap<verboseT> &lm, const flatlogs::timespecX &stime, const flatlogs::timespecX &atime, double maxGap )
{
#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    std::string vstr = value( lm, stime, atime, maxGap );

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    std::string keyw = fitsKeyword();

    if( vstr == m_invalidValue )
    {
        // always a string sentinel value, so return here to skip the valType conditional
        return mx::fits::fitsHeaderCard<verboseT>( keyw, vstr, m_spec.comment );
    }

    if( m_detail.valType == valTypes::String || m_detail.valType == valTypes::Vector_Bool ||
        m_detail.valType == valTypes::Vector_Float )
    {
        return mx::fits::fitsHeaderCard<verboseT>( keyw, vstr, m_spec.comment );
    }
    else
    {
        return mx::fits::fitsHeaderCard<verboseT>( keyw, vstr.c_str(), m_detail.valType, m_spec.comment );
    }
}

} // namespace logger
} // namespace MagAOX
