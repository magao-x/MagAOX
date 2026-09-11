/** \file logMeta_test.cpp
 * \brief Tests for the logMeta class and its free functions
 *
 * Technique: each test writes real flatlog files with logFileRaw into a scratch
 * directory under /tmp, registers them in a logMap, and reads values back through
 * logMeta, getLogStateVal, and getLogContVal. Some tests corrupt bytes in the loaded
 * buffer to reach the schema verification failure branches. Some tests use
 * logMetaExposed to install a hand-built accessor, so that value type and metadata type
 * combinations no real log type uses can be reached. The tests need write access to
 * /tmp.
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>

#include "../../file/fileTimes.hpp"

#include "../logFileRaw.hpp"
#include "../logMap.hpp"
#include "../logMap.cpp"

#include "../logMeta.hpp"
#include "../generated/logTypes.hpp"
#include "../logMeta.cpp"

namespace libXWCTest
{

namespace loggerTest
{

/** \defgroup logMeta_unit_test logMeta Unit Tests
 * \ingroup logger_unit_test
 */

/// Namespace for XWC::logger::logMeta tests
/** \ingroup logMeta_unit_test
 *
 */
namespace logMetaTest
{

// An int-valued state variable log used to exercise getLogStateVal. The merged logMeta
// verifies every scanned entry against the flatbuffer schema of its event code through
// verifyLogEntry. A raw int payload under an invented event code would fail that check.
// So this type wraps a real state_change log and carries the value in its 'to' field.
struct dummyLogInt
{
    static const flatlogs::eventCodeT eventCode;
    static const flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_NOTICE;

    typedef int messageT;

    static flatlogs::msgLenT length( const messageT &msg )
    {
        MagAOX::logger::state_change::messageT m( 0, msg );
        return MagAOX::logger::state_change::length( m );
    }

    static int format( void *msgBuffer, const messageT &msg )
    {
        MagAOX::logger::state_change::messageT m( 0, msg );
        return MagAOX::logger::state_change::format( msgBuffer, m );
    }
};
const flatlogs::eventCodeT dummyLogInt::eventCode = MagAOX::logger::eventCodes::STATE_CHANGE;

// A second int state log, used by the test whose value never changes. It uses the same
// state_change wrapping as dummyLogInt. The tests use separate devices and directories,
// so sharing the STATE_CHANGE event code is fine.
struct dummyLogIntConst
{
    static const flatlogs::eventCodeT eventCode;
    static const flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_NOTICE;

    typedef int messageT;

    static flatlogs::msgLenT length( const messageT &msg )
    {
        MagAOX::logger::state_change::messageT m( 0, msg );
        return MagAOX::logger::state_change::length( m );
    }

    static int format( void *msgBuffer, const messageT &msg )
    {
        MagAOX::logger::state_change::messageT m( 0, msg );
        return MagAOX::logger::state_change::format( msgBuffer, m );
    }
};
const flatlogs::eventCodeT dummyLogIntConst::eventCode = MagAOX::logger::eventCodes::STATE_CHANGE;

// A double-valued continuous variable log used to exercise getLogContVal. It wraps a
// real telem_teldata log for the reason given at dummyLogInt, and carries the value in
// the 'az' field.
struct dummyLogDouble
{
    static const flatlogs::eventCodeT eventCode;
    static const flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_NOTICE;

    typedef double messageT;

    static flatlogs::msgLenT length( const messageT &msg )
    {
        MagAOX::logger::telem_teldata::messageT m( 1, 1, 1, 0, 0, msg, 45.0, 5.0, 90.0, 0 );
        return MagAOX::logger::telem_teldata::length( m );
    }

    static int format( void *msgBuffer, const messageT &msg )
    {
        MagAOX::logger::telem_teldata::messageT m( 1, 1, 1, 0, 0, msg, 45.0, 5.0, 90.0, 0 );
        return MagAOX::logger::telem_teldata::format( msgBuffer, m );
    }
};
const flatlogs::eventCodeT dummyLogDouble::eventCode = MagAOX::logger::eventCodes::TELEM_TELDATA;

// Accessors handed to getLogStateVal, getLogContVal, and logMetaExposed::setDetail. Each
// reads the value back out of the real flatbuffer payload as one C++ type.
int getIntVal( void *buf )
{
    return MagAOX::logger::GetState_change_fb( buf )->to();
}

double getDoubleVal( void *buf )
{
    return MagAOX::logger::GetTelem_teldata_fb( buf )->az();
}

char getCharVal( void *buf )
{
    return (char)MagAOX::logger::GetState_change_fb( buf )->to();
}

unsigned char getUCharVal( void *buf )
{
    return (unsigned char)MagAOX::logger::GetState_change_fb( buf )->to();
}

short getShortVal( void *buf )
{
    return (short)MagAOX::logger::GetState_change_fb( buf )->to();
}

unsigned short getUShortVal( void *buf )
{
    return (unsigned short)MagAOX::logger::GetState_change_fb( buf )->to();
}

long getLongVal( void *buf )
{
    return (long)MagAOX::logger::GetState_change_fb( buf )->to();
}

unsigned long getULongVal( void *buf )
{
    return (unsigned long)MagAOX::logger::GetState_change_fb( buf )->to();
}

long long getLongLongVal( void *buf )
{
    return (long long)MagAOX::logger::GetState_change_fb( buf )->to();
}

unsigned int getUIntVal( void *buf )
{
    return (unsigned int)MagAOX::logger::GetState_change_fb( buf )->to();
}

unsigned long long getULongLongVal( void *buf )
{
    return (unsigned long long)MagAOX::logger::GetState_change_fb( buf )->to();
}

// Exposes the protected m_detail of logMeta. A test can then drive valueNumber() and
// valueString() with valType and metaType combinations that no real MagAO-X log type
// uses. That claim was checked by grepping every getAccessor() in types/*.hpp. This
// avoids the need for a substitute logMemberAccessor.
struct logMetaExposed : public MagAOX::logger::logMeta
{
    logMetaExposed( const MagAOX::logger::logMetaSpec &lms ) : MagAOX::logger::logMeta( lms )
    {
    }

    void setDetail( const MagAOX::logger::logMetaDetail &d )
    {
        m_detail = d;
    }
};

// Writes a single log file containing one entry per pair of offset from base and value,
// in order. This mirrors the on-disk setup used by the logMap tests. Only one file is
// needed because these tests do not exercise loadFiles.
template <class dummyLogT, typename valT>
void writeLogFile( const std::string                             &dir,
                   const std::string                             &dev,
                   time_t                                          base,
                   const std::vector<std::pair<time_t, valT>> &entries )
{
    std::filesystem::remove_all( dir );

    MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> writer;
    writer.logPath( dir );
    writer.logName( dev );
    writer.logExt( "xlog" );
    writer.maxLogSize( 1000000 ); // This is large enough that all entries land in one file.

    // Lead with a text_log entry 10 seconds before base. The first entry sets the
    // timestamp in the on-disk file name. The merged loadFiles only loads a file whose
    // timestamp is strictly before the queried instant. A file starting exactly at the
    // query time is not loaded. No test in this file queries TEXT_LOG, so the lead-in
    // never affects a result.
    flatlogs::bufferPtrT buf;
    flatlogs::logHeader::createLog<MagAOX::logger::text_log>(
        buf, flatlogs::timespecX( base - 10, 0 ), "lead-in", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );

    for( auto &e : entries )
    {
        flatlogs::logHeader::createLog<dummyLogT>(
            buf, flatlogs::timespecX( base + e.first, 0 ), e.second, flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    }
    writer.close();
}

// Inserts the single file written above into the file map of lm directly. This bypasses
// directory scanning. The logMap tests use the same shortcut for getPriorLog, getNextLog,
// and loadFiles.
template <class verboseT>
void insertLogFile( MagAOX::logger::logMap<verboseT> &lm, const std::string &dir, const std::string &dev, time_t base )
{
    std::string fileName, relPath;
    // The file is named for its first entry, which is the lead-in text_log at base minus
    // 10 seconds. See writeLogFile and writeRealLogFile.
    MagAOX::file::fileTimeRelPath( fileName, relPath, dev, "xlog", base - 10, 0 );

    MagAOX::file::stdFileName<verboseT> sfn( dir + '/' + relPath + '/' + fileName );
    REQUIRE( sfn.valid() );
    lm.m_appToFileMap[dev].insert( sfn );
}

/// logMetaSpec and logMetaDetail store the fields they are constructed with
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMetaSpec and logMetaDetail store the supplied fields", "[libMagAOX::logger::logMeta]" )
{
    SECTION( "logMetaSpec default construction" )
    {
        MagAOX::logger::logMetaSpec lms;
        REQUIRE( lms.device == "" );
        REQUIRE( lms.member == "" );
    }

    SECTION( "logMetaSpec full construction" )
    {
        MagAOX::logger::logMetaSpec lms( "dev1", 7, "memb", "KEYW", "%d", "a comment" );
        REQUIRE( lms.device == "dev1" );
        REQUIRE( lms.eventCode == 7 );
        REQUIRE( lms.member == "memb" );
        REQUIRE( lms.keyword == "KEYW" );
        REQUIRE( lms.format == "%d" );
        REQUIRE( lms.comment == "a comment" );
    }

    SECTION( "logMetaSpec minimal construction leaves overrides empty" )
    {
        MagAOX::logger::logMetaSpec lms( "dev2", 8, "memb2" );
        REQUIRE( lms.device == "dev2" );
        REQUIRE( lms.eventCode == 8 );
        REQUIRE( lms.member == "memb2" );
        REQUIRE( lms.keyword == "" );
        REQUIRE( lms.format == "" );
        REQUIRE( lms.comment == "" );
    }

    SECTION( "logMetaDetail default construction" )
    {
        MagAOX::logger::logMetaDetail lmd;
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "logMetaDetail(k,c,f,vt,mt,acc)" )
    {
        int acc = 0;
        MagAOX::logger::logMetaDetail lmd( "KEYW", "comment", "%d", 3, 0, &acc );
        REQUIRE( lmd.keyword == "KEYW" );
        REQUIRE( lmd.comment == "comment" );
        REQUIRE( lmd.format == "%d" );
        REQUIRE( lmd.valType == 3 );
        REQUIRE( lmd.metaType == 0 );
        REQUIRE( lmd.accessor == &acc );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "logMetaDetail(k,c,f,vt,mt,acc,h)" )
    {
        int acc = 0;
        MagAOX::logger::logMetaDetail lmd( "KEYW", "comment", "%d", 3, 0, &acc, false );
        REQUIRE( lmd.hierarch == false );
        REQUIRE( lmd.accessor == &acc );
    }

    SECTION( "logMetaDetail(k,c,vt,mt,acc,h) has no format override" )
    {
        int acc = 0;
        MagAOX::logger::logMetaDetail lmd( "KEYW", "comment", 3, 0, &acc, false );
        REQUIRE( lmd.keyword == "KEYW" );
        REQUIRE( lmd.comment == "comment" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == 3 );
        REQUIRE( lmd.metaType == 0 );
        REQUIRE( lmd.accessor == &acc );
        REQUIRE( lmd.hierarch == false );
    }

    SECTION( "logMetaDetail(k,vt,mt,acc)" )
    {
        int acc = 0;
        MagAOX::logger::logMetaDetail lmd( "KEYW", 3, 0, &acc );
        REQUIRE( lmd.keyword == "KEYW" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.valType == 3 );
        REQUIRE( lmd.metaType == 0 );
        REQUIRE( lmd.accessor == &acc );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "logMetaDetail(k,vt,mt,acc,h)" )
    {
        int acc = 0;
        MagAOX::logger::logMetaDetail lmd( "KEYW", 3, 0, &acc, false );
        REQUIRE( lmd.keyword == "KEYW" );
        REQUIRE( lmd.valType == 3 );
        REQUIRE( lmd.metaType == 0 );
        REQUIRE( lmd.accessor == &acc );
        REQUIRE( lmd.hierarch == false );
    }
}

/// getLogStateVal tracks the value of a state variable over a requested interval.
/// The log file holds five entries ten seconds apart with values 1, 1, 2, 2, and 3.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "getLogStateVal tracks a state variable's value over a requested interval",
           "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_state";
    const std::string dev  = "devS1";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    // Write five entries ten seconds apart, with values 1, 1, 2, 2, and 3.
    writeLogFile<dummyLogInt, int>(
        dir, dev, base, { { 0, 1 }, { 10, 1 }, { 20, 2 }, { 30, 2 }, { 40, 3 } } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    insertLogFile( lm, dir, dev, base );

    SECTION( "detects a state change before atime and reports the new value, updating the hint" )
    {
        char *h  = nullptr;
        int   val;
        int   rv = MagAOX::logger::getLogStateVal(
            val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 45, 0 ), getIntVal, &h );

        REQUIRE( rv == 0 );
        REQUIRE( val == 2 );
        REQUIRE( h != nullptr );
    }

    SECTION( "no state change occurs before atime, returns the anchor value, with no hint requested" )
    {
        int val;
        int rv = MagAOX::logger::getLogStateVal(
            val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 15, 0 ), getIntVal );

        REQUIRE( rv == 0 );
        REQUIRE( val == 1 );
    }

    SECTION( "no state change occurs before atime, and the hint is still updated on the natural loop exit" )
    {
        char *h  = nullptr;
        int   val;
        int   rv = MagAOX::logger::getLogStateVal(
            val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 15, 0 ), getIntVal, &h );

        REQUIRE( rv == 0 );
        REQUIRE( val == 1 );
        REQUIRE( h != nullptr );
    }

    SECTION( "an event code with no entries reports an error from the initial getPriorLog" )
    {
        int val;
        // 9999 is an arbitrary code that no log type uses.
        int rv = MagAOX::logger::getLogStateVal(
            val, lm, dev, 9999, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 45, 0 ), getIntVal );

        REQUIRE( rv == -1 );
    }

    SECTION( "an anchor at the very last entry causes the initial getNextLog to fail" )
    {
        // The forward search in getPriorLog looks for the last entry strictly before ts.
        // So it always lands one entry short of an exact time match. The only way to make
        // it land on the literal last entry in the buffer is to point a hint directly at
        // that entry. The logMap tests use the same technique to get real pointers into
        // the loaded buffer.
        char *tmp = nullptr;
        REQUIRE( lm.getPriorLog( tmp, dev, dummyLogInt::eventCode, flatlogs::timespecX( base, 0 ) ) == 0 );

        char *p0 = lm.m_appToBufferMap[dev].m_memory.data();
        char *p1 = p0 + flatlogs::logHeader::totalSize( p0 );
        char *p2 = p1 + flatlogs::logHeader::totalSize( p1 );
        char *p3 = p2 + flatlogs::logHeader::totalSize( p2 );
        char *p4 = p3 + flatlogs::logHeader::totalSize( p3 ); // This is the last entry, at base+40 with value 3.

        char *hint = p4;
        int   val;
        int   rv = MagAOX::logger::getLogStateVal( val,
                                                    lm,
                                                    dev,
                                                    dummyLogInt::eventCode,
                                                    flatlogs::timespecX( base + 40, 0 ),
                                                    flatlogs::timespecX( base + 1000, 0 ),
                                                    getIntVal,
                                                    &hint );

        REQUIRE( rv == -1 );
    }
}

/// getLogStateVal reports an error when it exhausts the buffer while searching for a
/// state change.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "getLogStateVal reports an error when it runs out of entries while searching for a state change",
           "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_state_tail";
    const std::string dev  = "devS2";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    // Write five entries ten seconds apart, with values 1, 1, 2, 2, and 2. The last three
    // never change. So a search that starts from the third entry runs off the end of the
    // buffer.
    writeLogFile<dummyLogIntConst, int>(
        dir, dev, base, { { 0, 1 }, { 10, 1 }, { 20, 2 }, { 30, 2 }, { 40, 2 } } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    insertLogFile( lm, dir, dev, base );

    int val;
    // stime is strictly between the third and fourth entries, so the anchor is the third
    // entry with value 2.
    int rv = MagAOX::logger::getLogStateVal( val,
                                              lm,
                                              dev,
                                              dummyLogIntConst::eventCode,
                                              flatlogs::timespecX( base + 21, 0 ),
                                              flatlogs::timespecX( base + 1000, 0 ),
                                              getIntVal );

    REQUIRE( rv == -1 );
}

/// getLogContVal interpolates a continuous variable between two log entries.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "getLogContVal interpolates a continuous variable between two log entries",
           "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_cont";
    const std::string dev  = "devC1";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    // Write three entries with uneven spacing. The values are 0.0 at t+0, 10.0 at t+10,
    // and 40.0 at t+40.
    writeLogFile<dummyLogDouble, double>( dir, dev, base, { { 0, 0.0 }, { 10, 10.0 }, { 40, 40.0 } } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    insertLogFile( lm, dir, dev, base );

    SECTION( "interpolates linearly between the surrounding entries, using and updating a hint" )
    {
        char  *h  = nullptr;
        double val;
        // The midpoint of base+5 and base+25 is base+15. That anchors on the t+10 entry
        // and interpolates toward the t+40 entry.
        int rv = MagAOX::logger::getLogContVal(
            val, lm, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base + 5, 0 ), flatlogs::timespecX( base + 25, 0 ), getDoubleVal, &h );

        REQUIRE( rv == 0 );
        REQUIRE( val == Approx( 15.0 ) );
        REQUIRE( h != nullptr );
    }

    SECTION( "an unknown app reports an error from the initial getPriorLog" )
    {
        double val;
        int    rv = MagAOX::logger::getLogContVal(
            val, lm, "no-such-app", dummyLogDouble::eventCode, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 20, 0 ), getDoubleVal );

        REQUIRE( rv == 1 );
    }

    SECTION( "midpoint lands on the final entry so the following getNextLog fails" )
    {
        // As above, getPriorLog can only land on the literal last entry through a hint
        // pointed directly at it.
        char *tmp = nullptr;
        REQUIRE( lm.getPriorLog( tmp, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base, 0 ) ) == 0 );

        char *p0 = lm.m_appToBufferMap[dev].m_memory.data();
        char *p1 = p0 + flatlogs::logHeader::totalSize( p0 );
        char *p2 = p1 + flatlogs::logHeader::totalSize( p1 ); // This is the last entry, at base+40 with value 40.0.

        char  *hint = p2;
        double val;
        int    rv = MagAOX::logger::getLogContVal( val,
                                                  lm,
                                                  dev,
                                                  dummyLogDouble::eventCode,
                                                  flatlogs::timespecX( base + 40, 0 ),
                                                  flatlogs::timespecX( base + 40, 0 ),
                                                  getDoubleVal,
                                                  &hint );

        REQUIRE( rv == 1 );
    }
}

// Writes a single log file for a real flatlog type rather than a dummy type. The
// writeEntries callback is handed the writer. It must call logHeader::createLog<Type>()
// and writer.writeLog() for each entry it wants. The firstTime argument must match the
// timestamp of the first entry the callback writes, because a lead-in entry is placed
// 10 seconds before it and that lead-in determines the on-disk file name.
MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY>
writeRealLogFile( const std::string                                              &dir,
                  const std::string                                              &dev,
                  time_t                                                          firstTime,
                  const std::function<void( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> & )> &writeEntries )
{
    std::filesystem::remove_all( dir );

    MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> writer;
    writer.logPath( dir );
    writer.logName( dev );
    writer.logExt( "xlog" );
    writer.maxLogSize( 1000000 );

    // This is the same lead-in as in writeLogFile. It gives the file a timestamp strictly
    // before firstTime, which the merged loadFiles requires.
    flatlogs::bufferPtrT leadBuf;
    flatlogs::logHeader::createLog<MagAOX::logger::text_log>(
        leadBuf, flatlogs::timespecX( firstTime - 10, 0 ), "lead-in", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( leadBuf );

    writeEntries( writer );

    writer.close();

    std::string fileName, relPath;
    MagAOX::file::fileTimeRelPath( fileName, relPath, dev, "xlog", firstTime - 10, 0 );

    MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> sfn( dir + '/' + relPath + '/' + fileName );
    REQUIRE( sfn.valid() );
    return sfn;
}

/// logMeta dispatches through the real generated log type accessors.
/**
 * This and the following real-type test cases exercise the logMeta constructor, setLog(),
 * value(), valueNumber(), valueString(), and card() against genuine on-disk flatlog
 * entries for real MagAO-X log types. This is necessary because logMeta::setLog() always
 * calls the real generated logMemberAccessor(). There is no way to substitute a fake
 * accessor without fabricating the binary format of a real log type.
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMeta reads String/State and Bool/Continuous members from a real git_state log",
           "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_gitstate";
    const std::string dev  = "devGit";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeRealLogFile( dir, dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::git_state>(
            buf, flatlogs::timespecX( base, 0 ), MagAOX::logger::git_state::messageT( "repoA", "sha1A", false ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
        flatlogs::logHeader::createLog<MagAOX::logger::git_state>(
            buf, flatlogs::timespecX( base + 10, 0 ), MagAOX::logger::git_state::messageT( "repoB", "sha1B", true ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap[dev].insert( sfn );

    SECTION( "String/State: repoName, with explicit keyword/format/comment overrides" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::git_state::eventCode, "repoName", "REPO", "%s", "custom comment" );
        MagAOX::logger::logMeta     lmeta( lms );

        REQUIRE( lmeta.device() == dev );
        REQUIRE( lmeta.keyword() == "REPO" );
        REQUIRE( lmeta.comment() == "custom comment" );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val == "repoA" );

        // With stime past the data the lookup fails, which exercises the failure return
        // for this member.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        auto card = lmeta.card( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( card.keyword() == "REPO" ); // hierarch is false for repoName, so there is no device prefix and no padding.
        REQUIRE( card.comment() == "custom comment" );
    }

    SECTION( "Bool/Continuous: modified, with default keyword/comment from the accessor" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::git_state::eventCode, "modified" );
        MagAOX::logger::logMeta     lmeta( lms );

        REQUIRE( lmeta.keyword() == "GIT REPO MODIFIED" );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) );
        REQUIRE( val != "invalid" );
    }

    SECTION( "an unrecognized member leaves the accessor null, so value() returns empty" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::git_state::eventCode, "notAMember" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val == "NOT AVAILABLE" ); // This is the merged unavailableValue() sentinel. It was "" before the merge.
    }

    SECTION( "a failed lookup (unknown device) produces the invalid-value sentinel via card()" )
    {
        MagAOX::logger::logMetaSpec lms( "no-such-device", MagAOX::logger::git_state::eventCode, "repoName" );
        MagAOX::logger::logMeta     lmeta( lms );

        auto card = lmeta.card( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( card.valueStr() == "NOT AVAILABLE" ); // This is the merged unavailableValue() sentinel. It was "invalid" before the merge.
    }
}

/// logMeta reads Bool/State, Float/State, Vector_Bool/State, and Vector_Float/State from
/// telem_dmspeck. It also exercises the keyword padding branches of card() that run when
/// hierarch is true.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMeta reads Bool/Float/Vector members from a real telem_dmspeck log", "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_dmspeck";
    const std::string dev  = "devDmspeck";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeRealLogFile( dir, dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::telem_dmspeck>(
            buf,
            flatlogs::timespecX( base, 0 ),
            MagAOX::logger::telem_dmspeck::messageT(
                true, false, 12.5f, std::vector<float>{ 1.5f, 2.5f }, std::vector<float>{ 0.1f, 0.2f }, std::vector<float>{ 3.0f, 4.0f }, std::vector<bool>{ true, false } ),
            flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
        flatlogs::logHeader::createLog<MagAOX::logger::telem_dmspeck>(
            buf,
            flatlogs::timespecX( base + 10, 0 ),
            MagAOX::logger::telem_dmspeck::messageT(
                true, false, 12.5f, std::vector<float>{ 1.5f, 2.5f }, std::vector<float>{ 0.1f, 0.2f }, std::vector<float>{ 3.0f, 4.0f }, std::vector<bool>{ true, false } ),
            flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap[dev].insert( sfn );
    // logMap keys files purely by the appName string, independent of the log content. So
    // the same file can be registered under extra aliases. Each section then picks a
    // device name length that exercises the padding logic in card().
    lm.m_appToFileMap["cameraOne"].insert( sfn );
    lm.m_appToFileMap["d"].insert( sfn );

    SECTION( "Bool/State: modulating, with a long device+keyword so card() skips padding" )
    {
        MagAOX::logger::logMetaSpec lms( "cameraOne", MagAOX::logger::telem_dmspeck::eventCode, "modulating" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val == "1" );

        // With stime past the data the lookup fails, which exercises the failure return
        // for this member.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        // With stime past the data the lookup fails, which exercises the failure return
        // for this member.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        auto card = lmeta.card( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( card.keyword() == "cameraOne MODULATING" ); // This is already at least 9 characters, so no padding is needed.
    }

    SECTION( "Float/State: frequency, with a short device+keyword override so card() pads" )
    {
        MagAOX::logger::logMetaSpec lms( "d", MagAOX::logger::telem_dmspeck::eventCode, "frequency", "F", "", "" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val != "invalid" );

        auto card = lmeta.card( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( card.keyword().size() >= 9 ); // The 3 character keyword "d F" is padded out to at least 9.
    }

    SECTION( "Vector_Bool/State: crosses" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_dmspeck::eventCode, "crosses" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val == "1,0" );

        // With stime past the data the lookup fails, which exercises the failure return
        // for this member.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "Vector_Float/State: separations" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_dmspeck::eventCode, "separations" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val != "invalid" );
        REQUIRE( val.find( ',' ) != std::string::npos );

        // With stime past the data the lookup fails, which exercises the failure return
        // for this member.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }
}

/// logMeta reads Int/State and Float/Continuous from a real telem_psfacq log.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMeta reads Int/State and Float/Continuous members from a real telem_psfacq log",
           "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_psfacq";
    const std::string dev  = "devPsfacq";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeRealLogFile( dir, dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::telem_psfacq>(
            buf, flatlogs::timespecX( base, 0 ), MagAOX::logger::telem_psfacq::messageT( 1, 3, 10.0f, 20.0f, 100.0f, 2.5f, 0.8f ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
        flatlogs::logHeader::createLog<MagAOX::logger::telem_psfacq>(
            buf, flatlogs::timespecX( base + 10, 0 ), MagAOX::logger::telem_psfacq::messageT( 1, 3, 30.0f, 20.0f, 100.0f, 2.5f, 0.8f ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap[dev].insert( sfn );

    SECTION( "Int/State: star_no" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_psfacq::eventCode, "star_no" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val == "1" );
    }

    SECTION( "Float/Continuous: x_pos interpolates between the two entries" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_psfacq::eventCode, "x_pos" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) );
        REQUIRE( val != "invalid" );
        REQUIRE( std::stof( val ) > 10.0f );
        REQUIRE( std::stof( val ) < 30.0f );
    }
}

/// logMeta reads UInt/State and Double/State from a real telem_saving log.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMeta reads UInt/State and Double/State members from a real telem_saving log",
           "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_saving";
    const std::string dev  = "devSaving";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeRealLogFile( dir, dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::telem_saving>(
            buf, flatlogs::timespecX( base, 0 ), MagAOX::logger::telem_saving::messageT( 1000, 500, 1.5f, 0.5f, 0.2f, 0.3f ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
        flatlogs::logHeader::createLog<MagAOX::logger::telem_saving>(
            buf, flatlogs::timespecX( base + 10, 0 ), MagAOX::logger::telem_saving::messageT( 1000, 500, 1.5f, 0.5f, 0.2f, 0.3f ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap[dev].insert( sfn );

    SECTION( "UInt/State: raw_size" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_saving::eventCode, "raw_size" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val == "1000" );

        // With stime past the data the lookup fails, which exercises the failure return
        // for this member.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "Double/State: encode_sate" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_saving::eventCode, "encode_sate" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val != "invalid" );
    }
}

/// logMeta reads Double/Continuous from a real telem_teldata log.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMeta reads a Double/Continuous member from a real telem_teldata log", "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_teldata";
    const std::string dev  = "devTeldata";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeRealLogFile( dir, dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::telem_teldata>(
            buf, flatlogs::timespecX( base, 0 ), MagAOX::logger::telem_teldata::messageT( 1, 1, 1, 0, 0, 10.0, 45.0, 5.0, 90.0, 0 ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
        flatlogs::logHeader::createLog<MagAOX::logger::telem_teldata>(
            buf, flatlogs::timespecX( base + 10, 0 ), MagAOX::logger::telem_teldata::messageT( 1, 1, 1, 0, 0, 20.0, 45.0, 5.0, 90.0, 0 ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap[dev].insert( sfn );

    MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_teldata::eventCode, "az" );
    MagAOX::logger::logMeta     lmeta( lms );

    std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) );
    REQUIRE( val != "invalid" );
    REQUIRE( std::stod( val ) > 10.0 );
    REQUIRE( std::stod( val ) < 20.0 );
}

/// logMeta reads ULongLong/State from a real telem_pokeloop log. One section uses the
/// default format fallback in setLog() for an unrecognized type. The other section uses
/// an explicit format override so that valueNumber() can safely format the value.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMeta reads a ULongLong/State member from a real telem_pokeloop log", "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_pokeloop";
    const std::string dev  = "devPokeloop";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeRealLogFile( dir, dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::telem_pokeloop>(
            buf, flatlogs::timespecX( base, 0 ), MagAOX::logger::telem_pokeloop::messageT( 1, 0.5f, 0.5f, 12345ULL ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
        flatlogs::logHeader::createLog<MagAOX::logger::telem_pokeloop>(
            buf, flatlogs::timespecX( base + 10, 0 ), MagAOX::logger::telem_pokeloop::messageT( 1, 0.5f, 0.5f, 12345ULL ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap[dev].insert( sfn );

    SECTION( "no format override: setLog() falls through to its unrecognized-type default" )
    {
        // ULongLong is not one of the explicit format cases in setLog(), so this exercises
        // the default branch. A mismatched printf format for a 64-bit value is unsafe to
        // format. So this only checks the bookkeeping in setLog(), not value() or card().
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_pokeloop::eventCode, "counter" );
        MagAOX::logger::logMeta     lmeta( lms );

        REQUIRE( lmeta.keyword() == "LOOP COUNTER" );
    }

    SECTION( "with an explicit format override, valueNumber()'s ULongLong case runs safely" )
    {
        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_pokeloop::eventCode, "counter", "", "%llu", "" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( val == "12345" ); // the counter written in the pokeloop entry above

        // With stime past the data the lookup fails, which exercises the failure return
        // for this member.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }
}

/// logMeta falls back to the invalid value sentinel when the valType of a real accessor is
/// not handled for its metaType. Here Vector_Float has no Continuous case in
/// valueNumber(). This also covers the invalid value branch of card().
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMeta falls back to the invalid sentinel for a valType unhandled by its metaType",
           "[libMagAOX::logger::logMeta]" )
{
    const std::string dir  = "/tmp/logMeta_test_dmmodes";
    const std::string dev  = "devDmmodes";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeRealLogFile( dir, dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
        flatlogs::bufferPtrT      buf;
        std::vector<float>        amps{ 1.0f, 2.0f };
        flatlogs::logHeader::createLog<MagAOX::logger::telem_dmmodes>(
            buf, flatlogs::timespecX( base, 0 ), MagAOX::logger::telem_dmmodes::messageT( amps ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
        flatlogs::logHeader::createLog<MagAOX::logger::telem_dmmodes>(
            buf, flatlogs::timespecX( base + 10, 0 ), MagAOX::logger::telem_dmmodes::messageT( amps ), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap[dev].insert( sfn );

    MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_dmmodes::eventCode, "amps" );
    MagAOX::logger::logMeta     lmeta( lms );

    std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) );
    REQUIRE( val == "NOT AVAILABLE" ); // This is the merged unavailableValue() sentinel. It was "invalid" before the merge.

    auto card = lmeta.card( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) );
    REQUIRE( card.valueStr() == "NOT AVAILABLE" );
}

// Dummy log types, one per C++ type needed to reach a valType and metaType combination
// that no real MagAO-X log type uses. That claim was checked by grepping every
// getAccessor() in types/*.hpp. Each type writes a real state_change flatbuffer with the
// value in its 'to' field. The merged logMeta verifies every scanned entry against the
// schema of its event code through verifyLogEntry, so the payload must genuinely verify.
// The per-type accessors defined above then read 'to' back as each C++ type. The EVCODE
// macro argument is kept for the call sites but is unused. All of these are state_change
// entries now.
#define DEFINE_DUMMY_RAW_LOG( NAME, TYPE, EVCODE )                                                                    \
    struct NAME                                                                                                       \
    {                                                                                                                 \
        static const flatlogs::eventCodeT eventCode;                                                                  \
        static const flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_NOTICE;                               \
        typedef TYPE                      messageT;                                                                   \
        static flatlogs::msgLenT          length( const messageT &msg )                                               \
        {                                                                                                             \
            MagAOX::logger::state_change::messageT m( 0, (int16_t)msg );                                              \
            return MagAOX::logger::state_change::length( m );                                                         \
        }                                                                                                             \
        static int format( void *msgBuffer, const messageT &msg )                                                     \
        {                                                                                                             \
            MagAOX::logger::state_change::messageT m( 0, (int16_t)msg );                                              \
            return MagAOX::logger::state_change::format( msgBuffer, m );                                              \
        }                                                                                                             \
    };                                                                                                                \
    const flatlogs::eventCodeT NAME::eventCode = MagAOX::logger::eventCodes::STATE_CHANGE;

DEFINE_DUMMY_RAW_LOG( dummyLogChar, char, 60001 ) // codes 60001 and up are arbitrary and above every real event code
DEFINE_DUMMY_RAW_LOG( dummyLogUChar, unsigned char, 60002 )
DEFINE_DUMMY_RAW_LOG( dummyLogShort, short, 60003 )
DEFINE_DUMMY_RAW_LOG( dummyLogUShort, unsigned short, 60004 )
DEFINE_DUMMY_RAW_LOG( dummyLogLong, long, 60005 )
DEFINE_DUMMY_RAW_LOG( dummyLogULong, unsigned long, 60006 )
DEFINE_DUMMY_RAW_LOG( dummyLogLongLong, long long, 60007 )
DEFINE_DUMMY_RAW_LOG( dummyLogIntCont, int, 60008 )
DEFINE_DUMMY_RAW_LOG( dummyLogUIntCont, unsigned int, 60009 )
DEFINE_DUMMY_RAW_LOG( dummyLogULongLongCont, unsigned long long, 60010 )

#undef DEFINE_DUMMY_RAW_LOG

/// logMeta covers valType and metaType combinations that no real log type uses. Each
/// section constructs a logMeta normally and then uses logMetaExposed to overwrite the
/// protected m_detail with a hand-built accessor of the target C++ type. The entries
/// themselves are real, verifiable state_change flatbuffers built by DEFINE_DUMMY_RAW_LOG,
/// so they pass the merged verifyLogEntry gate. The accessors read the 'to' field back as
/// each type.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "logMeta reaches valType/metaType branches unreachable via any real log type",
           "[libMagAOX::logger::logMeta]" )
{
    using MagAOX::logger::logMeta;
    const time_t base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    SECTION( "Char/State and Char/Continuous" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_char", "devChar", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT buf;
                flatlogs::logHeader::createLog<dummyLogChar>( buf, flatlogs::timespecX( base, 0 ), (char)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogChar>(
                    buf, flatlogs::timespecX( base + 10, 0 ), (char)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devChar"].insert( sfn );

        MagAOX::logger::logMetaSpec lmsState( "devChar", dummyLogChar::eventCode, "state" );
        logMetaExposed               lmetaState( lmsState );
        lmetaState.setDetail( { "", logMeta::valTypes::Char, logMeta::metaTypes::State, reinterpret_cast<void *>( &getCharVal ) } );
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "5" );
        // With stime past the data the lookup fails, which exercises the State failure
        // return for this valType.
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        MagAOX::logger::logMetaSpec lmsCont( "devChar", dummyLogChar::eventCode, "cont", "", "%d", "" );
        logMetaExposed               lmetaCont( lmsCont );
        lmetaCont.setDetail( { "", logMeta::valTypes::Char, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getCharVal ) } );
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // The same lookup with stime past the data fails. This exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        // Char/State has its own verbose braced error reporting form, unlike the other
        // scalar cases. A stime past the last entry in the buffer makes the internal
        // getPriorLog() walk run off the end, as in the getPriorLog tests for logMap.hpp.
        // So getLogStateVal() fails and the error branch of this specific case runs.
        MagAOX::logger::logMetaSpec lmsFail( "devChar", dummyLogChar::eventCode, "state" );
        logMetaExposed               lmetaFail( lmsFail );
        lmetaFail.setDetail( { "", logMeta::valTypes::Char, logMeta::metaTypes::State, reinterpret_cast<void *>( &getCharVal ) } );
        REQUIRE( lmetaFail.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" ); // This is the merged unavailableValue() sentinel. It was "invalid" before the merge.
    }

    SECTION( "UChar/State and UChar/Continuous" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_uchar", "devUChar", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT buf;
                flatlogs::logHeader::createLog<dummyLogUChar>(
                    buf, flatlogs::timespecX( base, 0 ), (unsigned char)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogUChar>(
                    buf, flatlogs::timespecX( base + 10, 0 ), (unsigned char)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devUChar"].insert( sfn );

        MagAOX::logger::logMetaSpec lmsState( "devUChar", dummyLogUChar::eventCode, "state" );
        logMetaExposed               lmetaState( lmsState );
        lmetaState.setDetail( { "", logMeta::valTypes::UChar, logMeta::metaTypes::State, reinterpret_cast<void *>( &getUCharVal ) } );
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "5" );
        // With stime past the data the lookup fails, which exercises the State failure
        // return for this valType.
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        MagAOX::logger::logMetaSpec lmsCont( "devUChar", dummyLogUChar::eventCode, "cont", "", "%u", "" );
        logMetaExposed               lmetaCont( lmsCont );
        lmetaCont.setDetail( { "", logMeta::valTypes::UChar, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getUCharVal ) } );
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // The same lookup with stime past the data fails. This exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "Short/State and Short/Continuous" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_short", "devShort", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT buf;
                flatlogs::logHeader::createLog<dummyLogShort>(
                    buf, flatlogs::timespecX( base, 0 ), (short)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogShort>(
                    buf, flatlogs::timespecX( base + 10, 0 ), (short)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devShort"].insert( sfn );

        MagAOX::logger::logMetaSpec lmsState( "devShort", dummyLogShort::eventCode, "state" );
        logMetaExposed               lmetaState( lmsState );
        lmetaState.setDetail( { "", logMeta::valTypes::Short, logMeta::metaTypes::State, reinterpret_cast<void *>( &getShortVal ) } );
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "5" );
        // With stime past the data the lookup fails, which exercises the State failure
        // return for this valType.
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        MagAOX::logger::logMetaSpec lmsCont( "devShort", dummyLogShort::eventCode, "cont", "", "%d", "" );
        logMetaExposed               lmetaCont( lmsCont );
        lmetaCont.setDetail( { "", logMeta::valTypes::Short, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getShortVal ) } );
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // The same lookup with stime past the data fails. This exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "UShort/State and UShort/Continuous" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_ushort", "devUShort", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT buf;
                flatlogs::logHeader::createLog<dummyLogUShort>(
                    buf, flatlogs::timespecX( base, 0 ), (unsigned short)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogUShort>(
                    buf, flatlogs::timespecX( base + 10, 0 ), (unsigned short)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devUShort"].insert( sfn );

        MagAOX::logger::logMetaSpec lmsState( "devUShort", dummyLogUShort::eventCode, "state" );
        logMetaExposed               lmetaState( lmsState );
        lmetaState.setDetail( { "", logMeta::valTypes::UShort, logMeta::metaTypes::State, reinterpret_cast<void *>( &getUShortVal ) } );
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "5" );
        // With stime past the data the lookup fails, which exercises the State failure
        // return for this valType.
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        MagAOX::logger::logMetaSpec lmsCont( "devUShort", dummyLogUShort::eventCode, "cont", "", "%u", "" );
        logMetaExposed               lmetaCont( lmsCont );
        lmetaCont.setDetail( { "", logMeta::valTypes::UShort, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getUShortVal ) } );
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // The same lookup with stime past the data fails. This exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "Long/State and Long/Continuous" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_long", "devLong", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT buf;
                flatlogs::logHeader::createLog<dummyLogLong>(
                    buf, flatlogs::timespecX( base, 0 ), (long)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogLong>(
                    buf, flatlogs::timespecX( base + 10, 0 ), (long)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devLong"].insert( sfn );

        MagAOX::logger::logMetaSpec lmsState( "devLong", dummyLogLong::eventCode, "state" );
        logMetaExposed               lmetaState( lmsState );
        lmetaState.setDetail( { "", logMeta::valTypes::Long, logMeta::metaTypes::State, reinterpret_cast<void *>( &getLongVal ) } );
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "5" );
        // With stime past the data the lookup fails, which exercises the State failure
        // return for this valType.
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        MagAOX::logger::logMetaSpec lmsCont( "devLong", dummyLogLong::eventCode, "cont", "", "%ld", "" );
        logMetaExposed               lmetaCont( lmsCont );
        lmetaCont.setDetail( { "", logMeta::valTypes::Long, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getLongVal ) } );
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // The same lookup with stime past the data fails. This exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "ULong/State and ULong/Continuous" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_ulong", "devULong", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT buf;
                flatlogs::logHeader::createLog<dummyLogULong>(
                    buf, flatlogs::timespecX( base, 0 ), (unsigned long)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogULong>(
                    buf, flatlogs::timespecX( base + 10, 0 ), (unsigned long)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devULong"].insert( sfn );

        MagAOX::logger::logMetaSpec lmsState( "devULong", dummyLogULong::eventCode, "state" );
        logMetaExposed               lmetaState( lmsState );
        lmetaState.setDetail( { "", logMeta::valTypes::ULong, logMeta::metaTypes::State, reinterpret_cast<void *>( &getULongVal ) } );
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "5" );
        // With stime past the data the lookup fails, which exercises the State failure
        // return for this valType.
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        MagAOX::logger::logMetaSpec lmsCont( "devULong", dummyLogULong::eventCode, "cont", "", "%lu", "" );
        logMetaExposed               lmetaCont( lmsCont );
        lmetaCont.setDetail( { "", logMeta::valTypes::ULong, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getULongVal ) } );
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // The same lookup with stime past the data fails. This exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "LongLong/State and LongLong/Continuous" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_longlong", "devLongLong", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT buf;
                flatlogs::logHeader::createLog<dummyLogLongLong>(
                    buf, flatlogs::timespecX( base, 0 ), (long long)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogLongLong>(
                    buf, flatlogs::timespecX( base + 10, 0 ), (long long)5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devLongLong"].insert( sfn );

        MagAOX::logger::logMetaSpec lmsState( "devLongLong", dummyLogLongLong::eventCode, "state", "", "%lld", "" );
        logMetaExposed               lmetaState( lmsState );
        lmetaState.setDetail( { "", logMeta::valTypes::LongLong, logMeta::metaTypes::State, reinterpret_cast<void *>( &getLongLongVal ) } );
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "5" );
        // With stime past the data the lookup fails, which exercises the State failure
        // return for this valType.
        REQUIRE( lmetaState.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );

        MagAOX::logger::logMetaSpec lmsCont( "devLongLong", dummyLogLongLong::eventCode, "cont", "", "%lld", "" );
        logMetaExposed               lmetaCont( lmsCont );
        lmetaCont.setDetail( { "", logMeta::valTypes::LongLong, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getLongLongVal ) } );
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // The same lookup with stime past the data fails. This exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmetaCont.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "Int/Continuous (State is already covered by real telem_psfacq/telem_teldata)" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_intcont", "devIntCont", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT buf;
                flatlogs::logHeader::createLog<dummyLogIntCont>( buf, flatlogs::timespecX( base, 0 ), 5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogIntCont>(
                    buf, flatlogs::timespecX( base + 10, 0 ), 5, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devIntCont"].insert( sfn );

        MagAOX::logger::logMetaSpec lms( "devIntCont", dummyLogIntCont::eventCode, "cont", "", "%d", "" );
        logMetaExposed               lmeta( lms );
        lmeta.setDetail( { "", logMeta::valTypes::Int, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getIntVal ) } );
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // With stime past the data the lookup fails, which exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "UInt/Continuous (State is already covered by real telem_saving)" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_uintcont", "devUIntCont", base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT      buf;
                unsigned int               v = 5;
                flatlogs::logHeader::createLog<dummyLogUIntCont>( buf, flatlogs::timespecX( base, 0 ), v, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogUIntCont>(
                    buf, flatlogs::timespecX( base + 10, 0 ), v, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devUIntCont"].insert( sfn );

        MagAOX::logger::logMetaSpec lms( "devUIntCont", dummyLogUIntCont::eventCode, "cont", "", "%u", "" );
        logMetaExposed               lmeta( lms );
        lmeta.setDetail( { "", logMeta::valTypes::UInt, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getUIntVal ) } );
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // With stime past the data the lookup fails, which exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "ULongLong/Continuous (State is already covered by real telem_pokeloop)" )
    {
        auto sfn = writeRealLogFile(
            "/tmp/logMeta_test_dead_ulonglongcont",
            "devULongLongCont",
            base,
            [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
                flatlogs::bufferPtrT      buf;
                unsigned long long         v = 5;
                flatlogs::logHeader::createLog<dummyLogULongLongCont>(
                    buf, flatlogs::timespecX( base, 0 ), v, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
                flatlogs::logHeader::createLog<dummyLogULongLongCont>(
                    buf, flatlogs::timespecX( base + 10, 0 ), v, flatlogs::logPrio::LOG_NOTICE );
                writer.writeLog( buf );
            } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap["devULongLongCont"].insert( sfn );

        MagAOX::logger::logMetaSpec lms( "devULongLongCont", dummyLogULongLongCont::eventCode, "cont", "", "%llu", "" );
        logMetaExposed               lmeta( lms );
        lmeta.setDetail( { "", logMeta::valTypes::ULongLong, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getULongLongVal ) } );
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) != "invalid" );
        // With stime past the data the lookup fails, which exercises the Continuous
        // failure return for this valType.
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    }

    SECTION( "String + Continuous: valueString()'s not-state fallback returns empty" )
    {
        // valueString() only touches the logMap when metaType==State, so an empty map is fine.
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::logger::logMetaSpec lms( "devAny", 60011, "cont" ); // arbitrary code above every real event code
        logMetaExposed               lmeta( lms );
        lmeta.setDetail( { "", logMeta::valTypes::String, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &getIntVal ) } );
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) == "" );
    }

    SECTION( "an out-of-range metaType falls through valueNumber()'s final return" )
    {
        // valueNumber() only touches the logMap inside the State/Continuous branches, neither
        // of which matches metaType==99, so an empty map is fine.
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::logger::logMetaSpec lms( "devAny", 60012, "member" ); // arbitrary code above every real event code
        logMetaExposed               lmeta( lms );
        lmeta.setDetail( { "", logMeta::valTypes::Int, 99, reinterpret_cast<void *>( &getIntVal ) } );
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) ==
                 "NOT AVAILABLE" ); // This is the merged unavailableValue() sentinel. It was "invalid" before the merge.
    }

    SECTION( "a valType unhandled by the State switch itself hits its own internal default" )
    {
        // Vector_String is not a case in the State switch of valueNumber(). Only
        // Vector_Bool and Vector_Float are. So this reaches the default of that switch
        // rather than the final fallback after the State and Continuous if-else. The
        // section above with an out-of-range metaType covers that final fallback.
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::logger::logMetaSpec lms( "devAny", 60013, "member" ); // arbitrary code above every real event code
        logMetaExposed               lmeta( lms );
        lmeta.setDetail( { "", logMeta::valTypes::Vector_String, logMeta::metaTypes::State, reinterpret_cast<void *>( &getIntVal ) } );
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) ) ==
                 "NOT AVAILABLE" ); // This is the merged unavailableValue() sentinel. It was "invalid" before the merge.
    }

    SECTION( "an unrecognized event code hits the real logMemberAccessor's own default" )
    {
        MagAOX::logger::logMetaSpec lms( "devAny", 60014, "member" ); // arbitrary code above every real event code
        MagAOX::logger::logMeta     lmeta( lms );
        REQUIRE( lmeta.keyword() == "" );
    }
}


/// Validate shortest-path angle deltas across wrap boundaries.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "Log metadata angle deltas wrap correctly", "[libMagAOX::logger::logMeta]" )
{
    using MagAOX::logger::logMetaAngleDelta;

    REQUIRE( logMetaAngleDelta( 10.0, 20.0 ) == Approx( 10.0 ) );
    REQUIRE( logMetaAngleDelta( 20.0, 10.0 ) == Approx( -10.0 ) );
    REQUIRE( logMetaAngleDelta( -179.0, 179.0 ) == Approx( -2.0 ) );
    REQUIRE( logMetaAngleDelta( 179.0, -179.0 ) == Approx( 2.0 ) );
    REQUIRE( logMetaAngleDelta( 359.0, 1.0 ) == Approx( 2.0 ) );
    REQUIRE( logMetaAngleDelta( 1.0, 359.0 ) == Approx( -2.0 ) );
}

/// Validate interpolated angle normalization across wrap boundaries.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "Log metadata angle interpolation normalizes wrapped midpoints", "[libMagAOX::logger::logMeta]" )
{
    using MagAOX::logger::logMetaAngleDelta;
    using MagAOX::logger::logMetaNormalizeInterpolatedAngle;

    auto midpoint = []( double a0, double a1 )
    {
        double interp = a0 + 0.5 * logMetaAngleDelta( a0, a1 );
        return logMetaNormalizeInterpolatedAngle( interp, a0, a1 );
    };

    REQUIRE( midpoint( -179.0, 179.0 ) == Approx( -180.0 ) );
    REQUIRE( midpoint( 179.0, -179.0 ) == Approx( -180.0 ) );
    REQUIRE( midpoint( 359.0, 1.0 ) == Approx( 0.0 ) );
    REQUIRE( midpoint( 1.0, 359.0 ) == Approx( 0.0 ) );
    REQUIRE( midpoint( 10.0, 20.0 ) == Approx( 15.0 ) );
    REQUIRE( midpoint( 20.0, 10.0 ) == Approx( 15.0 ) );
}

/// Validate angle normalization edge cases.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "Log metadata angle normalization handles exact and repeated wraps", "[libMagAOX::logger::logMeta]" )
{
    using MagAOX::logger::logMetaNormalizeAngle180;
    using MagAOX::logger::logMetaNormalizeAngle360;

    REQUIRE( logMetaNormalizeAngle360( 0.0 ) == Approx( 0.0 ) );
    REQUIRE( logMetaNormalizeAngle360( 360.0 ) == Approx( 0.0 ) );
    REQUIRE( logMetaNormalizeAngle360( 720.0 ) == Approx( 0.0 ) );
    REQUIRE( logMetaNormalizeAngle360( -1.0 ) == Approx( 359.0 ) );

    REQUIRE( logMetaNormalizeAngle180( 180.0 ) == Approx( -180.0 ) );
    REQUIRE( logMetaNormalizeAngle180( 540.0 ) == Approx( -180.0 ) );
    REQUIRE( logMetaNormalizeAngle180( -181.0 ) == Approx( 179.0 ) );
}

/// Validate that parallactic angle metadata uses wrapped interpolation.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "telem_teldata parallactic angle accessor is continuous angle metadata", "[libMagAOX::logger::logMeta]" )
{
    MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_teldata::getAccessor( "pa" );

    REQUIRE( lmd.keyword == "PARANG" );
    REQUIRE( lmd.valType == MagAOX::logger::logMeta::valTypes::Double );
    REQUIRE( lmd.metaType == MagAOX::logger::logMeta::metaTypes::Continuous_Angle );
}


/// The per-type dispatch in verifyLogEntry. One real, minimally populated log entry of
/// every event code in the switch is built, and each must verify against its own schema.
/// A final section checks the null entry, unknown code, and corrupt payload failure paths.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "verifyLogEntry verifies a real entry of every dispatched log type", "[libMagAOX::logger::logMeta]" )
{
    flatlogs::timespecX ts( 1732176000, 0 );

    // Several messageT constructors take vectors by non-const reference. These lvalues
    // are passed to them.
    std::vector<float>       vf;
    std::vector<int64_t>     vi64;
    std::vector<std::string> vs;

// Build one real entry and check it verifies against its own event code.
#define CHECK_VERIFY( TYPE, CODE, MSG )                                                                               \
    {                                                                                                                 \
        flatlogs::bufferPtrT buf;                                                                                     \
        flatlogs::logHeader::createLog<MagAOX::logger::TYPE>( buf, ts, MSG, flatlogs::logPrio::LOG_NOTICE );          \
        REQUIRE( MagAOX::logger::verifyLogEntry( MagAOX::logger::eventCodes::CODE, buf.get() ) );                     \
    }

// empty_log-derived types take an empty messageT.
#define CHECK_VERIFY_EMPTY( TYPE, CODE ) CHECK_VERIFY( TYPE, CODE, MagAOX::logger::TYPE::messageT() )

    CHECK_VERIFY( git_state, GIT_STATE, MagAOX::logger::git_state::messageT( "", "", true ) );
    CHECK_VERIFY( text_log, TEXT_LOG, MagAOX::logger::text_log::messageT( "text" ) );
    CHECK_VERIFY( user_log, USER_LOG, MagAOX::logger::user_log::messageT( "", "" ) );
    CHECK_VERIFY( state_change, STATE_CHANGE, MagAOX::logger::state_change::messageT( (int16_t)1, (int16_t)1 ) );
    // software_log itself has no defaultLevel because it is the abstract base.
    // software_error is a concrete subtype with the same event code and schema.
    CHECK_VERIFY( software_error, SOFTWARE_LOG,
                  MagAOX::logger::software_error::messageT( __FILE__, (uint32_t)1, (int32_t)1, (int32_t)1, "message" ) );
    CHECK_VERIFY( config_log, CONFIG_LOG, MagAOX::logger::config_log::messageT( "", (int)1, "", "" ) );
    CHECK_VERIFY_EMPTY( indidriver_start, INDIDRIVER_START );
    CHECK_VERIFY_EMPTY( indidriver_stop, INDIDRIVER_STOP );
    CHECK_VERIFY_EMPTY( loop_closed, LOOP_CLOSED );
    CHECK_VERIFY_EMPTY( loop_paused, LOOP_PAUSED );
    CHECK_VERIFY_EMPTY( loop_open, LOOP_OPEN );
    CHECK_VERIFY( observer, OBSERVER, MagAOX::logger::observer::messageT( "", "", "", "" ) );
    CHECK_VERIFY( ao_operator, AO_OPERATOR, MagAOX::logger::ao_operator::messageT( "", "", "", "" ) );
    CHECK_VERIFY( pico_channel, PICO_CHANNEL, MagAOX::logger::pico_channel::messageT( "", (uint8_t)1 ) );
    CHECK_VERIFY( outlet_state, OUTLET_STATE, MagAOX::logger::outlet_state::messageT( (uint8_t)1, (uint8_t)1 ) );
    CHECK_VERIFY( outlet_channel_state, OUTLET_CHANNEL_STATE,
                  MagAOX::logger::outlet_channel_state::messageT( "", (uint8_t)1 ) );
    CHECK_VERIFY( telem_saving_state, TELEM_SAVING_STATE,
                  MagAOX::logger::telem_saving_state::messageT( (int16_t)1, (uint64_t)1 ) );
    CHECK_VERIFY( telem_fxngen, TELEM_FXNGEN,
                  MagAOX::logger::telem_fxngen::messageT( (uint8_t)1, 1.0, 1.0, 1.0, 1.0, (uint8_t)1, (uint8_t)1, 1.0,
                                                          1.0, 1.0, 1.0, (uint8_t)1, (uint8_t)1, (uint8_t)1, 1.0, 1.0 ) );
    CHECK_VERIFY( ttmmod_params, TTMMOD_PARAMS,
                  MagAOX::logger::ttmmod_params::messageT( (uint8_t)1, 1.0, 1.0, 1.0, 1.0 ) );
    CHECK_VERIFY( ocam_temps, OCAM_TEMPS,
                  MagAOX::logger::ocam_temps::messageT( 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( cred2_temps, CRED2_TEMPS,
                  MagAOX::logger::cred2_temps::messageT( 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( saving_start, SAVING_START, MagAOX::logger::saving_start::messageT( (int16_t)1, (uint64_t)1 ) );
    CHECK_VERIFY( saving_stop, SAVING_STOP, MagAOX::logger::saving_stop::messageT( (int16_t)1, (uint64_t)1 ) );
    CHECK_VERIFY( telem_saving, TELEM_SAVING,
                  MagAOX::logger::telem_saving::messageT( (uint32_t)1, (uint32_t)1, 1.0f, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_telpos, TELEM_TELPOS,
                  MagAOX::logger::telem_telpos::messageT( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) );
    CHECK_VERIFY( telem_teldata, TELEM_TELDATA,
                  MagAOX::logger::telem_teldata::messageT( 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1 ) );
    CHECK_VERIFY( telem_telvane, TELEM_TELVANE,
                  MagAOX::logger::telem_telvane::messageT( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) );
    CHECK_VERIFY( telem_telenv, TELEM_TELENV,
                  MagAOX::logger::telem_telenv::messageT( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) );
    CHECK_VERIFY( telem_telcat, TELEM_TELCAT,
                  MagAOX::logger::telem_telcat::messageT( "", "", 1.0, 1.0, 1.0, 1.0 ) );
    CHECK_VERIFY( telem_telsee, TELEM_TELSEE, MagAOX::logger::telem_telsee::messageT( 1, 1.0, 1, 1.0, 1, 1.0 ) );
    CHECK_VERIFY( telem_tcsi_tiptilt, TELEM_TCSI_TIPTILT,
                  MagAOX::logger::telem_tcsi_tiptilt::messageT( true, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_tcsi_focus, TELEM_TCSI_FOCUS,
                  MagAOX::logger::telem_tcsi_focus::messageT( true, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_tcsi_labmode, TELEM_TCSI_LABMODE, MagAOX::logger::telem_tcsi_labmode::messageT( true ) );
    CHECK_VERIFY( telem_stage, TELEM_STAGE, MagAOX::logger::telem_stage::messageT( (int8_t)1, 1.0f, "" ) );
    CHECK_VERIFY( telem_zaber, TELEM_ZABER, MagAOX::logger::telem_zaber::messageT( 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_pico, TELEM_PICO, MagAOX::logger::telem_pico::messageT( vi64 ) );
    CHECK_VERIFY( telem_position, TELEM_POSITION, MagAOX::logger::telem_position::messageT( 1.0f ) );
    CHECK_VERIFY( telem_psfacq, TELEM_PSFACQ,
                  MagAOX::logger::telem_psfacq::messageT( 1, 1, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_pokecenter, TELEM_POKECENTER,
                  MagAOX::logger::telem_pokecenter::messageT( (uint8_t)1, 1.0f, 1.0f, std::vector<float>(),
                                                              std::vector<float>() ) );
    CHECK_VERIFY( telem_pokeloop, TELEM_POKELOOP,
                  MagAOX::logger::telem_pokeloop::messageT( (uint8_t)1, 1.0f, 1.0f, (uint64_t)1 ) );
    CHECK_VERIFY( telem_observer, TELEM_OBSERVER,
                  MagAOX::logger::telem_observer::messageT( "", "", true, "", "" ) );
    CHECK_VERIFY( telem_rhusb, TELEM_RHUSB, MagAOX::logger::telem_rhusb::messageT( 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_temps, TELEM_TEMPS, MagAOX::logger::telem_temps::messageT( std::vector<float>() ) );
    CHECK_VERIFY( telem_stdcam, TELEM_STDCAM,
                  MagAOX::logger::telem_stdcam::messageT( "", 1.0f, 1.0f, 1, 1, 1, 1, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                                          1.0f, (uint8_t)1, (uint8_t)1, "", "", (int8_t)1, (uint8_t)1,
                                                          1.0f, (uint8_t)1, "", "", "", (int8_t)1 ) );
    CHECK_VERIFY( telem_coretemps, TELEM_CORETEMPS,
                  MagAOX::logger::telem_coretemps::messageT( vf ) );
    CHECK_VERIFY( telem_coreloads, TELEM_CORELOADS,
                  MagAOX::logger::telem_coreloads::messageT( vf ) );
    CHECK_VERIFY( telem_drivetemps, TELEM_DRIVETEMPS,
                  MagAOX::logger::telem_drivetemps::messageT( vs, vf ) );
    CHECK_VERIFY( telem_usage, TELEM_USAGE, MagAOX::logger::telem_usage::messageT( 1.0f, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_cooler, TELEM_COOLER,
                  MagAOX::logger::telem_cooler::messageT( 1.0f, 1.0f, (uint8_t)1, (uint16_t)1, (uint8_t)1,
                                                          (uint16_t)1 ) );
    CHECK_VERIFY( telem_chrony_status, TELEM_CHRONY_STATUS,
                  MagAOX::logger::telem_chrony_status::messageT( "", "", "", "" ) );
    CHECK_VERIFY( telem_chrony_stats, TELEM_CHRONY_STATS,
                  MagAOX::logger::telem_chrony_stats::messageT( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) );
    CHECK_VERIFY( telem_dmspeck, TELEM_DMSPECK,
                  MagAOX::logger::telem_dmspeck::messageT( true, true, 1.0f, std::vector<float>(),
                                                           std::vector<float>(), std::vector<float>(),
                                                           std::vector<bool>() ) );
    CHECK_VERIFY( telem_fgtimings, TELEM_FGTIMINGS,
                  MagAOX::logger::telem_fgtimings::messageT( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) );
    CHECK_VERIFY( telem_dmmodes, TELEM_DMMODES, MagAOX::logger::telem_dmmodes::messageT( vf ) );
    CHECK_VERIFY( telem_loopgain, TELEM_LOOPGAIN,
                  MagAOX::logger::telem_loopgain::messageT( (uint8_t)1, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_modalgainopt, TELEM_MODALGAINOPT,
                  MagAOX::logger::telem_modalgainopt::messageT( true, true, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_blockgains, TELEM_BLOCKGAINS,
                  MagAOX::logger::telem_blockgains::messageT( std::vector<float>(), std::vector<uint8_t>(),
                                                              std::vector<float>(), std::vector<uint8_t>(),
                                                              std::vector<float>(), std::vector<uint8_t>() ) );
    CHECK_VERIFY( telem_offloading, TELEM_OFFLOADING,
                  MagAOX::logger::telem_offloading::messageT( (uint32_t)1, (uint32_t)1, 1.0f ) );
    CHECK_VERIFY( telem_w2tcsoffloader, TELEM_W2TCSOFFLOADER,
                  MagAOX::logger::telem_w2tcsoffloader::messageT( vf ) );
    CHECK_VERIFY( telem_flowrpm, TELEM_FLOWRPM, MagAOX::logger::telem_flowrpm::messageT( 1.0, 1.0, true ) );
    CHECK_VERIFY( telem_pi335, TELEM_PI335,
                  MagAOX::logger::telem_pi335::messageT( 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f ) );
    CHECK_VERIFY( telem_sparkleclock, TELEM_SPARKLECLOCK,
                  MagAOX::logger::telem_sparkleclock::messageT( true, true, 1.0f, 1.0f, std::vector<float>(), 1.0f,
                                                                1.0f ) );
    CHECK_VERIFY( telem_poltrack, TELEM_POLTRACK,
                  MagAOX::logger::telem_poltrack::messageT( 1.0f, 1.0f, "", true ) );
    CHECK_VERIFY( telem_adctrack, TELEM_ADCTRACK,
                  MagAOX::logger::telem_adctrack::messageT( true, 1.0f, 1.0f, 1.0f, 1.0f ) );

#undef CHECK_VERIFY
#undef CHECK_VERIFY_EMPTY

    SECTION( "a null entry, an unknown event code, and a corrupt payload all fail" )
    {
        REQUIRE( !MagAOX::logger::verifyLogEntry( MagAOX::logger::eventCodes::GIT_STATE, nullptr ) );

        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::git_state>(
            buf, ts, MagAOX::logger::git_state::messageT( "repo", "sha", true ), flatlogs::logPrio::LOG_NOTICE );

        // An unknown code is not a case in the dispatch switch.
        REQUIRE( !MagAOX::logger::verifyLogEntry( 59999, buf.get() ) );

        // Overwrite the message bytes with garbage so the schema verifier genuinely fails
        // for a known code.
        char  *msg = buf.get() + flatlogs::logHeader::headerSize( buf.get() );
        size_t len = flatlogs::logHeader::msgLen( buf.get() );
        memset( msg, 0xFF, len );
        REQUIRE( !MagAOX::logger::verifyLogEntry( MagAOX::logger::eventCodes::GIT_STATE, buf.get() ) );
    }
}


/// The verification failure, gap, and failure reason branches of getLogStateVal and
/// getLogContVal. Each section corrupts the message bytes of specific entries in the
/// loaded buffer. The headers stay sane, so scanning works while schema verification
/// fails.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "getLogStateVal/getLogContVal skip unverifiable entries and honor maxGap", "[libMagAOX::logger::logMeta]" )
{
    const time_t base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    // Corrupts the message payload of entry n in the loaded buffer. Entry numbering
    // starts at 0.
    auto corruptEntry = []( MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> &lm, const std::string &dev, int n ) {
        char *p = lm.m_appToBufferMap[dev].m_memory.data();
        for( int i = 0; i < n; ++i )
        {
            p += flatlogs::logHeader::totalSize( p );
        }
        memset( p + flatlogs::logHeader::headerSize( p ), 0xFF, flatlogs::logHeader::msgLen( p ) );
    };

    SECTION( "an unverifiable prior entry falls back to an earlier verified one" )
    {
        const std::string dev = "devVprior";
        writeLogFile<dummyLogInt, int>( "/tmp/logMeta_vprior", dev, base, { { 0, 1 }, { 10, 2 }, { 20, 3 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_vprior", dev, base );

        // Load the buffer, then corrupt the entry at base+10. That is index 2, after the
        // lead-in text_log.
        int  val;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 15, 0 ),
                                                 flatlogs::timespecX( base + 16, 0 ), getIntVal ) == 0 );
        corruptEntry( lm, dev, 2 );

        std::string reason;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 15, 0 ),
                                                 flatlogs::timespecX( base + 16, 0 ), getIntVal, nullptr, -1, &reason ) == 0 );
        REQUIRE( val == 1 ); // The lookup fell back to the verified entry at base+0.

        // Corrupt the earlier one too, so that no verified prior remains.
        corruptEntry( lm, dev, 1 );
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 15, 0 ),
                                                 flatlogs::timespecX( base + 16, 0 ), getIntVal, nullptr, -1, &reason ) != 0 );
        REQUIRE( reason == "due to unverifiable prior telemetry" );
    }

    SECTION( "an unverifiable following entry is skipped to a later verified one" )
    {
        const std::string dev = "devVnext";
        writeLogFile<dummyLogInt, int>( "/tmp/logMeta_vnext", dev, base,
                                        { { 0, 1 }, { 10, 1 }, { 20, 1 }, { 30, 2 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_vnext", dev, base );

        int val;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 35, 0 ), getIntVal ) == 0 );
        corruptEntry( lm, dev, 2 ); // This is the entry at base+10, which is the first following entry.

        std::string reason;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 35, 0 ), getIntVal, nullptr, -1, &reason ) == 0 );
        REQUIRE( val == 2 ); // The state change at base+30 is still found.

        // Repeat the lookup. The second pass reports the same corrupt entry again, which
        // exercises the report dedupe. The report only prints once per unique entry.
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 35, 0 ), getIntVal ) == 0 );

        // Corrupt everything after the prior, so that no verified following entry remains.
        corruptEntry( lm, dev, 3 );
        corruptEntry( lm, dev, 4 );
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 35, 0 ), getIntVal, nullptr, -1, &reason ) != 0 );
        REQUIRE( reason == "due to unverifiable following telemetry" );
    }

    SECTION( "maxGap rejects lookups across telemetry gaps, with the gap reason recorded" )
    {
        const std::string dev = "devGap";
        writeLogFile<dummyLogInt, int>( "/tmp/logMeta_gap", dev, base, { { 0, 1 }, { 100, 1 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_gap", dev, base );

        int         val;
        std::string reason;

        // The gap between the prior entry at base+0 and stime at base+50 exceeds 10
        // seconds.
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 50, 0 ),
                                                 flatlogs::timespecX( base + 60, 0 ), getIntVal, nullptr, 10.0,
                                                 &reason ) != 0 );
        REQUIRE( reason.find( "due to gap in telemetry exceeding" ) != std::string::npos );

        // A wide open gap allowance succeeds across the same data.
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 50, 0 ),
                                                 flatlogs::timespecX( base + 60, 0 ), getIntVal, nullptr, 1000.0,
                                                 &reason ) == 0 );
        REQUIRE( val == 1 );

        // The gap between the last entry at base+100 and atime at base+300 exceeds 50
        // seconds. The end of data gap check inside the scan fires.
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 99, 0 ),
                                                 flatlogs::timespecX( base + 300, 0 ), getIntVal, nullptr, 50.0,
                                                 &reason ) != 0 );
        REQUIRE( reason.find( "due to gap in telemetry exceeding" ) != std::string::npos );

        // atime is before the following entry, but the gap from the prior to atime exceeds
        // maxGap. The final gap check after the scan loop fires.
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 95, 0 ), getIntVal, nullptr, 60.0,
                                                 &reason ) != 0 );
        REQUIRE( reason.find( "due to gap in telemetry exceeding" ) != std::string::npos );
    }

    SECTION( "getLogContVal skips unverifiable entries and honors maxGap" )
    {
        const std::string dev = "devCgap";
        writeLogFile<dummyLogDouble, double>( "/tmp/logMeta_cgap", dev, base, { { 0, 10.0 }, { 10, 20.0 }, { 20, 30.0 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_cgap", dev, base );

        double      val;
        std::string reason;

        // Do a normal interpolation first. This loads the buffer.
        REQUIRE( MagAOX::logger::getLogContVal( val, lm, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base, 0 ),
                                                flatlogs::timespecX( base + 10, 0 ), getDoubleVal ) == 0 );
        REQUIRE( val > 10.0 );
        REQUIRE( val < 20.0 );

        // The prior of the midpoint is the entry at base+0, which is index 1. Corrupt it.
        // There is no earlier verified entry of this code, so the lookup fails.
        corruptEntry( lm, dev, 1 );
        REQUIRE( MagAOX::logger::getLogContVal( val, lm, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base, 0 ),
                                                flatlogs::timespecX( base + 10, 0 ), getDoubleVal, nullptr, -1,
                                                &reason ) != 0 );
        REQUIRE( reason == "due to unverifiable prior telemetry" );

        // Now test an unverifiable following entry. Load fresh, then corrupt the entry
        // after the prior of the midpoint. The later verified entry at base+20 is used
        // instead.
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm2;
        insertLogFile( lm2, "/tmp/logMeta_cgap", dev, base );
        REQUIRE( MagAOX::logger::getLogContVal( val, lm2, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base, 0 ),
                                                flatlogs::timespecX( base + 10, 0 ), getDoubleVal ) == 0 );
        corruptEntry( lm2, dev, 2 );
        REQUIRE( MagAOX::logger::getLogContVal( val, lm2, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base, 0 ),
                                                flatlogs::timespecX( base + 10, 0 ), getDoubleVal, nullptr, -1,
                                                &reason ) == 0 );

        // With the tail corrupted too, no verified following entry remains.
        corruptEntry( lm2, dev, 3 );
        REQUIRE( MagAOX::logger::getLogContVal( val, lm2, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base, 0 ),
                                                flatlogs::timespecX( base + 10, 0 ), getDoubleVal, nullptr, -1,
                                                &reason ) != 0 );
        REQUIRE( reason == "due to unverifiable following telemetry" );

        // Now test maxGap. The midpoint lands at base+9, whose prior entry is at base+0.
        // That is a 9 second gap against a 0.5 second allowance.
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm3;
        insertLogFile( lm3, "/tmp/logMeta_cgap", dev, base );
        REQUIRE( MagAOX::logger::getLogContVal( val, lm3, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base + 8, 0 ),
                                                flatlogs::timespecX( base + 10, 0 ), getDoubleVal, nullptr, 0.5,
                                                &reason ) != 0 );
        REQUIRE( reason.find( "due to gap in telemetry exceeding" ) != std::string::npos );
    }

    SECTION( "a Continuous_Angle member interpolates with wrap-aware angle math" )
    {
        // Real telem_teldata entries with pa on both sides of the wrap at plus or minus
        // 180. Linear interpolation would give about 0. The wrap-aware midpoint is plus or
        // minus 180.
        const std::string dev = "devPa";
        auto sfn = writeRealLogFile( "/tmp/logMeta_pa", dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
            flatlogs::bufferPtrT buf;
            flatlogs::logHeader::createLog<MagAOX::logger::telem_teldata>(
                buf, flatlogs::timespecX( base, 0 ),
                MagAOX::logger::telem_teldata::messageT( 1, 1, 1, 0, 0, 10.0, 45.0, -179.0, 90.0, 0 ),
                flatlogs::logPrio::LOG_NOTICE );
            writer.writeLog( buf );
            flatlogs::logHeader::createLog<MagAOX::logger::telem_teldata>(
                buf, flatlogs::timespecX( base + 10, 0 ),
                MagAOX::logger::telem_teldata::messageT( 1, 1, 1, 0, 0, 20.0, 45.0, 179.0, 90.0, 0 ),
                flatlogs::logPrio::LOG_NOTICE );
            writer.writeLog( buf );
        } );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        lm.m_appToFileMap[dev].insert( sfn );

        MagAOX::logger::logMetaSpec lms( dev, MagAOX::logger::telem_teldata::eventCode, "pa" );
        MagAOX::logger::logMeta     lmeta( lms );

        std::string val = lmeta.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 10, 0 ) );
        REQUIRE( val != "NOT AVAILABLE" );
        REQUIRE( std::fabs( std::fabs( std::stod( val ) ) - 180.0 ) < 2.0 ); // The midpoint is wrapped, not about 0.
    }

    SECTION( "logMeta's small accessors and the unavailable card" )
    {
        MagAOX::logger::logMetaSpec lms( "devX", MagAOX::logger::git_state::eventCode, "repoName", "KEYW", "%s", "cmt" );
        MagAOX::logger::logMeta     lmeta( lms );
        REQUIRE( lmeta.comment() == "cmt" );
        REQUIRE( lmeta.unavailableReason() == "" );

        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm; // The map is empty, so the lookup fails.
        auto card = lmeta.card( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) );
        REQUIRE( card.valueStr() == "NOT AVAILABLE" );
        REQUIRE( lmeta.unavailableReason() != "" );

        // unavailableCard() is never invoked by card() or value() themselves. It is a
        // separate public entry point. Callers use it to build a placeholder card without
        // attempting a lookup at all, for example for a metadata item known in advance to
        // be unavailable. Call it directly.
        auto unavailCard = lmeta.unavailableCard();
        REQUIRE( unavailCard.valueStr() == MagAOX::logger::logMeta::unavailableValue() );
        REQUIRE( unavailCard.comment() == "cmt" );
    }
}


/// The remaining branches of the verified lookup helpers are called directly. The gap and
/// end of data variants that the sections above cannot reach are also covered.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "verified-lookup helper branches and end-of-data gap variants", "[libMagAOX::logger::logMeta]" )
{
    const time_t base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    // Corrupts the message payload of the entry at p. The header is left intact.
    auto corruptPayload = []( char *p ) {
        memset( p + flatlogs::logHeader::headerSize( p ), 0xFF, flatlogs::logHeader::msgLen( p ) );
    };

    SECTION( "logMetaNormalizeAngle360 wraps a tiny negative to exactly 360" )
    {
        // fmod(-1e-20, 360) is -1e-20. Adding 360 rounds to exactly 360.0 in doubles.
        // That result must normalize to 0.
        REQUIRE( MagAOX::logger::logMetaNormalizeAngle360( -1e-20 ) == 0.0 );
    }

    SECTION( "reportUnverifiableLogEntry prints every optional-field variant" )
    {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<dummyLogInt>( buf, flatlogs::timespecX( base, 0 ), 1,
                                                     flatlogs::logPrio::LOG_NOTICE );

        // With no failure entry at all, the dedupe bookkeeping is skipped and failureTs
        // prints as <none>.
        MagAOX::logger::reportUnverifiableLogEntry( "devR", 20, nullptr, nullptr, nullptr, "src", 0, "test" );

        // Use an unknown source byte and no before or after entries.
        MagAOX::logger::reportUnverifiableLogEntry(
            "devR", 20, nullptr, buf.get(), nullptr, "src", std::numeric_limits<size_t>::max(), "test" );

        // Use the full form, with before and after entries present and a known byte.
        MagAOX::logger::reportUnverifiableLogEntry( "devR", 20, buf.get(), buf.get(), buf.get(), "src", 5, "test2" );
        REQUIRE( true ); // This helper only produces output. Reaching here without aborting is the test.
    }

    SECTION( "getPriorVerifiedLog and getNextVerifiedLog reject missing inputs" )
    {
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        REQUIRE( MagAOX::logger::getPriorVerifiedLog( lm, "no-such-app", 20, flatlogs::timespecX( base, 0 ) ) ==
                 nullptr );
        REQUIRE( MagAOX::logger::getNextVerifiedLog( lm, nullptr, "no-such-app" ) == nullptr );
    }

    SECTION( "getPriorVerifiedLog resyncs past a header-corrupt entry and stops at the query time" )
    {
        const std::string dev = "devPVL";
        writeLogFile<dummyLogInt, int>( "/tmp/logMeta_pvl", dev, base, { { 0, 1 }, { 10, 2 }, { 20, 3 }, { 30, 4 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_pvl", dev, base );

        int val;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 25, 0 ),
                                                 flatlogs::timespecX( base + 26, 0 ), getIntVal ) == 0 );

        // The entry layout is index 0 for the lead-in text, then base+0, base+10,
        // base+20, and base+30 at indices 1 to 4.
        std::vector<char> &mem = lm.m_appToBufferMap[dev].m_memory;
        char *p = mem.data();
        std::vector<char *> e;
        while( p < mem.data() + mem.size() )
        {
            e.push_back( p );
            p += flatlogs::logHeader::totalSize( p );
        }

        // Corrupting the payload at base+20 makes the prior for ts=25 unverifiable.
        // Setting an invalid priority at base+10 makes that header corrupt, which forces
        // a resync in the scan of getPriorVerifiedLog. The entry at base+30 is beyond the
        // query time, which exercises the early return with the prior found so far.
        corruptPayload( e[3] );
        e[2][0] = 30;

        std::string reason;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 25, 0 ),
                                                 flatlogs::timespecX( base + 26, 0 ), getIntVal, nullptr, -1,
                                                 &reason ) == 0 );
        REQUIRE( val == 1 ); // The lookup fell back to the entry at base+0.
    }

    SECTION( "getPriorVerifiedLog gives up when a corrupt tail cannot be resynced" )
    {
        const std::string dev = "devPVL2";
        writeLogFile<dummyLogInt, int>( "/tmp/logMeta_pvl2", dev, base, { { 0, 1 }, { 10, 2 }, { 20, 3 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_pvl2", dev, base );

        int val;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 15, 0 ),
                                                 flatlogs::timespecX( base + 16, 0 ), getIntVal ) == 0 );

        // The entry layout is index 0 for the lead-in text, then base+0, base+10, and
        // base+20 at indices 1 to 3.
        std::vector<char> &mem = lm.m_appToBufferMap[dev].m_memory;
        char *p = mem.data();
        std::vector<char *> e;
        while( p < mem.data() + mem.size() )
        {
            e.push_back( p );
            p += flatlogs::logHeader::totalSize( p );
        }
        // Corrupting the payload at base+10 makes the prior for ts=15 unverifiable.
        // Setting an invalid priority at base+20 makes the header of the last entry
        // corrupt. The resync in the fallback scan has nothing valid after it and gives
        // up. It returns the best verified prior found so far.
        corruptPayload( e[2] );
        e[3][0] = 30;

        // The fallback scan runs, gives up during the resync, and returns the verified
        // prior at base+0. The lookup as a whole still fails afterwards. With base+10
        // unverifiable and base+20 corrupt there is no verified following entry.
        std::string reason;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 15, 0 ),
                                                 flatlogs::timespecX( base + 16, 0 ), getIntVal, nullptr, -1,
                                                 &reason ) != 0 );
        REQUIRE( reason == "due to unverifiable following telemetry" );
    }

    SECTION( "gap checks at the end of the data" )
    {
        const std::string dev = "devEGap";
        writeLogFile<dummyLogInt, int>( "/tmp/logMeta_egap", dev, base, { { 0, 1 }, { 100, 1 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_egap", dev, base );

        int         val;
        std::string reason;

        // The prior is the last entry. The very first getNextLog hits the end of the data,
        // and the gap check from the prior to atime fires.
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 105, 0 ),
                                                 flatlogs::timespecX( base + 300, 0 ), getIntVal, nullptr, 10.0,
                                                 &reason ) != 0 );
        REQUIRE( reason.find( "due to gap in telemetry exceeding" ) != std::string::npos );

        // The loop is entered because a following entry is within atime. The gap between
        // consecutive entries exceeds the allowance.
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 300, 0 ), getIntVal, nullptr, 10.0,
                                                 &reason ) != 0 );
        REQUIRE( reason.find( "due to gap in telemetry exceeding" ) != std::string::npos );

        // The loop walks to the last entry, which has the same value. Then it hits the end
        // of the data with a gap to atime.
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 300, 0 ), getIntVal, nullptr, 150.0,
                                                 &reason ) != 0 );
        REQUIRE( reason.find( "due to gap in telemetry exceeding" ) != std::string::npos );
    }

    SECTION( "an unverifiable entry found mid-walk fails when nothing verified follows" )
    {
        const std::string dev = "devMWalk";
        writeLogFile<dummyLogInt, int>( "/tmp/logMeta_mwalk", dev, base, { { 0, 1 }, { 10, 1 }, { 20, 1 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_mwalk", dev, base );

        int val;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 300, 0 ), getIntVal ) != 0 );

        // Corrupt the last entry. The walk advances from base+0 to base+10, which has the
        // same value. Then the in-loop verify of base+20 fails with nothing verified after
        // it.
        std::vector<char> &mem = lm.m_appToBufferMap[dev].m_memory;
        char *p = mem.data();
        for( int i = 0; i < 3; ++i )
        {
            p += flatlogs::logHeader::totalSize( p );
        }
        corruptPayload( p );

        std::string reason;
        REQUIRE( MagAOX::logger::getLogStateVal( val, lm, dev, dummyLogInt::eventCode, flatlogs::timespecX( base + 5, 0 ),
                                                 flatlogs::timespecX( base + 300, 0 ), getIntVal, nullptr, -1,
                                                 &reason ) != 0 );
        REQUIRE( reason == "due to unverifiable following telemetry" );
    }

    SECTION( "getLogContVal end-of-data and midpoint-to-following gap failures" )
    {
        const std::string dev = "devCEnd";
        writeLogFile<dummyLogDouble, double>( "/tmp/logMeta_cend", dev, base, { { 0, 10.0 }, { 100, 20.0 } } );
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        insertLogFile( lm, "/tmp/logMeta_cend", dev, base );

        double      val;
        std::string reason;

        // The prior of the midpoint is the last entry, so no following entry exists.
        REQUIRE( MagAOX::logger::getLogContVal( val, lm, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base + 105, 0 ),
                                                flatlogs::timespecX( base + 115, 0 ), getDoubleVal, nullptr, -1,
                                                &reason ) != 0 );
        REQUIRE( reason == "due to missing following telemetry" );

        // The gap from the midpoint to the following entry exceeds the allowance. The gap
        // from the prior to the midpoint is fine.
        REQUIRE( MagAOX::logger::getLogContVal( val, lm, dev, dummyLogDouble::eventCode, flatlogs::timespecX( base + 1, 0 ),
                                                flatlogs::timespecX( base + 3, 0 ), getDoubleVal, nullptr, 5.0,
                                                &reason ) != 0 );
        REQUIRE( reason.find( "due to gap in telemetry exceeding" ) != std::string::npos );
    }
}


/// Vector members whose stored vectors are empty format as the empty string.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "empty vector members format as an empty value", "[libMagAOX::logger::logMeta]" )
{
    const std::string dev  = "devEmptyVec";
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeRealLogFile( "/tmp/logMeta_emptyvec", dev, base, [&]( MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> &writer ) {
        flatlogs::bufferPtrT buf;
        for( int i = 0; i < 2; ++i )
        {
            flatlogs::logHeader::createLog<MagAOX::logger::telem_dmspeck>(
                buf, flatlogs::timespecX( base + 10 * i, 0 ),
                MagAOX::logger::telem_dmspeck::messageT( true, false, 1.0f, std::vector<float>(),
                                                         std::vector<float>(), std::vector<float>(),
                                                         std::vector<bool>() ),
                flatlogs::logPrio::LOG_NOTICE );
            writer.writeLog( buf );
        }
    } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap[dev].insert( sfn );

    MagAOX::logger::logMetaSpec lmsB( dev, MagAOX::logger::telem_dmspeck::eventCode, "crosses" );
    MagAOX::logger::logMeta     lmB( lmsB );
    REQUIRE( lmB.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "" );

    MagAOX::logger::logMetaSpec lmsF( dev, MagAOX::logger::telem_dmspeck::eventCode, "separations" );
    MagAOX::logger::logMeta     lmF( lmsF );
    REQUIRE( lmF.value( lm, flatlogs::timespecX( base, 0 ), flatlogs::timespecX( base + 5, 0 ) ) == "" );
}


/// The failure returns of the remaining valType cases. Their success paths are covered by
/// the real-type tests above. A lookup past the end of the data fails identically
/// regardless of valType, so one shared file drives the failure return of each case.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "remaining valType failure returns", "[libMagAOX::logger::logMeta]" )
{
    using MagAOX::logger::logMeta;
    const time_t      base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day
    const std::string dev  = "devVFail";

    writeLogFile<dummyLogInt, int>( "/tmp/logMeta_vfail", dev, base, { { 0, 5 }, { 10, 5 } } );
    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    insertLogFile( lm, "/tmp/logMeta_vfail", dev, base );

    auto checkFail = [&]( int valType, int metaType ) {
        MagAOX::logger::logMetaSpec lms( dev, dummyLogInt::eventCode, "m", "", "%d", "" );
        logMetaExposed               lmeta( lms );
        lmeta.setDetail( { "", valType, metaType, reinterpret_cast<void *>( &getIntVal ) } );
        REQUIRE( lmeta.value( lm, flatlogs::timespecX( base + 11, 0 ), flatlogs::timespecX( base + 1000, 0 ) ) ==
                 "NOT AVAILABLE" );
    };

    checkFail( logMeta::valTypes::Int, logMeta::metaTypes::State );
    checkFail( logMeta::valTypes::Float, logMeta::metaTypes::State );
    checkFail( logMeta::valTypes::Double, logMeta::metaTypes::State );
    checkFail( logMeta::valTypes::Bool, logMeta::metaTypes::Continuous );
    checkFail( logMeta::valTypes::Float, logMeta::metaTypes::Continuous );
    checkFail( logMeta::valTypes::Double, logMeta::metaTypes::Continuous );
}

} // namespace logMetaTest
} // namespace loggerTest
} // namespace libXWCTest
