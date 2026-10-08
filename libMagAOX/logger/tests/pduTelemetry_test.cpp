/** \file pduTelemetry_test.cpp
 * \brief Outlet/PDU FlatBuffer verification, accessors, generated dispatch, and FITS cards.
 * \ingroup logger_files
 */
#include "../../../tests/testXWC.hpp"
#include "../../libMagAOX.hpp"
#include "../generated/logMemberAccessor.hpp"

namespace libXWCTest
{
namespace loggerTest
{
/** \defgroup pduTelemetry_unit_test Outlet and PDU Telemetry Tests
 * \ingroup unit_test
 */
namespace pduTelemetryTest
{
using namespace MagAOX::logger;
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Append an actual flatlog envelope to an in-memory device log for real metadata lookup.
template <class Type>
void append( logMap<> &map /**< [in/out] in-memory log map */,
             const std::string &device /**< [in] app name */,
             const typename Type::messageT &message /**< [in] actual FlatBuffer input */,
             unsigned seconds /**< [in] record timestamp */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE(flatlogs::logHeader::createLog<Type>(buffer,{seconds,0},message,flatlogs::logPrio::LOG_TELEM)==0);
    map.m_appToFileMap[device].emplace(device+"_20260101000000000000000.bintel");
    auto &memory=map.m_appToBufferMap[device];
    auto size=flatlogs::logHeader::totalSize(buffer);
    memory.m_memory.insert(memory.m_memory.end(),buffer.get(),buffer.get()+size);
    memory.m_startTime=flatlogs::timespecX(0,0); memory.m_endTime=flatlogs::timespecX(100,0);
}
/// \endcond

/// Round-trip signed outlet states, including arbitrary vector length and missing/empty vectors.
/** \ingroup pduTelemetry_unit_test */
TEST_CASE("Outlet telemetry retains numeric signed states and numbering", "[pduTelemetry]")
{
    for(const auto &states : std::vector<std::vector<int8_t>>{{},{-1},{-1,0,1,2},std::vector<int8_t>(300,2)})
    {
        telem_outlet::messageT message(1,states);
        auto *payload=message.builder.GetBufferPointer();
        REQUIRE(telem_outlet::first_outlet(payload)==1);
        REQUIRE(telem_outlet::states(payload)==states);
        REQUIRE(telem_outlet::msgString(payload,message.builder.GetSize()).starts_with("[outlet] first: 1 states: "));
        flatlogs::bufferPtrT buffer;
        REQUIRE(flatlogs::logHeader::createLog<telem_outlet>(buffer,{1,0},message,flatlogs::logPrio::LOG_TELEM)==0);
        REQUIRE(telem_outlet::verify(buffer,flatlogs::logHeader::msgLen(buffer)));
        REQUIRE(verifyLogEntry(telem_outlet::eventCode,buffer.get()));
        std::memset(flatlogs::logHeader::messageBuffer(buffer),0xff,4);
        REQUIRE(!telem_outlet::verify(buffer,flatlogs::logHeader::msgLen(buffer)));
    }
    telem_outlet::messageT missing(0,{});
    missing.builder.Clear(); missing.builder.Finish(CreateTelem_outlet_fb(missing.builder,0));
    REQUIRE(telem_outlet::states(missing.builder.GetBufferPointer()).empty());
    REQUIRE(telem_outlet::stateString(missing.builder.GetBufferPointer()).empty());
    telem_outlet::messageT message(0,{-1,0,1,2});
    REQUIRE(telem_outlet::stateString(message.builder.GetBufferPointer())=="-1,0,1,2");
    REQUIRE(telem_outlet::getAccessor("absent").accessor==nullptr);
    REQUIRE(logMemberAccessor(telem_outlet::eventCode,"states").valType==logMeta::String);
    REQUIRE(logMemberAccessor(telem_outlet::eventCode,"first_outlet").valType==logMeta::UChar);
}

/// Verify electrical measurements, validity, units, all accessors, and corrupted-buffer rejection.
/** \ingroup pduTelemetry_unit_test */
TEST_CASE("Electrical telemetry verifies samples and metadata dispatch", "[pduTelemetry]")
{
    for(bool valid : {false,true})
    {
        telem_pdu::messageT message(60,120,4,valid);
        auto *payload=message.builder.GetBufferPointer();
        REQUIRE(telem_pdu::frequency(payload)==60); REQUIRE(telem_pdu::voltage(payload)==120);
        REQUIRE(telem_pdu::current(payload)==4); REQUIRE(telem_pdu::valid(payload)==valid);
        REQUIRE(telem_pdu::msgString(payload,message.builder.GetSize()).find(" Hz ")!=std::string::npos);
        flatlogs::bufferPtrT buffer;
        REQUIRE(flatlogs::logHeader::createLog<telem_pdu>(buffer,{1,0},message,flatlogs::logPrio::LOG_TELEM)==0);
        REQUIRE(telem_pdu::verify(buffer,flatlogs::logHeader::msgLen(buffer)));
        REQUIRE(verifyLogEntry(telem_pdu::eventCode,buffer.get()));
        std::memset(flatlogs::logHeader::messageBuffer(buffer),0xff,4);
        REQUIRE(!telem_pdu::verify(buffer,flatlogs::logHeader::msgLen(buffer)));
    }
    for(auto field : {"frequency","voltage","current","valid"})
    {
        auto metadata=logMemberAccessor(telem_pdu::eventCode,field);
        REQUIRE(metadata.accessor!=nullptr); REQUIRE(metadata.metaType==logMeta::State);
    }
    REQUIRE(telem_pdu::getAccessor("absent").accessor==nullptr);
}

/// Create real FITS cards using verified logs with state-based selection and large outlet strings.
/** \ingroup pduTelemetry_unit_test */
TEST_CASE("PDU telemetry forms actual FITS metadata cards", "[pduTelemetry]")
{
    logMap<> map;
    telem_outlet::messageT outlets(1,{-1,0,1,2});
    append<telem_outlet>(map,"vpdu",outlets,1);
    append<telem_outlet>(map,"vpdu",outlets,20);
    char *prior=nullptr;
    auto *start=map.m_appToBufferMap["vpdu"].m_memory.data();
    INFO("event " << flatlogs::logHeader::eventCode(start) << " time " << flatlogs::logHeader::timespec(start).time_s
         << " priority " << int(flatlogs::logHeader::logLevel(start)) << " size " << flatlogs::logHeader::totalSize(start)
         << " bytes " << map.m_appToBufferMap["vpdu"].m_memory.size());
    REQUIRE(map.getPriorLog(prior,"vpdu",telem_outlet::eventCode,{5,0})==0);
    size_t envelopeSize=0;
    REQUIRE(logMapEntrySane(envelopeSize,map.m_appToBufferMap["vpdu"].m_memory.data(),
                           map.m_appToBufferMap["vpdu"].m_memory.data()+map.m_appToBufferMap["vpdu"].m_memory.size()));
    logMeta states({"vpdu",telem_outlet::eventCode,"states"});
    auto card=states.card(map,{5,0},{10,0});
    REQUIRE(card.keyword().find("vpdu OUTLET STATES")!=std::string::npos);
    INFO(states.unavailableReason());
    REQUIRE(card.valueStr()=="-1,0,1,2");
    logMeta first({"vpdu",telem_outlet::eventCode,"first_outlet"});
    REQUIRE(first.card(map,{5,0},{10,0}).valueStr()=="1");
    telem_pdu::messageT electrical(60,120,4,true);
    append<telem_pdu>(map,"pdu",electrical,1);
    append<telem_pdu>(map,"pdu",electrical,20);
    for(const auto &[field,expected] : std::vector<std::pair<std::string,std::string>>{
        {"frequency","60"},{"voltage","120"},{"current","4"},{"valid","1"}})
    {
        logMeta value({"pdu",telem_pdu::eventCode,field});
        auto measured=value.card(map,{5,0},{10,0});
        REQUIRE(measured.valueStr()==expected);
    }
    logMap<> large;
    telem_outlet::messageT many(0,std::vector<int8_t>(300,-1));
    append<telem_outlet>(large,"vpdu",many,1); append<telem_outlet>(large,"vpdu",many,20);
    logMeta list({"vpdu",telem_outlet::eventCode,"states"});
    REQUIRE(list.card(large,{5,0},{10,0}).valueStr()==telem_outlet::stateString(many.builder.GetBufferPointer()));
}
} // namespace pduTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
