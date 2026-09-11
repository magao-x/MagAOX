//#define CATCH_CONFIG_MAIN
/** \file indiUtils_test.cpp
  * \brief Catch2 tests for the MagAOX::app::indi utility functions.
  *
  * Covers parseIndiKey(), addTextElement(), addNumberElement(), and the family of
  * updateIfChanged() helpers. The update helpers are exercised against real pcf::IndiProperty
  * objects and a fakeIndiDriver stand-in that counts sends and can be made to throw.
  * No INDI server, shared memory, or files are needed.
  */
#include "../../../tests/catch2/catch.hpp"

#include "../indiUtils.hpp"
using namespace MagAOX::app::indi;

namespace
{

/// Minimal stand-in for a MagAOX INDI driver.
/// It satisfies the interface required by the updateXXXIfChanged() and updatesIfChanged()
/// functions, which is just a sendSetProperty(pcf::IndiProperty&) method. It counts sends and
/// keeps a copy of the last property sent. It can be told to throw on send, either a
/// std::exception or a non-standard type, so tests can exercise the catch blocks in the
/// functions under test. This is ordinary dependency-injection test-doubling. The code under
/// test is not modified.
struct fakeIndiDriver
{
    int                sendCount{ 0 };
    pcf::IndiProperty  lastSent;
    bool               throwOnSend{ false };
    bool               throwNonStd{ false };

    void sendSetProperty( pcf::IndiProperty &p )
    {
        if( throwOnSend )
        {
            if( throwNonStd )
            {
                throw 42;
            }
            throw std::runtime_error( "forced send failure" );
        }

        ++sendCount;
        lastSent = p;
    }
};

} // namespace


SCENARIO( "Parsing INDI unique key", "[indiUtils]" )
{
    GIVEN("valid keys")
    {
        WHEN("standard dev.prop")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "dev.prop" );

            REQUIRE( rv == 0 );
            REQUIRE( devName == "dev" );
            REQUIRE( propName == "prop" );
        }


    }

    GIVEN("invalid keys")
    {
        WHEN("empty")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "" );

            REQUIRE( rv == -1 );
        }

        WHEN(". only")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "." );

            REQUIRE( rv == -1 );
        }

        WHEN("no .")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "nada" );

            REQUIRE( rv == -2 );
        }

        WHEN("dev.")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "dev." );

            REQUIRE( rv == -4 );
        }

        WHEN(".prop")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, ".prop" );

            REQUIRE( rv == -3 );
        }
    }
}

/// Verify that addTextElement() adds a text element to a property, with and without a label.
SCENARIO( "Adding a standard INDI text element", "[indiUtils]" )
{
    GIVEN( "a property" )
    {
        pcf::IndiProperty prop;

        WHEN( "adding an element without a label" )
        {
            int rv = addTextElement( prop, "elname" );

            THEN( "it succeeds and the label is left blank" )
            {
                REQUIRE( rv == 0 );
                REQUIRE( prop.find( "elname" ) == true );
                REQUIRE( prop["elname"].getLabel() == "" );
            }
        }

        WHEN( "adding an element with a label" )
        {
            int rv = addTextElement( prop, "elname", "El Label" );

            THEN( "it succeeds and the label is set" )
            {
                REQUIRE( rv == 0 );
                REQUIRE( prop.find( "elname" ) == true );
                REQUIRE( prop["elname"].getLabel() == "El Label" );
            }
        }
    }
}

/// Verify that addNumberElement() adds a number element and records its min, max, step,
/// format, and optional label. Both the int and double template instantiations are covered.
SCENARIO( "Adding a standard INDI number element", "[indiUtils]" )
{
    GIVEN( "a property" )
    {
        pcf::IndiProperty prop;

        WHEN( "adding an element without a label" )
        {
            int rv = addNumberElement<double>( prop, "elname", -1.0, 1.0, 0.1, "%f" );

            THEN( "it succeeds, min/max/step/format are set, and the label is blank" )
            {
                REQUIRE( rv == 0 );
                REQUIRE( prop.find( "elname" ) == true );
                REQUIRE( prop["elname"].getMin() == "-1" );
                REQUIRE( prop["elname"].getMax() == "1" );
                REQUIRE( prop["elname"].getStep() == "0.1" );
                REQUIRE( prop["elname"].getFormat() == "%f" );
                REQUIRE( prop["elname"].getLabel() == "" );
            }
        }

        WHEN( "adding an element with a label, T=int" )
        {
            int rv = addNumberElement<int>( prop, "elname", 0, 100, 1, "%d", "El Label" );

            THEN( "it succeeds and the label is set" )
            {
                REQUIRE( rv == 0 );
                REQUIRE( prop.find( "elname" ) == true );
                REQUIRE( prop["elname"].getLabel() == "El Label" );
            }
        }

        WHEN( "adding an element with a label, T=double" )
        {
            int rv = addNumberElement<double>( prop, "elname", -1.0, 1.0, 0.1, "%f", "El Label" );

            THEN( "it succeeds and the label is set" )
            {
                REQUIRE( rv == 0 );
                REQUIRE( prop.find( "elname" ) == true );
                REQUIRE( prop["elname"].getLabel() == "El Label" );
            }
        }
    }
}

/// Verify the single-element updateIfChanged(). A fakeIndiDriver counts sends. The cases are a
/// null driver, no change, a value change, a state-only change, a missing element name, and
/// a driver that throws on send.
SCENARIO( "updateIfChanged for a single element", "[indiUtils]" )
{
    GIVEN( "a property with one numeric element" )
    {
        pcf::IndiProperty prop;
        prop.setName( "prop" );
        prop.add( pcf::IndiElement( "val" ) );
        prop["val"].setValue( 1.0 );
        prop.setState( pcf::IndiProperty::Ok );

        WHEN( "the driver pointer is null" )
        {
            fakeIndiDriver *nullDriver = nullptr;
            updateIfChanged( prop, "val", 2.0, nullDriver );

            THEN( "nothing happens" )
            {
                REQUIRE( prop["val"].getValue() == "1" );
            }
        }

        WHEN( "neither the value nor the state changes" )
        {
            fakeIndiDriver driver;
            updateIfChanged( prop, "val", 1.0, &driver, pcf::IndiProperty::Ok );

            THEN( "no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the value changes" )
        {
            fakeIndiDriver driver;
            updateIfChanged( prop, "val", 5.0, &driver, pcf::IndiProperty::Ok );

            THEN( "an update is sent and the value is updated" )
            {
                REQUIRE( driver.sendCount == 1 );
                REQUIRE( prop["val"].getValue() == "5" );
            }
        }

        WHEN( "only the state changes" )
        {
            fakeIndiDriver driver;
            updateIfChanged( prop, "val", 1.0, &driver, pcf::IndiProperty::Busy );

            THEN( "an update is sent" )
            {
                REQUIRE( driver.sendCount == 1 );
                REQUIRE( prop.getState() == pcf::IndiProperty::Busy );
            }
        }

        WHEN( "the element name does not exist in the property" )
        {
            fakeIndiDriver driver;
            updateIfChanged( prop, "nonexistent", 2.0, &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught and no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the driver throws a non-standard exception on send" )
        {
            fakeIndiDriver driver;
            driver.throwOnSend = true;
            driver.throwNonStd = true;
            updateIfChanged( prop, "val", 5.0, &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }
    }
}

/// Verify the vector form of updateIfChanged() with the same set of cases as the single-element
/// form. One changed value out of three must trigger exactly one send.
SCENARIO( "updateIfChanged for a vector of elements", "[indiUtils]" )
{
    GIVEN( "a property with three numeric elements" )
    {
        pcf::IndiProperty prop;
        prop.setName( "prop" );
        prop.add( pcf::IndiElement( "e0" ) );
        prop.add( pcf::IndiElement( "e1" ) );
        prop.add( pcf::IndiElement( "e2" ) );
        prop["e0"].setValue( 0.0 );
        prop["e1"].setValue( 1.0 );
        prop["e2"].setValue( 2.0 );
        prop.setState( pcf::IndiProperty::Ok );

        std::vector<std::string> els( { "e0", "e1", "e2" } );

        WHEN( "the driver pointer is null" )
        {
            fakeIndiDriver *nullDriver = nullptr;
            updateIfChanged( prop, els, std::vector<double>( { 9.0, 9.0, 9.0 } ), nullDriver );

            THEN( "nothing happens" )
            {
                REQUIRE( prop["e0"].getValue() == "0" );
            }
        }

        WHEN( "no values or state change" )
        {
            fakeIndiDriver driver;
            updateIfChanged( prop, els, std::vector<double>( { 0.0, 1.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "only the state changes" )
        {
            fakeIndiDriver driver;
            updateIfChanged( prop, els, std::vector<double>( { 0.0, 1.0, 2.0 } ), &driver, pcf::IndiProperty::Busy );

            THEN( "an update is sent" )
            {
                REQUIRE( driver.sendCount == 1 );
            }
        }

        WHEN( "one of the values changes" )
        {
            fakeIndiDriver driver;
            updateIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "an update is sent and all values are set" )
            {
                REQUIRE( driver.sendCount == 1 );
                REQUIRE( prop["e1"].getValue() == "42" );
            }
        }

        WHEN( "one of the element names does not exist" )
        {
            fakeIndiDriver           driver;
            std::vector<std::string> badEls( { "e0", "nonexistent", "e2" } );
            updateIfChanged( prop, badEls, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught and no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the driver throws a std::exception on send" )
        {
            fakeIndiDriver driver;
            driver.throwOnSend = true;
            updateIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the driver throws a non-standard exception on send" )
        {
            fakeIndiDriver driver;
            driver.throwOnSend  = true;
            driver.throwNonStd  = true;
            updateIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

    }
}

/// Verify updatesIfChanged() with std::vector<std::string> element names.
/// updatesIfChanged() is templated on the element-name container type. It is exposed through two
/// thin overloads, one for std::vector<std::string> and one for std::vector<const char *>. Each
/// overload forwards to the generic template and is a separate instantiation. Both are
/// therefore exercised with the full set of branches, here and in the next scenario.
SCENARIO( "updatesIfChanged with std::string element names", "[indiUtils]" )
{
    GIVEN( "a property with three numeric elements" )
    {
        pcf::IndiProperty prop;
        prop.setName( "prop" );
        prop.add( pcf::IndiElement( "e0" ) );
        prop.add( pcf::IndiElement( "e1" ) );
        prop.add( pcf::IndiElement( "e2" ) );
        prop["e0"].setValue( 0.0 );
        prop["e1"].setValue( 1.0 );
        prop["e2"].setValue( 2.0 );
        prop.setState( pcf::IndiProperty::Ok );

        std::vector<std::string> els( { "e0", "e1", "e2" } );

        WHEN( "the driver pointer is null" )
        {
            fakeIndiDriver *nullDriver = nullptr;
            updatesIfChanged( prop, els, std::vector<double>( { 9.0, 9.0, 9.0 } ), nullDriver );

            THEN( "nothing happens" )
            {
                REQUIRE( prop["e0"].getValue() == "0" );
            }
        }

        WHEN( "no values or state change" )
        {
            fakeIndiDriver driver;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 1.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "only the state changes" )
        {
            fakeIndiDriver driver;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 1.0, 2.0 } ), &driver, pcf::IndiProperty::Busy );

            THEN( "an update is sent" )
            {
                REQUIRE( driver.sendCount == 1 );
            }
        }

        WHEN( "one of the values changes" )
        {
            fakeIndiDriver driver;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "an update is sent and all values are set" )
            {
                REQUIRE( driver.sendCount == 1 );
                REQUIRE( prop["e1"].getValue() == "42" );
            }
        }

        WHEN( "one of the element names does not exist" )
        {
            fakeIndiDriver           driver;
            std::vector<std::string> badEls( { "e0", "nonexistent", "e2" } );
            updatesIfChanged( prop, badEls, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught and no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the driver throws a std::exception on send" )
        {
            fakeIndiDriver driver;
            driver.throwOnSend = true;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the driver throws a non-standard exception on send" )
        {
            fakeIndiDriver driver;
            driver.throwOnSend = true;
            driver.throwNonStd = true;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

    }
}

/// Verify updatesIfChanged() with std::vector<const char *> element names. The cases mirror
/// the std::string scenario above.
SCENARIO( "updatesIfChanged with const char* element names", "[indiUtils]" )
{
    GIVEN( "a property with three numeric elements" )
    {
        pcf::IndiProperty prop;
        prop.setName( "prop" );
        prop.add( pcf::IndiElement( "e0" ) );
        prop.add( pcf::IndiElement( "e1" ) );
        prop.add( pcf::IndiElement( "e2" ) );
        prop["e0"].setValue( 0.0 );
        prop["e1"].setValue( 1.0 );
        prop["e2"].setValue( 2.0 );
        prop.setState( pcf::IndiProperty::Ok );

        std::vector<const char *> els( { "e0", "e1", "e2" } );

        WHEN( "the driver pointer is null" )
        {
            fakeIndiDriver *nullDriver = nullptr;
            updatesIfChanged( prop, els, std::vector<double>( { 9.0, 9.0, 9.0 } ), nullDriver );

            THEN( "nothing happens" )
            {
                REQUIRE( prop["e0"].getValue() == "0" );
            }
        }

        WHEN( "no values or state change" )
        {
            fakeIndiDriver driver;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 1.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "only the state changes" )
        {
            fakeIndiDriver driver;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 1.0, 2.0 } ), &driver, pcf::IndiProperty::Busy );

            THEN( "an update is sent" )
            {
                REQUIRE( driver.sendCount == 1 );
            }
        }

        WHEN( "one of the values changes" )
        {
            fakeIndiDriver driver;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "an update is sent and all values are set" )
            {
                REQUIRE( driver.sendCount == 1 );
                REQUIRE( prop["e1"].getValue() == "42" );
            }
        }

        WHEN( "one of the element names does not exist" )
        {
            fakeIndiDriver             driver;
            std::vector<const char *> badEls( { "e0", "nonexistent", "e2" } );
            updatesIfChanged( prop, badEls, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught and no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the driver throws a std::exception on send" )
        {
            fakeIndiDriver driver;
            driver.throwOnSend = true;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the driver throws a non-standard exception on send" )
        {
            fakeIndiDriver driver;
            driver.throwOnSend = true;
            driver.throwNonStd = true;
            updatesIfChanged( prop, els, std::vector<double>( { 0.0, 42.0, 2.0 } ), &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

    }
}

/// Verify updateSwitchIfChanged() on a property with one switch element. A send must happen
/// only when the switch state or the property state changes. A missing element name must be
/// caught without a send.
SCENARIO( "updateSwitchIfChanged", "[indiUtils]" )
{
    GIVEN( "a property with one switch element" )
    {
        pcf::IndiProperty prop( pcf::IndiProperty::Switch );
        prop.setName( "prop" );
        prop.add( pcf::IndiElement( "sw" ) );
        prop["sw"].setSwitchState( pcf::IndiElement::Off );
        prop.setState( pcf::IndiProperty::Ok );

        WHEN( "the driver pointer is null" )
        {
            fakeIndiDriver *nullDriver = nullptr;
            updateSwitchIfChanged( prop, "sw", pcf::IndiElement::On, nullDriver );

            THEN( "nothing happens" )
            {
                REQUIRE( prop["sw"].getSwitchState() == pcf::IndiElement::Off );
            }
        }

        WHEN( "neither the switch state nor the property state changes" )
        {
            fakeIndiDriver driver;
            updateSwitchIfChanged( prop, "sw", pcf::IndiElement::Off, &driver, pcf::IndiProperty::Ok );

            THEN( "no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the switch value changes" )
        {
            fakeIndiDriver driver;
            updateSwitchIfChanged( prop, "sw", pcf::IndiElement::On, &driver, pcf::IndiProperty::Ok );

            THEN( "an update is sent and the switch is updated" )
            {
                REQUIRE( driver.sendCount == 1 );
                REQUIRE( prop["sw"].getSwitchState() == pcf::IndiElement::On );
            }
        }

        WHEN( "only the property state changes" )
        {
            fakeIndiDriver driver;
            updateSwitchIfChanged( prop, "sw", pcf::IndiElement::Off, &driver, pcf::IndiProperty::Busy );

            THEN( "an update is sent" )
            {
                REQUIRE( driver.sendCount == 1 );
            }
        }

        WHEN( "the element name does not exist" )
        {
            fakeIndiDriver driver;
            updateSwitchIfChanged( prop, "nonexistent", pcf::IndiElement::On, &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught and no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }
    }
}

/// Verify updateSelectionSwitchIfChanged() on a one-of-many switch property with three elements.
/// Selecting a different element must turn that element on and every other element off, and
/// send once. A missing element name logs an error and does not send.
SCENARIO( "updateSelectionSwitchIfChanged", "[indiUtils]" )
{
    GIVEN( "a one-of-many switch property with three elements" )
    {
        pcf::IndiProperty prop( pcf::IndiProperty::Switch );
        prop.setName( "prop" );
        prop.add( pcf::IndiElement( "el1" ) );
        prop.add( pcf::IndiElement( "el2" ) );
        prop.add( pcf::IndiElement( "el3" ) );
        prop["el1"].setSwitchState( pcf::IndiElement::On );
        prop["el2"].setSwitchState( pcf::IndiElement::Off );
        prop["el3"].setSwitchState( pcf::IndiElement::Off );
        prop.setState( pcf::IndiProperty::Ok );

        WHEN( "the driver pointer is null" )
        {
            fakeIndiDriver *nullDriver = nullptr;
            updateSelectionSwitchIfChanged( prop, "el2", nullDriver );

            THEN( "nothing happens" )
            {
                REQUIRE( prop["el1"].getSwitchState() == pcf::IndiElement::On );
            }
        }

        WHEN( "the requested element does not exist in the property" )
        {
            fakeIndiDriver driver;
            updateSelectionSwitchIfChanged( prop, "nonexistent", &driver );

            THEN( "an error is logged (to stderr) and no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "the requested element is already the only one on, and the state is unchanged" )
        {
            fakeIndiDriver driver;
            updateSelectionSwitchIfChanged( prop, "el1", &driver, pcf::IndiProperty::Ok );

            THEN( "no update is sent" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }

        WHEN( "a different element is selected" )
        {
            fakeIndiDriver driver;
            updateSelectionSwitchIfChanged( prop, "el2", &driver, pcf::IndiProperty::Ok );

            THEN( "an update is sent, and only the selected element is on" )
            {
                REQUIRE( driver.sendCount == 1 );
                REQUIRE( prop["el1"].getSwitchState() == pcf::IndiElement::Off );
                REQUIRE( prop["el2"].getSwitchState() == pcf::IndiElement::On );
                REQUIRE( prop["el3"].getSwitchState() == pcf::IndiElement::Off );
            }
        }

        WHEN( "only the property state changes" )
        {
            fakeIndiDriver driver;
            updateSelectionSwitchIfChanged( prop, "el1", &driver, pcf::IndiProperty::Busy );

            THEN( "an update is sent even though the selection did not change" )
            {
                REQUIRE( driver.sendCount == 1 );
            }
        }

        WHEN( "the driver throws while sending" )
        {
            fakeIndiDriver driver;
            driver.throwOnSend = true;
            updateSelectionSwitchIfChanged( prop, "el2", &driver, pcf::IndiProperty::Ok );

            THEN( "the exception is caught" )
            {
                REQUIRE( driver.sendCount == 0 );
            }
        }
    }
}

