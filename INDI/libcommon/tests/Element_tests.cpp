/** \file Types_tests.cpp
 * \brief Catch2 tests for Types.hpp.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * History:
 */

#include <iostream>
#include <cstring>

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../Element.hpp"
#include "../Element.cpp"

/** Scenario: Construction of xindi::Element
 *
 * Verify construction
 *
 * \anchor tests_xindi_Element_construction
 */
SCENARIO( "Construction of xindi::Element", "[xindi::Element]" )
{
    GIVEN( "An Element to construct without type conversion" )
    {
        WHEN( "default construction" )
        {
            xindi::Element el;

            REQUIRE( el.type() == xindi::Type::Unknown);
            REQUIRE( el.hasValidType() == false);
            REQUIRE( el.name() == "" );
            REQUIRE( el.hasValidName() == false);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Number" )
        {
            xindi::Element el(xindi::Type::Number);

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "" );
            REQUIRE( el.hasValidName() == false);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Text" )
        {
            xindi::Element el(xindi::Type::Text);

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "" );
            REQUIRE( el.hasValidName() == false);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Switch" )
        {
            xindi::Element el(xindi::Type::Switch);

            REQUIRE( el.type() == xindi::Type::Switch);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "" );
            REQUIRE( el.hasValidName() == false);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Light" )
        {
            xindi::Element el(xindi::Type::Light);

            REQUIRE( el.type() == xindi::Type::Light);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "" );
            REQUIRE( el.hasValidName() == false);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as BLOB" )
        {
            xindi::Element el(xindi::Type::BLOB);

            REQUIRE( el.type() == xindi::Type::BLOB);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "" );
            REQUIRE( el.hasValidName() == false);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing with a name" )
        {
            xindi::Element el("newel");

            REQUIRE( el.type() == xindi::Type::Unknown);
            REQUIRE( el.hasValidType() == false);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Number with a name" )
        {
            xindi::Element el(xindi::Type::Number, "newel");

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Text with a name" )
        {
            xindi::Element el(xindi::Type::Text, "newel");

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Switch with a name" )
        {
            xindi::Element el(xindi::Type::Switch, "newel");

            REQUIRE( el.type() == xindi::Type::Switch);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Light with a name" )
        {
            xindi::Element el(xindi::Type::Light, "newel");

            REQUIRE( el.type() == xindi::Type::Light);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as BLOB with a name" )
        {
            xindi::Element el(xindi::Type::BLOB, "newel");

            REQUIRE( el.type() == xindi::Type::BLOB);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "");
            REQUIRE( el.hasValidValue() == false);
        }

        WHEN( "constructing as Number with a name and string value" )
        {
            xindi::Element el(xindi::Type::Number, "newel", "2");

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<int>() == 2);
            REQUIRE( el.hasValidValue() == true);
        }

        WHEN( "constructing as Text with a name and string value" )
        {
            xindi::Element el(xindi::Type::Text, "newel", "newval");

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == "newval");
            REQUIRE( el.hasValidValue() == true);
        }

        WHEN( "constructing as Switch with a name and string value" )
        {
            xindi::Element el(xindi::Type::Switch, "newel", "On");

            REQUIRE( el.type() == xindi::Type::Switch);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<xindi::Switch>() == xindi::Switch::On);
            REQUIRE( el.hasValidValue() == true);
        }

        WHEN( "constructing as Light with a name and string value" )
        {
            xindi::Element el(xindi::Type::Light, "newel", "Ok");

            REQUIRE( el.type() == xindi::Type::Light);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newel" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<xindi::Light>() == xindi::Light::Ok);
            REQUIRE( el.hasValidValue() == true);
        }
    }
    GIVEN( "An Element to construct with type conversion" )
    {
        WHEN( "constructing a Number as int" )
        {
            xindi::Element el("newval", static_cast<int>(2));

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<int>() == 2);
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Number as float" )
        {
            xindi::Element el("newval", static_cast<float>(3.14159));

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<float>() == static_cast<float>(3.14159));
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Text from string" )
        {
            xindi::Element el("newval", std::string("newel"));

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<std::string>() == std::string("newel"));
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Text from const char *" )
        {
            xindi::Element el("newval", "newel");

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == std::string("newel"));
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Text from char *" )
        {
            size_t sz = sizeof("newel");
            char * ne = new char[sz];
            strncpy(ne, "newel", sz);

            xindi::Element el("newval", ne);

            delete[] ne;

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value() == std::string("newel"));
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Light from an Idle Light" )
        {
            xindi::Element el("newval", xindi::Light::Idle);

            REQUIRE( el.type() == xindi::Type::Light);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<xindi::Light>() == xindi::Light::Idle);
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Light from an Ok Light" )
        {
            xindi::Element el("newval", xindi::Light::Ok);

            REQUIRE( el.type() == xindi::Type::Light);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<xindi::Light>() == xindi::Light::Ok);
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Light from a Busy Light" )
        {
            xindi::Element el("newval", xindi::Light::Busy);

            REQUIRE( el.type() == xindi::Type::Light);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<xindi::Light>() == xindi::Light::Busy);
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Light from an Alert Light" )
        {
            xindi::Element el("newval", xindi::Light::Alert);

            REQUIRE( el.type() == xindi::Type::Light);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<xindi::Light>() == xindi::Light::Alert);
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Switch from an On Switch" )
        {
            xindi::Element el("newval", xindi::Switch::On);

            REQUIRE( el.type() == xindi::Type::Switch);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<xindi::Switch>() == xindi::Switch::On);
            REQUIRE( el.hasValidValue() == true);
        }
        WHEN( "constructing a Switch from an Off Switch" )
        {
            xindi::Element el("newval", xindi::Switch::Off);

            REQUIRE( el.type() == xindi::Type::Switch);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.format() == "");
            REQUIRE( el.hasValidFormat() == false);
            REQUIRE( el.label() == "");
            REQUIRE( el.hasValidLabel() == false);
            REQUIRE( el.min() == "");
            REQUIRE( el.hasValidMin() == false);
            REQUIRE( el.max() == "");
            REQUIRE( el.hasValidMax() == false);
            REQUIRE( el.step() == "");
            REQUIRE( el.hasValidStep() == false);
            REQUIRE( el.size() == "");
            REQUIRE( el.hasValidSize() == false);
            REQUIRE( el.value<xindi::Switch>() == xindi::Switch::Off);
            REQUIRE( el.hasValidValue() == true);
        }
    }

}

//todo: setting parameters
//todo: setting value (get rid of separate setSwitchState, etc.)

/** Scenario: Assignment of values to xindi::Element
 *
 * Verify the assignment operators for values
 *
 * \anchor tests_xindi_Element_value_assignment
 */
SCENARIO( "Assignment of values to xindi::Element", "[xindi::Element]" )
{
    GIVEN( "An Element constructed with a value" )
    {
        WHEN( "Type is int" )
        {
            xindi::Element el("newval", static_cast<int>(2));

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<int>() == 2);
            REQUIRE( el.hasValidValue() == true);

            el = static_cast<int>(4);
            REQUIRE( el.value<int>() == 4);
        }
        WHEN( "Type is uint64_t" )
        {
            xindi::Element el("newval", static_cast<uint64_t>(18446744073709551615ul));

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<uint64_t>() == 18446744073709551615ul);
            REQUIRE( el.hasValidValue() == true);

            el = static_cast<uint64_t>(18446744073709551611ul);
            REQUIRE( el.value<uint64_t>() == 18446744073709551611ul);
        }
        WHEN( "Type is float" )
        {
            xindi::Element el("newval", static_cast<float>(1.01234567));

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<float>() == static_cast<float>(1.01234567));
            REQUIRE( el.hasValidValue() == true);

            el = static_cast<float>(1.01234564);
            REQUIRE( el.value<float>() == static_cast<float>(1.01234564));
            REQUIRE( el.value<float>() == static_cast<float>(1.01234567));
        }
        WHEN( "Type is double" )
        {
            xindi::Element el("newval", static_cast<double>(1.0123456789123456));

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<double>() == static_cast<double>(1.0123456789123456));
            REQUIRE( el.hasValidValue() == true);

            el = static_cast<double>(1.0123456789123457);
            REQUIRE( el.value<double>() == static_cast<double>(1.0123456789123457));
            REQUIRE( el.value<double>() != static_cast<double>(1.0123456789123456));
        }
        WHEN( "Type is complex float" )
        {
            xindi::Element el("newval", std::complex<float>(1.34,3.57));

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<std::complex<float>>() == std::complex<float>(1.34,3.57));
            REQUIRE( el.hasValidValue() == true);

            el = std::complex<float>(4.31,8.75);
            REQUIRE( el.value<std::complex<float>>() == std::complex<float>(4.31,8.75));
        }
        WHEN( "Type is complex double" )
        {
            xindi::Element el("newval", std::complex<double>(1.34,3.57));

            REQUIRE( el.type() == xindi::Type::Number);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<std::complex<double>>() == std::complex<double>(1.34,3.57));
            REQUIRE( el.hasValidValue() == true);

            el = std::complex<double>(4.31,8.75);
            REQUIRE( el.value<std::complex<double>>() == std::complex<double>(4.31,8.75));
        }
        WHEN( "Type is std::string" )
        {
            xindi::Element el("newval", std::string("newtext"));

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<std::string>() == std::string("newtext"));
            REQUIRE( el.hasValidValue() == true);

            el = std::string("bettertext");
            REQUIRE( el.value<std::string>() == std::string("bettertext"));
        }
        WHEN( "Type is const char *" )
        {
            xindi::Element el("newval", "newtext");

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<std::string>() == "newtext");
            REQUIRE( el.hasValidValue() == true);

            el = "bettertext";
            REQUIRE( el.value<std::string>() == "bettertext");
        }
        WHEN( "Type is char *" )
        {
            size_t sz = sizeof("newtext");
            char * ne = new char[sz];
            strncpy(ne, "newtext", sz);

            xindi::Element el("newval", ne);

            delete[] ne;

            REQUIRE( el.type() == xindi::Type::Text);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<std::string>() == "newtext");
            REQUIRE( el.hasValidValue() == true);

            sz = sizeof("bettertext");
            ne = new char[sz];
            strncpy(ne, "bettertext", sz);

            el = ne;
            delete[] ne;

            REQUIRE( el.value<std::string>() == "bettertext");
        }
        WHEN( "Type is Switch" )
        {
            xindi::Element el("newval", xindi::Switch::On);

            REQUIRE( el.type() == xindi::Type::Switch);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<xindi::Switch>() == xindi::Switch::On);
            REQUIRE( el.hasValidValue() == true);

            el = xindi::Switch::Off;

            REQUIRE( el.value<xindi::Switch>() == xindi::Switch::Off);
        }
        WHEN( "Type is Light" )
        {
            xindi::Element el("newval", xindi::Light::Ok);

            REQUIRE( el.type() == xindi::Type::Light);
            REQUIRE( el.hasValidType() == true);
            REQUIRE( el.name() == "newval" );
            REQUIRE( el.hasValidName() == true);
            REQUIRE( el.value<xindi::Light>() == xindi::Light::Ok);
            REQUIRE( el.hasValidValue() == true);

            el = xindi::Light::Idle;

            REQUIRE( el.value<xindi::Light>() == xindi::Light::Idle);
        }
    }
}

