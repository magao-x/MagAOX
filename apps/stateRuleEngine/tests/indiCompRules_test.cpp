#if(__cplusplus == 201703L)

#include "../../../tests/catch2/catch.hpp"

#include "../indiCompRules.hpp"


SCENARIO( "basic INDI Property Element-value rules", "[stateRuleEngine::rules]" ) 
{
    GIVEN("string comparison")
    {
        pcf::IndiProperty prop1(pcf::IndiProperty::Text);
        prop1.setDevice("ruleTest");
        prop1.setName("prop1");
        prop1.setPerm(pcf::IndiProperty::ReadWrite); 
        prop1.setState(pcf::IndiProperty::Idle);
        prop1.add(pcf::IndiElement("current"));
        prop1["current"] = "test";
        prop1.add(pcf::IndiElement("target"));
        prop1["target"] = "tset";

        txtValRule rule1;

        rule1.property(&prop1);
        rule1.element("current");
        
        WHEN("string should be equal and is")
        {
            rule1.comparison(ruleComparison::Eq);
            rule1.target("test");

            REQUIRE(rule1.value() == true);
        }

        WHEN("string should be equal and is not")
        {
            rule1.comparison(ruleComparison::Eq);
            rule1.target("tset");

            REQUIRE(rule1.value() == false);
        }

        WHEN("string should be not equal and is not equal")
        {
            rule1.comparison(ruleComparison::Neq);
            rule1.target("tset");

            REQUIRE(rule1.value() == true);
        }

        WHEN("string should be not equal and is equal")
        {
            rule1.comparison(ruleComparison::Neq);
            rule1.target("test");

            REQUIRE(rule1.value() == false);
        }
    }

    GIVEN("float comparison")
    {
        pcf::IndiProperty prop1(pcf::IndiProperty::Number);
        prop1.setDevice("ruleTest");
        prop1.setName("prop1");
        prop1.setPerm(pcf::IndiProperty::ReadWrite); 
        prop1.setState(pcf::IndiProperty::Idle);
        prop1.add(pcf::IndiElement("current"));
        prop1["current"].setValue(2.314159);
        prop1.add(pcf::IndiElement("target"));
        prop1["target"].setValue(1.567202);

        numValRule rule1;

        rule1.property(&prop1);
        rule1.element("current");

        WHEN("float should be equal and is")
        {    
            rule1.comparison(ruleComparison::Eq);
            rule1.target(2.314159);
            REQUIRE(rule1.value() == true);
        }

        WHEN("float should be equal and aren't")
        {
            rule1.comparison(ruleComparison::Eq);
            rule1.target(3.314159);
            REQUIRE(rule1.value() == false);
        }

        WHEN("float should be equal and aren't within tol")
        {
            rule1.comparison(ruleComparison::Eq);
            rule1.target(2.314158);

            REQUIRE(rule1.value() == false);
        }

        WHEN("float should be equal and are within tol")
        {
            rule1.comparison(ruleComparison::Eq);
            rule1.target(2.314158);
            rule1.tol(1e-4);

            REQUIRE(rule1.value() == true);
        }

        WHEN("float should be less than and is")
        {    
            rule1.comparison(ruleComparison::Lt);
            rule1.target(4.67892);
            REQUIRE(rule1.value() == true);
        }

        WHEN("float should be less than but is not")
        {    
            rule1.comparison(ruleComparison::Lt);
            rule1.target(1.67892);
            REQUIRE(rule1.value() == false);
        }

        WHEN("float should be greater than and is")
        {    
            rule1.comparison(ruleComparison::Gt);
            rule1.target(1.2);
            REQUIRE(rule1.value() == true);
        }

        WHEN("float should be greater than but is not")
        {    
            rule1.comparison(ruleComparison::Gt);
            rule1.target(7.9);
            REQUIRE(rule1.value() == false);
        }

        WHEN("float should be less-or-equal than and is less than")
        {    
            rule1.comparison(ruleComparison::LtEq);
            rule1.target(4.67892);
            REQUIRE(rule1.value() == true);
        }

        WHEN("float should be less-or-equal than and is equal")
        {    
            rule1.comparison(ruleComparison::LtEq);
            rule1.target(2.314159);
            REQUIRE(rule1.value() == true);
        }

        WHEN("float should be less-or-equal than but is not")
        {    
            rule1.comparison(ruleComparison::LtEq);
            rule1.target(0.789);
            REQUIRE(rule1.value() == false);
        }

        WHEN("float should be greater-or-equal than and is greater than")
        {    
            rule1.comparison(ruleComparison::GtEq);
            rule1.target(2.05);
            REQUIRE(rule1.value() == true);
        }

        WHEN("float should be greater-or-equal than and is equal")
        {    
            rule1.comparison(ruleComparison::GtEq);
            rule1.target(2.314159);
            REQUIRE(rule1.value() == true);
        }

        WHEN("float should be greater-or-equal than but is not")
        {    
            rule1.comparison(ruleComparison::GtEq);
            rule1.target(3.789);
            REQUIRE(rule1.value() == false);
        }
    }

    GIVEN("switch comparison")
    {
        pcf::IndiProperty prop1(pcf::IndiProperty::Switch);
        prop1.setDevice("ruleTest");
        prop1.setName("prop1");
        prop1.setPerm(pcf::IndiProperty::ReadWrite); 
        prop1.setState(pcf::IndiProperty::Idle);
        prop1.add(pcf::IndiElement("toggle"));
        
        swValRule rule1;

        rule1.property(&prop1);
        rule1.element("toggle");

        WHEN("switch is on and should be equal and is")
        {
            prop1["toggle"].setSwitchState(pcf::IndiElement::On);    
            rule1.target("On");
            rule1.comparison(ruleComparison::Eq);
            REQUIRE(rule1.value() == true);
        }

        WHEN("switch is on and should be equal but isn't")
        {
            prop1["toggle"].setSwitchState(pcf::IndiElement::On);    
            rule1.target("Off");
            rule1.comparison(ruleComparison::Eq);
            REQUIRE(rule1.value() == false);
        }

        WHEN("switch is on and should be not equal and is not")
        {
            prop1["toggle"].setSwitchState(pcf::IndiElement::On);    
            rule1.target("Off");
            rule1.comparison(ruleComparison::Neq);
            REQUIRE(rule1.value() == true);
        }

        WHEN("switch is on and should be not equal but is")
        {
            prop1["toggle"].setSwitchState(pcf::IndiElement::On);    
            rule1.target("On");
            rule1.comparison(ruleComparison::Neq);
            REQUIRE(rule1.value() == false);
        }

        WHEN("switch is off and should be equal and is")
        {
            prop1["toggle"].setSwitchState(pcf::IndiElement::Off);    
            rule1.target("Off");
            rule1.comparison(ruleComparison::Eq);
            REQUIRE(rule1.value() == true);
        }

        WHEN("switch is off and should be equal but isn't")
        {
            prop1["toggle"].setSwitchState(pcf::IndiElement::Off);    
            rule1.target("On");
            rule1.comparison(ruleComparison::Eq);
            REQUIRE(rule1.value() == false);
        }

        WHEN("switch is off and should be not equal and is not")
        {
            prop1["toggle"].setSwitchState(pcf::IndiElement::Off);    
            rule1.target("On");
            rule1.comparison(ruleComparison::Neq);
            REQUIRE(rule1.value() == true);
        }

        WHEN("switch is off and should be not equal but is")
        {
            prop1["toggle"].setSwitchState(pcf::IndiElement::Off);    
            rule1.target("Off");
            rule1.comparison(ruleComparison::Neq);
            REQUIRE(rule1.value() == false);
        }
    }
}

SCENARIO( "INDI element comparison", "[stateRuleEngine::rules]" ) 
{
    GIVEN("string comparison within same property")
    {
        pcf::IndiProperty prop1(pcf::IndiProperty::Text);
        prop1.setDevice("ruleTest");
        prop1.setName("prop1");
        prop1.setPerm(pcf::IndiProperty::ReadWrite); 
        prop1.setState(pcf::IndiProperty::Idle);
        prop1.add(pcf::IndiElement("current"));
        prop1.add(pcf::IndiElement("target"));

        elCompTxtRule rule1;
        rule1.property1(&prop1);
        rule1.property2(&prop1);
        rule1.element1("current");    
        rule1.element2("target");

        WHEN("string elements wihtin same property should be equal and are")
        {
            prop1["current"] = "test";
            prop1["target"] = "test";
            rule1.comparison(ruleComparison::Eq);

            REQUIRE(rule1.value() == true);
        }

        WHEN("string elements within same property should be equal and are not")
        {
            prop1["current"] = "test";
            prop1["target"] = "tset";
            rule1.comparison(ruleComparison::Eq);

            REQUIRE(rule1.value() == false);
        }

        WHEN("string elements within same property should not be equal and are not")
        {
            prop1["current"] = "test";
            prop1["target"] = "tset";
            rule1.comparison(ruleComparison::Neq);

            REQUIRE(rule1.value() == true);
        }

        WHEN("string elements within same property should not be equal and are")
        {
            prop1["current"] = "test";
            prop1["target"] = "test";
            rule1.comparison(ruleComparison::Neq);

            REQUIRE(rule1.value() == false);
        }
    }

    GIVEN("switch comparison")
    {
        pcf::IndiProperty prop1(pcf::IndiProperty::Switch);
        prop1.setDevice("ruleTest1");
        prop1.setName("prop1");
        prop1.setPerm(pcf::IndiProperty::ReadWrite); 
        prop1.setState(pcf::IndiProperty::Idle);
        prop1.add(pcf::IndiElement("nameTest"));
        
        pcf::IndiProperty prop2(pcf::IndiProperty::Switch);
        prop2.setDevice("ruleTest2");
        prop2.setName("prop2");
        prop2.setPerm(pcf::IndiProperty::ReadWrite); 
        prop2.setState(pcf::IndiProperty::Idle);
        prop2.add(pcf::IndiElement("badgeTest"));

        elCompSwRule rule1;
        rule1.property1(&prop1);
        rule1.property2(&prop2);
        rule1.element1("nameTest");    
        rule1.element2("badgeTest");

        WHEN("switches should be On and equal and are")
        {
            prop1["nameTest"].setSwitchState(pcf::IndiElement::On);
            prop2["badgeTest"].setSwitchState(pcf::IndiElement::On);
            rule1.comparison(ruleComparison::Eq);

            REQUIRE(rule1.value() == true);
        }

        WHEN("switches should be On and equal but are not")
        {
            prop1["nameTest"].setSwitchState(pcf::IndiElement::On);
            prop2["badgeTest"].setSwitchState(pcf::IndiElement::Off);
            rule1.comparison(ruleComparison::Eq);

            REQUIRE(rule1.value() == false);
        }

        WHEN("switches should be On and not equal and are not")
        {
            prop1["nameTest"].setSwitchState(pcf::IndiElement::On);
            prop2["badgeTest"].setSwitchState(pcf::IndiElement::Off);
            rule1.comparison(ruleComparison::Neq);

            REQUIRE(rule1.value() == true);
        }

        WHEN("switches should be On and not equal but are")
        {
            prop1["nameTest"].setSwitchState(pcf::IndiElement::On);
            prop2["badgeTest"].setSwitchState(pcf::IndiElement::On);
            rule1.comparison(ruleComparison::Neq);

            REQUIRE(rule1.value() == false);
        }

        WHEN("switches should be Off and equal and are")
        {
            prop1["nameTest"].setSwitchState(pcf::IndiElement::Off);
            prop2["badgeTest"].setSwitchState(pcf::IndiElement::Off);
            rule1.comparison(ruleComparison::Eq);

            REQUIRE(rule1.value() == true);
        }

        WHEN("switches should be Off and equal but are not")
        {
            prop1["nameTest"].setSwitchState(pcf::IndiElement::Off);
            prop2["badgeTest"].setSwitchState(pcf::IndiElement::On);
            rule1.comparison(ruleComparison::Eq);

            REQUIRE(rule1.value() == false);
        }

        WHEN("switches should be Off and not equal and are not")
        {
            prop1["nameTest"].setSwitchState(pcf::IndiElement::Off);
            prop2["badgeTest"].setSwitchState(pcf::IndiElement::On);
            rule1.comparison(ruleComparison::Neq);

            REQUIRE(rule1.value() == true);
        }

        WHEN("switches should be Off and not equal but are")
        {
            prop1["nameTest"].setSwitchState(pcf::IndiElement::Off);
            prop2["badgeTest"].setSwitchState(pcf::IndiElement::Off);
            rule1.comparison(ruleComparison::Neq);

            REQUIRE(rule1.value() == false);
        }
    }
    GIVEN("numeric comparison")
    {
        pcf::IndiProperty prop1(pcf::IndiProperty::Number);
        prop1.setDevice("ruleTest1");
        prop1.setName("prop1");
        prop1.setPerm(pcf::IndiProperty::ReadWrite); 
        prop1.setState(pcf::IndiProperty::Idle);
        prop1.add(pcf::IndiElement("nameTest"));
        
        pcf::IndiProperty prop2(pcf::IndiProperty::Number);
        prop2.setDevice("ruleTest2");
        prop2.setName("prop2");
        prop2.setPerm(pcf::IndiProperty::ReadWrite); 
        prop2.setState(pcf::IndiProperty::Idle);
        prop2.add(pcf::IndiElement("badgeTest"));

        elCompNumRule rule1;
        rule1.property1(&prop1);
        rule1.property2(&prop2);
        rule1.element1("nameTest");    
        rule1.element2("badgeTest");

        WHEN("numbers should be equal and are")
        {
            prop1["nameTest"].set(2.5);
            prop2["badgeTest"].set(2.5);
            rule1.comparison(ruleComparison::Eq);

            REQUIRE(rule1.value() == true);
        }

        WHEN("numbers should be equal but are not")
        {
            prop1["nameTest"].set(2.5);
            prop2["badgeTest"].set(2.6);
            rule1.comparison(ruleComparison::Eq);

            REQUIRE(rule1.value() == false);
        }
    }
}


SCENARIO( "basic rule comparisons", "[stateRuleEngine::rules]" ) 
{
    GIVEN("INDI Property rule comparison")
    {
        WHEN("two strings should be equal and are")
        {
            pcf::IndiProperty prop1(pcf::IndiProperty::Text);
            prop1.setDevice("ruleTest");
            prop1.setName("prop1");
            prop1.setPerm(pcf::IndiProperty::ReadWrite); 
            prop1.setState(pcf::IndiProperty::Idle);
            prop1.add(pcf::IndiElement("current"));
            prop1["current"] = "test";
            prop1.add(pcf::IndiElement("target"));
            prop1["target"] = "tset";

            txtValRule rule1;

            rule1.property(&prop1);
            rule1.element("current");
            rule1.comparison(ruleComparison::Eq);
            rule1.target("test");


            pcf::IndiProperty prop2(pcf::IndiProperty::Text);
            prop2.setDevice("ruleTest2");
            prop2.setName("prop2");
            prop2.setPerm(pcf::IndiProperty::ReadWrite); 
            prop2.setState(pcf::IndiProperty::Idle);
            prop2.add(pcf::IndiElement("current"));
            prop2["current"] = "fail";
            prop2.add(pcf::IndiElement("target"));
            prop2["target"] = "liaf";

            txtValRule rule2;

            rule2.property(&prop2);
            rule2.element("target");
            rule2.comparison(ruleComparison::Eq);
            rule2.target("liaf");


            ruleCompRule rule3;
            rule3.rule1(&rule1);
            rule3.rule2(&rule2);
            rule3.comparison(ruleComparison::And);

            REQUIRE(rule3.value() == true);
         
        }
    }
}      

SCENARIO( "compound rule compariaons", "[stateRuleEngine::rules]" ) 
{
    GIVEN("(A && B) || C")
    {
        pcf::IndiProperty prop1(pcf::IndiProperty::Text);
        prop1.setDevice("ruleTest");
        prop1.setName("prop1");
        prop1.setPerm(pcf::IndiProperty::ReadWrite); 
        prop1.setState(pcf::IndiProperty::Idle);
        prop1.add(pcf::IndiElement("current"));
        prop1["current"] = "test";
        prop1.add(pcf::IndiElement("target"));
        prop1["target"] = "tset";

        txtValRule rule1; //A

        rule1.property(&prop1);
        rule1.element("current");
        rule1.comparison(ruleComparison::Eq);
        
        pcf::IndiProperty prop2(pcf::IndiProperty::Text);
        prop2.setDevice("ruleTest2");
        prop2.setName("prop2");
        prop2.setPerm(pcf::IndiProperty::ReadWrite); 
        prop2.setState(pcf::IndiProperty::Idle);
        prop2.add(pcf::IndiElement("current"));
        prop2["current"] = "fail";
        prop2.add(pcf::IndiElement("target"));
        prop2["target"] = "liaf";

        txtValRule rule2; //B

        rule2.property(&prop2);
        rule2.element("target");
        rule2.comparison(ruleComparison::Eq);
        
        ruleCompRule rule3; //A&&B
        rule3.rule1(&rule1);
        rule3.rule2(&rule2);
        rule3.comparison(ruleComparison::And);

        pcf::IndiProperty prop3(pcf::IndiProperty::Text);
        prop3.setDevice("ruleTest3");
        prop3.setName("prop3");
        prop3.setPerm(pcf::IndiProperty::ReadWrite); 
        prop3.setState(pcf::IndiProperty::Idle);
        prop3.add(pcf::IndiElement("current"));
        prop3["current"] = "pass";
        prop3.add(pcf::IndiElement("target"));
        prop3["target"] = "ssap";

        txtValRule rule4; //C

        rule4.property(&prop3);
        rule4.element("current");
        rule4.comparison(ruleComparison::Eq);
        
        ruleCompRule rule5; // (A&&B) || C
        rule5.rule1(&rule3); // A&&B
        rule5.rule2(&rule4); // C
        rule5.comparison(ruleComparison::Or);

        WHEN("A==1, B==0, C==1")
        {
            rule1.target("test"); //A==1
            rule2.target("fail"); //B==0
            rule4.target("pass"); //C==1

            //(A && B) || C
            REQUIRE(rule5.value() == true);
         
        }

        WHEN("A==0, B==0, C==1")
        {
            rule1.target("tset"); //A==0
            rule2.target("fail"); //B==0
            rule4.target("pass"); //C==1

            //(A && B) || C
            REQUIRE(rule5.value() == true);
        }

        WHEN("A==1, B==0, C==0")
        {
            rule1.target("test"); //A==1
            rule2.target("fail"); //B==0
            rule4.target("ssap"); //C==0

            //(A && B) || C
            REQUIRE(rule5.value() == false);
        }

        WHEN("A==1, B==1, C==0")
        {
            rule1.target("test"); //A==1
            rule2.target("liaf"); //B==1
            rule4.target("ssap"); //C==0

            //(A && B) || C
            REQUIRE(rule5.value() == true);
        }
    }
}      

#endif
