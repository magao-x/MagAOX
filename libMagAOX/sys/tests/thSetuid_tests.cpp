#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"


#include "../thSetuid.hpp"

#include <iostream>
#include <thread>


namespace thSetuid_test
{

bool timeToDie;

uid_t th_euidReal;
uid_t th_euidCalled;
uid_t th_suid;
   
void logThreadStart(  )
{
   while(timeToDie == false)
   {
      getresuid(&th_euidReal, &th_euidCalled, &th_suid);
   }
}


SCENARIO( "Setting per-thread setuid privileges", "[libMagAOX::sys]" ) 
{
   GIVEN("A process with setuid bit set")
   {
      uid_t euidReal, er;
      uid_t euidCalled, ec;
      uid_t suid;

      std::thread thrd;
   
      timeToDie = false;
   
      getresuid(&euidReal, &euidCalled, &suid);
   
      bool is_setuid = ((euidReal != euidCalled) && (euidReal != suid)); 
      REQUIRE(is_setuid == true); //Must run this test from setuid as unprivileged user
         
      MagAOX::sys::th_seteuid(euidReal);
   
      getresuid(&er, &ec, &suid);
   
      thrd = std::thread( logThreadStart );
   
      usleep(10000);
   
      REQUIRE( er == th_euidReal );  // Thread real uid not synched
      REQUIRE( ec == th_euidCalled ); // Thread called uid not synched
   
      MagAOX::sys::th_seteuid(euidCalled);
      
      getresuid(&er, &ec, &suid);
      usleep(10000);
      
      REQUIRE( er == th_euidReal ); //Thread real uid not synched
      REQUIRE( ec != th_euidCalled ); //Thread called uid synched
   
      timeToDie = true;
   
      thrd.join();
   }
}

} //namespace thSetuid_test
