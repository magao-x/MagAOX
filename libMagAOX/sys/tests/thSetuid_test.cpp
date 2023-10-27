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
   uid_t euidReal, er;
   uid_t euidCalled, ec;
   uid_t suid;

   getresuid(&euidReal, &euidCalled, &suid);

   if(euidReal == suid) {
      std::cerr << "Can't test setuid as root, moving on\n";
      SUCCEED("Can't test setuid as root, moving on");
   } else {
   GIVEN("A process with setuid bit set")
   {
      std::thread thrd;
   
      timeToDie = false;

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
}

} //namespace thSetuid_test
