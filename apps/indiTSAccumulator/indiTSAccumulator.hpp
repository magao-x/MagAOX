/** \file indiTSAccumulator.hpp
  * \brief The MagAO-X indiTSAccumulator header file
  *
  * \ingroup indiTSAccumulator_files
  */

#ifndef indiTSAccumulator_hpp
#define indiTSAccumulator_hpp

#include <map>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup indiTSAccumulator
  * \brief The indiTSAccumulator application to do YYYYYYY
  *
  * <a href="../handbook/operating/software/apps/indiTSAccumulator.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup indiTSAccumulator_files
  * \ingroup indiTSAccumulator
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X indiTSAccumulator
/** An application to accumulate a time-series from an INDI element. 
  * 
  * \ingroup indiTSAccumulator
  */
class indiTSAccumulator : public MagAOXApp<true>
{

   //Give the test harness access.
   friend class indiTSAccumulator_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   int m_maxEntries {36000};
   
   ///@}

   struct element
   {
      std::string m_name;
      
      timespec m_lastUpdate {0,0};

      IMAGE * m_imageStream {nullptr};

      explicit element(const std::string & el) : m_name{el}
      {}
   };

   struct property
   {
      pcf::IndiProperty m_property;
      std::vector<element> m_elements;
   };

   std::map<std::string, property> m_properties;


   static int st_setCallBack_all( void * app, const pcf::IndiProperty &ipRecv)
   {
      return static_cast<indiTSAccumulator *>(app)->setCallBack_all(ipRecv);
   }

   

public:
   /// Default c'tor.
   indiTSAccumulator();

   /// D'tor, declared and defined for noexcept.
   ~indiTSAccumulator() noexcept
   {}

   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   virtual void loadConfig();

   /// Startup function
   /**
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for indiTSAccumulator.
   /** 
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Shutdown the app.
   /** 
     *
     */
   virtual int appShutdown();

pcf::IndiProperty prop;
   int setCallBack_all( const pcf::IndiProperty &ipRecv );

};

indiTSAccumulator::indiTSAccumulator() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void indiTSAccumulator::setupConfig()
{
   config.add("elements", "", "elements", argType::Required, "", "elements", false, "vector<string>", "Comma separated List of elements specified as device.property.element.");
}

int indiTSAccumulator::loadConfigImpl( mx::app::appConfigurator & _config )
{
   std::vector<std::string> elements;
   _config(elements, "elements");
   
   if(elements.size() < 1)
   {
      log<text_log>("no elements specified", logPrio::LOG_CRITICAL);
      m_shutdown = true;
      return -1;
   }

   for(size_t n = 0; n < elements.size(); ++n)
   {
      size_t p1 = elements[n].find('.');
      if(p1 == std::string::npos)
      {
         log<text_log>("error parsing " + elements[n] + ": no . found", logPrio::LOG_CRITICAL);
         m_shutdown = true;
         return -1;
      }
      size_t p2 = elements[n].find('.', p1+1);
      if(p2 == std::string::npos)
      {
         log<text_log>("error parsing " + elements[n] + ": second . not found", logPrio::LOG_CRITICAL);
         m_shutdown = true;
         return -1;
      }

      try
      {
         std::string devName = elements[n].substr(0,p1);
         std::string propName = elements[n].substr(p1+1, p2-p1-1);
         std::string elName = elements[n].substr(p2+1);
         std::string key = devName + "." + propName;

         m_properties[key].m_property.setDevice(devName);
         m_properties[key].m_property.setName(propName);
         m_properties[key].m_elements.push_back(element(elName));
      }
      catch(const std::exception& e)
      {
         log<software_critical>({__FILE__, __LINE__, std::string("Exception caught: ") + e.what() + " [on element " + std::to_string(n) + "]"});
         m_shutdown = true;
         return -1;
      }
   }

   return 0;
}

void indiTSAccumulator::loadConfig()
{
   loadConfigImpl(config);
}

int indiTSAccumulator::appStartup()
{
   for(auto it = m_properties.begin(); it != m_properties.end(); ++it)
   {
      //Have to make these pass-by-const-referencable
      std::string devName = it->second.m_property.getDevice();
      std::string propName = it->second.m_property.getName();
      if( registerIndiPropertySet( it->second.m_property, devName, propName, st_setCallBack_all) < 0 )
      {
         log<software_critical>({__FILE__, __LINE__, "Error inserting property: " + devName + "." + propName});
         return -1;
      }

      for(size_t n=0; n < it->second.m_elements.size(); ++n)
      {
         it->second.m_elements[n].m_imageStream = (IMAGE *) malloc(sizeof(IMAGE));
         
         uint32_t imsize[3] = {0,0,0};
         imsize[0] = 1; 
         imsize[1] = 1;
         imsize[2] = m_maxEntries;
         std::string shmimName = devName + "." + propName + "." + it->second.m_elements[n].m_name;
      
         std::cerr << "Creating: " << shmimName << " " << imsize[0] << " " << imsize[1] << " " << imsize[2] << "\n";
      
         ImageStreamIO_createIm_gpu(it->second.m_elements[n].m_imageStream, shmimName.c_str(), 3, imsize, IMAGESTRUCT_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);

         it->second.m_elements[n].m_imageStream->md->cnt1 = it->second.m_elements[n].m_imageStream->md->size[2] - 1;

         it->second.m_elements[n].m_imageStream->md->atime = {0,0};
         it->second.m_elements[n].m_imageStream->md->writetime = {0,0};

         for(size_t m = 0; m < it->second.m_elements[n].m_imageStream->md->size[2]; ++m )
         {
            it->second.m_elements[n].m_imageStream->cntarray[m] = std::numeric_limits<uint64_t>::max();
            it->second.m_elements[n].m_imageStream->atimearray[m] = {0,0};
            it->second.m_elements[n].m_imageStream->writetimearray[m] = {0,0};
         }
      }
   }

   return 0;
}

int indiTSAccumulator::appLogic()
{
   return 0;
}

int indiTSAccumulator::appShutdown()
{
   return 0;
}

int indiTSAccumulator::setCallBack_all( const pcf::IndiProperty &ipRecv )
{
   std::string key = ipRecv.createUniqueKey();
   if(m_properties.count(key) > 0)
   {
      if(ipRecv.getType() != pcf::IndiProperty::Number) 
      {
         log<text_log>(key + " is not a Number property.  Can't time-series this.", logPrio::LOG_WARNING);
         return -1; //only numbers are supported for now.
      }

      //Now look for this property's elements
      for(size_t n=0; n < m_properties[key].m_elements.size(); ++n)
      {
         if(!ipRecv.find( m_properties[key].m_elements[n].m_name)) continue;

         IMAGE * image = m_properties[key].m_elements[n].m_imageStream;
         
         if(image == nullptr) 
         {
            log<software_error>({__FILE__, __LINE__, "Image for " + key + "." + m_properties[key].m_elements[n].m_name + " is nullptr"});
            continue;
         }

         timespec ts;
         ts.tv_sec = ipRecv.getTimeStamp().getTimeValSecs();
         ts.tv_nsec = ipRecv.getTimeStamp().getTimeValMicros()*1000;

         if(ts.tv_sec != image->md->atime.tv_sec || ts.tv_nsec != image->md->atime.tv_nsec)
         {
            //Get the value as a float
            float val = ipRecv[ m_properties[key].m_elements[n].m_name].get<float>();

            //Get cnt1 and rollover if needed
            uint64_t cnt1 = image->md->cnt1 + 1;
            if(cnt1 >= image->md->size[2]) cnt1 = 0;

            //Set the writing flag
            image->md->write=1;

            //Set the times            
            clock_gettime(CLOCK_REALTIME, &image->md->writetime);
            image->writetimearray[cnt1] = image->md->writetime;

            image->md->atime = ts;
            image->atimearray[cnt1] = ts;
            
            //Set the value
            image->array.F[cnt1] = val;
            
            //Now update counters
            image->md->cnt0++;
            image->cntarray[cnt1] = image->md->cnt0;
            image->md->cnt1 = cnt1;

            //And post
            image->md->write=0;
            ImageStreamIO_sempost(image,-1);
            std::cerr << ipRecv.createUniqueKey() << " " << ts.tv_sec << " " << ts.tv_nsec << "\n";
         }
      }
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //indiTSAccumulator_hpp
