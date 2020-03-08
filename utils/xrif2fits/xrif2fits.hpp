/** \file xrif2fits.hpp
  * \brief The xrif2fits class declaration and definition.
  *
  * \ingroup xrif2fits_files
  */

#ifndef xrif2fits_hpp
#define xrif2fits_hpp

#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include <xrif/xrif.h>




#include <mx/ioutils/fileUtils.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/improc/fitsFile.hpp>

#include <mx/timeUtils.hpp>
using namespace mx::tscomp;
using namespace mx::tsop;


#include "../../libMagAOX/libMagAOX.hpp"



                      
/** \defgroup xrif2fits xrif2fits: xrif-archive to FITS cube converter
  * \brief Read images from an xrif archive and write to FITS
  *
  * <a href="../handbook/utils/xrif2fits.html">Utility Documentation</a>
  *
  * \ingroup utils
  *
  */

/** \defgroup xrif2fits_files xrif2fits Files
  * \ingroup xrif2fits
  */

bool g_timeToDie = false;

void sigTermHandler( int signum,
                     siginfo_t *siginf,
                     void *ucont
                    )
{
   //Suppress those warnings . . .
   static_cast<void>(signum);
   static_cast<void>(siginf);
   static_cast<void>(ucont);

   std::cerr << "\n"; //clear out the ^C char

   g_timeToDie = true;
}

/// A utility to stream MagaO-X images from xrif compressed archives to an ImageStreamIO stream.
/**
  * \todo finish md doc for xrif2fits
  *
  * \ingroup xrif2fits
  */
class xrif2fits : public mx::app::application
{
protected:
   /** \name Configurable Parameters
     * @{
     */

   std::string m_dir; ///< The directory to search for files.  Can be empty if full path given in files.  If files is empty, all archives in dir will be used.

   std::vector<std::string> m_files; ///< List of files to use.  If dir is not empty, it will be pre-pended to each name.

   std::string m_logDir = "/home/jrmales/Data/MagAO-X/Tel/2019B/logs/";
   
   std::string m_telDir = "/home/jrmales/Data/MagAO-X/Tel/2019B/telem/";
   
   std::string m_outDir = "fits/";   
   
   bool m_noMeta {false};
   
   bool m_metaOnly {false};
   
protected:
   ///@}


   xrif_t m_xrif {nullptr};
   xrif_t m_xrif_timing {nullptr};


public:

   ~xrif2fits();

   virtual void setupConfig();

   virtual void loadConfig();

   virtual int execute();
};

inline
xrif2fits::~xrif2fits()
{
   if(m_xrif)
   {
      xrif_delete(m_xrif);
   }
   
   if(m_xrif_timing)
   {
      xrif_delete(m_xrif_timing);
   }
}

inline
void xrif2fits::setupConfig()
{
   config.add("dir","d", "dir" , argType::Required, "", "dir", false,  "string", "The directory to search for files.  Can be empty if full path given in files.");
   config.add("files","f", "files" , argType::Required, "", "files", false,  "vector<string>", "List of files to use.  If dir is not empty, it will be pre-pended to each name.");
   
   config.add("outDir","D", "outDir" , argType::Required, "", "outDir", false,  "string", "The directory in which to write output files.  Default is ./fits/.");
   
   config.add("metaOnly","", "metaOnly" , argType::True, "", "metaOnly", false,  "bool", "If true, output only meta data, without decoding images.  Default is false.");
   
   config.add("noMeta","", "noMeta" , argType::True, "", "noMeta", false,  "bool", "If true, the meta data file is not written (FITS headers will still be).  Default is false.");
   
}

inline
void xrif2fits::loadConfig()
{
   config(m_dir, "dir");
   config(m_files, "files");
   config(m_outDir, "outDir");
   
   config(m_metaOnly, "metaOnly");
   config(m_noMeta, "noMeta");
}

inline
int xrif2fits::execute()
{
   //Install signal handling
   struct sigaction act;
   sigset_t set;

   act.sa_sigaction = sigTermHandler;
   act.sa_flags = SA_SIGINFO;
   sigemptyset(&set);
   act.sa_mask = set;

   errno = 0;
   if( sigaction(SIGTERM, &act, 0) < 0 )
   {
      std::cerr << " (" << invokedName << "): error setting SIGTERM handler: " << strerror(errno) << "\n";
      return -1;
   }

   errno = 0;
   if( sigaction(SIGQUIT, &act, 0) < 0 )
   {
      std::cerr << " (" << invokedName << "): error setting SIGQUIT handler: " << strerror(errno) << "\n";
      return -1;
   }

   errno = 0;
   if( sigaction(SIGINT, &act, 0) < 0 )
   {
      std::cerr << " (" << invokedName << "): error setting SIGINT handler: " << strerror(errno) << "\n";
      return -1;
   }

   //Figure out which files to use
   if(m_files.size() == 0)
   {
      if(m_dir == "")
      {
         m_dir = "./";
      }

      m_files =  mx::ioutils::getFileNames( m_dir, "", "", ".xrif");
   }
   else
   {
      if(m_dir != "")
      {
         if(m_dir[m_dir.size()-1] != '/') m_dir += '/';
      }

      for(size_t n=0; n<m_files.size(); ++n)
      {
         m_files[n] = m_dir + m_files[n];
      }
   }

   if(m_files.size() == 0)
   {
      std::cerr << " (" << invokedName << "): No files found.\n";
      return -1;
   }

   if(m_outDir == "")
   {
      m_outDir = "./";
   }
   else
   {
      mkdir(m_outDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   }
   
   xrif_error_t rv;
   rv = xrif_new(&m_xrif);

   if(rv < 0)
   {
      std::cerr << " (" << invokedName << "): Error allocating xrif.\n";
      return -1;
   }

   rv = xrif_new(&m_xrif_timing);

   if(rv < 0)
   {
      std::cerr << " (" << invokedName << "): Error allocating xrif_timing.\n";
      return -1;
   }
   
   char header[XRIF_HEADER_SIZE];

   
   std::vector<logMeta> logMetas;
   logMetas.push_back({"CATOBJ", "catalog name of object", "tcsi", telem_telcat::eventCode, "catObj", "", 0, 0});
   logMetas.push_back({"PARANG", "parallactic angle at time of observation", "tcsi", telem_teldata::eventCode, "pa", "%0.4f", 1, 1});
   logMetas.push_back({"SCIBS", "science beamsplitter name", "stagescibs", telem_stage::eventCode, "presetName", "", 0, 0});
   logMetas.push_back({"SCIBSPOS", "preset index of science beamsplitter", "stagescibs", telem_stage::eventCode, "preset", "%0.2f", 1, 1});
   logMetas.push_back({"FWS1FILT", "filter name of fwsci1", "fwsci1", telem_stage::eventCode, "presetName", "", 0, 0});
   logMetas.push_back({"FWS1FPOS", "filter position of fwsci1", "fwsci1", telem_stage::eventCode, "preset", "%0.2f",1, 1});
   logMetas.push_back({"FWS2FILT", "filter name of fwsci2", "fwsci2", telem_stage::eventCode, "presetName", "", 0, 0});
   logMetas.push_back({"FWS2FPOS", "filter position of fwsci2", "fwsci2", telem_stage::eventCode, "preset", "%0.2f", 1, 1});
   
         
   logMap logs;
   logMap tels;
   
   std::cerr << "loading log file names . . .\n";
   logs.loadAppToFileMap( m_logDir, ".binlog");
   std::cerr << "loading telemetry file names . . .\n";
   tels.loadAppToFileMap( m_telDir, ".bintel");
   
   std::ofstream metaOut;
   //Print the meta-file header
   if(!m_noMeta)
   {
      metaOut.open(m_outDir + "meta_data.txt");
      metaOut << "#DATE-OBS FRAMENO ACQSEC ACQNSEC WRTSEC WRTNSEC";
      metaOut << " EXPTIME";
      for(size_t u=0;u<logMetas.size();++u)
      {
         metaOut << " " << logMetas[u].keyword() ;
      }
      metaOut << "\n";
   }
      
   //Now de-compress and load the frames
   //Only decompressing the number of files needed, and only copying the number of frames needed
   for(size_t n=0; n < m_files.size(); ++n)
   {
      if(g_timeToDie == true) break; //check before going on

      logFileName lfn(m_files[n]);
      
      tels.loadFiles(lfn.appName(), lfn.timestamp());
      
      logMeta exptimeMeta({"EXPTIME", "exposure time in seconds", lfn.appName(), telem_stdcam::eventCode, "exptime", "%f", 0, 1});
         
      std::cout << "******************************************************\n";
      std::cout << "* xrif2fits: decoding for " << lfn.appName() << "\n";
      std::cout << "******************************************************\n";
      
      FILE * fp_xrif = fopen(m_files[n].c_str(), "rb");
      size_t nr = fread(header, 1, XRIF_HEADER_SIZE, fp_xrif);
      if(nr != XRIF_HEADER_SIZE)
      {
         std::cerr << " (" << invokedName << "): Error reading header of " << m_files[n] << "\n";
         fclose(fp_xrif);
         return -1;
      }

      uint32_t header_size;
      xrif_read_header(m_xrif, &header_size , header);

      std::cout << "xrif compression details:\n";
      std::cout << "  difference method:  " << xrif_difference_method_string(m_xrif->difference_method) << '\n';
      std::cout << "  reorder method:     " << xrif_reorder_method_string(m_xrif->reorder_method) << '\n';
      std::cout << "  compression method: " << xrif_compress_method_string( m_xrif->compress_method) << '\n';
      if(m_xrif->compress_method == XRIF_COMPRESS_LZ4)
      {
         std::cout << "    LZ4 acceleration: " << m_xrif->lz4_acceleration << '\n';
      }
      
      std::cout << "  raw size:           " << m_xrif->width*m_xrif->height*m_xrif->depth*m_xrif->frames*m_xrif->data_size << " bytes\n";
      std::cout << "  encoded size:       " << m_xrif->compressed_size << " bytes\n";
      std::cout << "  ratio:              " << ((double)m_xrif->compressed_size) / (m_xrif->width*m_xrif->height*m_xrif->depth*m_xrif->frames*m_xrif->data_size) << '\n';

         
      xrif_allocate_raw(m_xrif);
      xrif_allocate_reordered(m_xrif);

      nr = fread(m_xrif->raw_buffer, 1, m_xrif->compressed_size, fp_xrif);
      
      if(nr != m_xrif->compressed_size)
      {
         std::cerr << " (" << invokedName << "): Error reading data from " << m_files[n] << "\n";
         return -1;
      }

      
      //Now get timing data
      nr = fread(header, 1, XRIF_HEADER_SIZE, fp_xrif);
      if(nr != XRIF_HEADER_SIZE)
      {
         std::cerr << " (" << invokedName << "): Error reading timing header of " << m_files[n] << "\n";
         fclose(fp_xrif);
         return -1;
      }
      
      xrif_read_header(m_xrif_timing, &header_size , header);
      xrif_allocate_raw(m_xrif_timing);
      xrif_allocate_reordered(m_xrif_timing);
      nr = fread(m_xrif_timing->raw_buffer, 1, m_xrif_timing->compressed_size, fp_xrif);
    
      if(nr != m_xrif_timing->compressed_size)
      {
         std::cerr << " (" << invokedName << "): Error reading timing data from " << m_files[n] << "\n";
         return -1;
      }
      
      fclose(fp_xrif);

      if(g_timeToDie == true) break; //check after the long read.

      if(!m_metaOnly)
      {
         xrif_decode(m_xrif);
      }
      
      xrif_decode(m_xrif_timing);
      
      if(g_timeToDie == true) break; //check after the decompress.

      mx::improc::eigenCube<unsigned short> tmpc( (unsigned short*) m_xrif->raw_buffer, m_xrif->width, m_xrif->height, m_xrif->frames);

      mx::improc::fitsFile<unsigned short> ff;
      mx::improc::fitsHeader fh;
      

      
      for( int q=0; q < tmpc.planes(); ++q)
      {
          uint64_t cnt0;
         timespec atime; //This is the acquisition time of the exposure
         timespec wtime;
         timespec stime = {0,0}; //This is the start time of the exposure, calculated as atime-exptime.
      
         uint64_t * curr_timing = (uint64_t*) m_xrif_timing->raw_buffer + 5*q;
         
         cnt0 = curr_timing[0];
         atime.tv_sec = curr_timing[1];
         atime.tv_nsec = curr_timing[2]; 
         wtime.tv_sec = curr_timing[3];
         wtime.tv_nsec = curr_timing[4];

         //We have to bootstrap the exposure time
         char * prior = nullptr;
         tels.getPriorLog(prior, lfn.appName(), eventCodes::TELEM_STDCAM, atime);
         double exptime = -1;
         if(prior)
         {
            char * priorprior = nullptr;
            exptime = telem_stdcam::exptime(logHeader::messageBuffer(prior));
         
            stime = atime-exptime;
            tels.getPriorLog(priorprior, lfn.appName(), eventCodes::TELEM_STDCAM, stime);
      
            if(telem_stdcam::exptime(logHeader::messageBuffer(priorprior)) != exptime) ///\todo this needs to check for any log entries between end and start
            {
               std::cerr << "Change in exposure time mid-exposure\n";
            }
         }
         //std::cerr << "Exptime: " << telem_stdcam::exptime(logHeader::messageBuffer(priorprior)) << "\n";
         
         //timespecX midexp = mx::meanTimespec( atime, stime);
         
         
         std::string timestamp;
         mx::timeStamp(timestamp, atime);
         std::string outfname = m_outDir + lfn.appName() + "_" + timestamp + ".fits";

         fh.clear();
         
         std::string dateobs = mx::ISO8601DateTimeStr(atime, 1);
         
         fh.append("DATE-OBS", dateobs, "Date of obs. YYYY-mm-ddTHH:MM:SS");
         fh.append("FRAMENO", cnt0);
         fh.append("ACQSEC", atime.tv_sec);
         fh.append("ACQNSEC", atime.tv_nsec);
         fh.append("WRTSEC", wtime.tv_sec);
         fh.append("WRTNSEC", wtime.tv_nsec);

         if(!m_noMeta)
         {
            metaOut << dateobs << " " << cnt0 << " " << atime.tv_sec << " " << atime.tv_nsec << " " << wtime.tv_sec << " " << wtime.tv_nsec << " ";
         }
         
         if(exptime > -1)
         {
            fh.append(exptimeMeta.card(tels,stime,atime));
      
            if(!m_noMeta) metaOut << exptimeMeta.value(tels, stime, atime);
            for(size_t u=0;u<logMetas.size();++u)
            {
               fh.append(logMetas[u].card(tels, stime, atime));
               if(!m_noMeta) metaOut << " " << logMetas[u].value(tels, stime, atime) ;
            }
            
            if(!m_noMeta) metaOut << "\n";
            
         }

         if(!m_metaOnly)
         {
            mx::improc::eigenImage<unsigned short> im = tmpc.image(q);
            ff.write(outfname, tmpc.image(q), fh);
         }
      }
      
      metaOut.close();
      //Below is for cubes
      /*
      outname = m_files[n];
      ext = outname.find(".xrif");
      outname.replace( ext, 5, ".time");
      
      std::ofstream fout;
      fout.open(outname);
      fout << "#cnt0   atime-sec  atime-nsec wtime-sec  wtime-nsec\n";
      for(int i=0; i< tmpc.planes(); ++i)
      {
         uint64_t * curr_timing = (uint64_t*) m_xrif_timing->raw_buffer + 5*i;
         
         fout << curr_timing[0] << " " << curr_timing[1] << " " << curr_timing[2] << "  " << curr_timing[3] << " " << curr_timing[4] << "\n";
      }*/
   }

   std::cerr << " (" << invokedName << "): exited normally.\n";

   return 0;
}

#endif //xrif2fits_hpp
