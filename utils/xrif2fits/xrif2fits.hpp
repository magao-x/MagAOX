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
#include <mx/improc/fitsFile.hpp>

#include <mx/timeUtils.hpp>

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

   std::string m_dir; ///< The directory to search for files.  Can be empty if pull path given in files.  If files is empty, all archives in dir will be used.

   std::vector<std::string> m_files; ///< List of files to use.  If dir is not empty, it will be pre-pended to each name.

   ///@}


   xrif_t m_xrif {nullptr};
   xrif_t m_xrif_timing {nullptr};

   /** \name Image Data
     * @{
     */

   //uint32_t m_width {0}; ///< The width of the image.
   //uint32_t m_height {0}; ///< The height of the image.

   //mx::improc::eigenCube<unsigned short> m_frames;

   //uint8_t m_dataType; ///< The ImageStreamIO type code.

   //size_t m_typeSize {0}; ///< The size of the type, in bytes.  Result of sizeof.


   

   ///@}

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
   config.add("dir","d", "dir" , argType::Required, "", "dir", false,  "string", "The directory to search for files.  Can be empty if pull path given in files.");
   config.add("files","f", "files" , argType::Required, "", "files", false,  "vector<string>", "List of files to use.  If dir is not empty, it will be pre-pended to each name.");
}

inline
void xrif2fits::loadConfig()
{
   config(m_dir, "dir");
   config(m_files, "files");
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

   
   //Now de-compress and load the frames
   //Only decompressing the number of files needed, and only copying the number of frames needed
   for(size_t n=0; n < m_files.size(); ++n)
   {
      if(g_timeToDie == true) break; //check before going on

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

     

      xrif_decode(m_xrif);

      xrif_decode(m_xrif_timing);
      
      if(g_timeToDie == true) break; //check after the decompress.

      mx::improc::eigenCube<unsigned short> tmpc( (unsigned short*) m_xrif->raw_buffer, m_xrif->width, m_xrif->height, m_xrif->frames);

      
      
      std::cerr << " (" << invokedName << "): Creating fits file: " << "  (" << m_xrif->width << " x " <<  m_xrif->height << " x " << m_xrif->frames << ")\n";

      mx::improc::fitsFile<unsigned short> ff;
   
      std::string outname = m_files[n];
      size_t ext = outname.find(".xrif");
      outname.replace( ext, 5, ".fits");
   
      ff.write(outname, tmpc);
      
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
      }
   }

   std::cerr << " (" << invokedName << "): exited normally.\n";

   return 0;
}

#endif //xrif2fits_hpp
