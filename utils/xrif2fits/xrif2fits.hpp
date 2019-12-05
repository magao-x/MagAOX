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

   /** \name Image Data
     * @{
     */

   uint32_t m_width {0}; ///< The width of the image.
   uint32_t m_height {0}; ///< The height of the image.

   mx::improc::eigenCube<float> m_frames;

   uint8_t m_dataType; ///< The ImageStreamIO type code.

   size_t m_typeSize {0}; ///< The size of the type, in bytes.  Result of sizeof.


   IMAGE m_imageStream; ///< The ImageStreamIO shared memory buffer.


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

   long st = 0;
   long ed = m_files.size();
   int stp = 1;


   char header[XRIF_HEADER_SIZE];

   size_t nframes = 0;

   //First get number of frames.
   for(long n=st; n != ed; n += stp)
   {
      FILE * fp_xrif = fopen(m_files[n].c_str(), "rb");
      size_t nr = fread(header, 1, XRIF_HEADER_SIZE, fp_xrif);
      fclose(fp_xrif);
      if(nr != XRIF_HEADER_SIZE)
      {
         std::cerr << " (" << invokedName << "): Error reading header of " << m_files[n] << "\n";
         std::cerr << " read " << nr << "\n";
         return -1;
      }

      uint32_t header_size;
      xrif_read_header(m_xrif, &header_size , header);

      if(n==st)
      {
         m_width = m_xrif->width;
         m_height = m_xrif->height;
         m_dataType = m_xrif->type_code;
      }
      else
      {
         if(m_xrif->width != m_width)
         {
            std::cerr << " (" << invokedName << "): width mis-match in " << m_files[n] << "\n";
            return -1;
         }
         if(m_xrif->height != m_height)
         {
            std::cerr << " (" << invokedName << "): height mis-match in " << m_files[n] << "\n";
            return -1;
         }
         if(m_xrif->type_code != m_dataType)
         {
            std::cerr << " (" << invokedName << "): data type mismatch in " << m_files[n] << "\n";
         }
      }

      if(m_xrif->depth != 1)
      {
         std::cerr << " (" << invokedName << "): Cubes detected in " << m_files[n] << "\n";
         return -1;
      }

      /*if(m_dataType !=  XRIF_TYPECODE_INT16)
      {
         std::cerr << " (" << invokedName << "): Only 16-bit signed integers (short) supported" << "\n";
         return -1;
      }*/

      nframes += m_xrif->frames;

   }

   if(g_timeToDie != false)
   {
      std::cerr << " (" << invokedName << "): exiting.\n";
      return -1;
   }

   //Now record the actual number of frames
   //m_numFrames = nframes;

   std::cerr << " (" << invokedName << "): Reading " << nframes << " frames in " << (ed-st)*stp << " file";
   if( (ed-st)*stp > 1) std::cerr << "s";
   std::cerr << "\n";

   //Allocate the storage
   m_typeSize = xrif_typesize(m_dataType);

   m_frames.resize(m_width, m_height, nframes);

   
   int findex = 0;

   //Now de-compress and load the frames
   //Only decompressing the number of files needed, and only copying the number of frames needed
   for(long n=st; n < ed; ++n)
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
      fclose(fp_xrif);

      if(g_timeToDie == true) break; //check after the long read.

      if(nr != m_xrif->compressed_size)
      {
         std::cerr << " (" << invokedName << "): Error reading data from " << m_files[n] << "\n";
         return -1;
      }

      xrif_decode(m_xrif);

      if(g_timeToDie == true) break; //check after the decompress.

      mx::improc::eigenCube<unsigned short> tmpc( (unsigned short*) m_xrif->raw_buffer, m_xrif->width, m_xrif->height, m_xrif->frames);

      //Determine the order in which frames in tmpc are read
      long pst = 0;
      long ped = tmpc.planes();
      
      for( int p = pst; p != ped; ++p)
      {
         for(int ii=0;ii<m_frames.rows();++ii)
         {
            for(int jj=0;jj<m_frames.cols();++jj)
            {
               m_frames.image(findex)(ii,jj) = tmpc.image(p)(ii,jj);
            }
         }
         ++findex;
      }
   
      //De-allocate xrif
      xrif_delete(m_xrif);
      m_xrif = nullptr; //This is so destructor doesn't choke

      
      std::cerr << " (" << invokedName << "): Creating fits file: " << "  (" << m_width << " x " << m_height << " x " << nframes << ")\n";

      mx::improc::fitsFile<float> ff;
   
      std::string outname = m_files[n];
      size_t ext = outname.find(".xrif");
      outname.replace( ext, 5, ".fits");
   
      ff.write(outname, m_frames);
   }

   std::cerr << " (" << invokedName << "): exited normally.\n";

   return 0;
}

#endif //xrif2fits_hpp
