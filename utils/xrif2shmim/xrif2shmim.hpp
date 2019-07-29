/** \file xrif2shmim.hpp
  * \brief The xrif2shmim class declaration and definition.
  */

#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include <xrif/xrif.h>

#include <mx/ioutils/fileUtils.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/fitsFile.hpp>

#include <mx/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"

/// Sleep for a specified period in microseconds.
inline
void microsleep( unsigned usec /**< [in] the number of microseconds to sleep. */)
{
   std::this_thread::sleep_for(std::chrono::microseconds(usec));
}


/// A utility to stream MagaO-X images from xrif compressed archives to an ImageStreamIO stream.
class xrif2shmim : public mx::app::application
{
protected:
   /** \name Configurable Parameters
     * @{
     */
   
   std::string m_dir; ///< The directory to search for files.  Can be empty if pull path given in files.  If files is empty, all archives in dir will be used. 
   
   std::vector<std::string> m_files; ///< List of files to use.  If dir is not empty, it will be pre-pended to each name.  
   
   size_t m_numFrames {0}; ///< The number of frames to store in memory.  This defines how many different images will be streamed.  If 0 (the default), all frames found using dir and files are loaded and stored.
   
   bool m_earliest {false}; ///< If true, then the earliest numFrames in the archive are used.  By default (if not set) the latest numFrames are used.
   
   std::string m_shmimName {"xrif2shmim"}; ///< The name of the shared memory buffer to stream to.  Default is "xrif2shmim".
   
   uint32_t m_circBuffLength {1}; ///< The length of the shared memory circular buffer. Default is 1.
   
   double m_fps {10}; ///< The rate, in frames per second, at which to stream images.  Default is 10 fps.
   
   ///@}
   
   
   xrif_t m_xrif {nullptr};
   
   /** \name Image Data
     * @{
     */
   
   uint32_t m_width {0}; ///< The width of the image.
   uint32_t m_height {0}; ///< The height of the image.
   
   mx::improc::eigenCube<short> m_frames;
   
   uint8_t m_dataType; ///< The ImageStreamIO type code.
   
   size_t m_typeSize {0}; ///< The size of the type, in bytes.  Result of sizeof.
   
   
   IMAGE m_imageStream; ///< The ImageStreamIO shared memory buffer.
   
   
   ///@}
   
public:
   
   ~xrif2shmim();
   
   virtual void setupConfig();

   virtual void loadConfig();

   virtual int execute();
};

inline 
xrif2shmim::~xrif2shmim()
{
   if(m_xrif)
   {
      xrif_delete(m_xrif);
   }
}

inline
void xrif2shmim::setupConfig()
{
   config.add("dir","d", "dir" , argType::Required, "", "dir", false,  "string", "The directory to search for files.  Can be empty if pull path given in files.");
   config.add("files","f", "files" , argType::Required, "", "files", false,  "vector<string>", "List of files to use.  If dir is not empty, it will be pre-pended to each name.");
   config.add("numFrames","N", "numFrames" , argType::Required, "", "numFrames", false,  "int", "The number of frames to store in memory.  This defines how many different images will be streamed.  If 0 (the default), all frames found using dir and files are loaded and stored.");
   config.add("earliest","e", "earliest" , argType::True, "", "earliest", false,  "bool", "If set or true, then the earliest numFrames in the archive are used.  By default (if not set) the latest numFrames are used.");
   config.add("shmimName","n", "shmimName" , argType::Required, "", "shmimName", false,  "string", "The name of the shared memory buffer to stream to.  Default is \"xrif2shmim\"");
   config.add("circBuffLength","L", "circBuffLength" , argType::Required, "", "circBuffLength", false,  "int", "The length of the shared memory circular buffer. Default is 1.");
   
   config.add("fps","F", "fps" , argType::Required, "", "fps", false,  "float", "The rate, in frames per second, at which to stream images. Default is 10 fps.");
}

inline
void xrif2shmim::loadConfig()
{
   config(m_dir, "dir");
   config(m_files, "files");
   config(m_numFrames, "numFrames");
   config(m_earliest, "earliest");
   config(m_shmimName, "shmimName");
   config(m_circBuffLength, "circBuffLength");
   config(m_fps, "fps");
}

inline 
int xrif2shmim::execute()
{
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
      std::cerr << "No files found.\n";
      return -1;
   }

   
   xrif_error_t rv;
   rv = xrif_new(&m_xrif); 
   
   if(rv < 0)
   {
      std::cerr << "Error allocating xrif.\n";
      return -1;
   }
      
   long st = 0;
   long ed = m_files.size();
   int stp = 1;
   
   if(m_numFrames != 0 && !m_earliest)
   {
      st = m_files.size()-1;
      ed = -1;
      stp = -1;
   }
   
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
         std::cerr << "Error reading header of " << m_files[n] << "\n";
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
            std::cerr << "width mis-match in " << m_files[n] << "\n";
            return -1;
         }
         if(m_xrif->height != m_height)
         {
            std::cerr << "height mis-match in " << m_files[n] << "\n";
            return -1;
         }
         if(m_xrif->type_code != m_dataType)
         {
            std::cerr << "data type mismatch in " << m_files[n] << "\n";
         }
      }
      
      if(m_xrif->depth != 1)
      {
         std::cerr << "Cubes detected in " << m_files[n] << "\n";
         return -1;
      }
      
      if(m_dataType !=  XRIF_TYPECODE_INT16)
      {
         std::cerr << "Only 16-bit signed integerss (short) supported" << "\n";
         return -1;
      }
      
      nframes += m_xrif->frames;
      
      if(nframes >= m_numFrames && m_numFrames > 0)
      {
         ed = n + stp;
         break;
      }
   }
   
   //Now record the actual number of frames
   if(m_numFrames == 0 || nframes < m_numFrames) m_numFrames = nframes;
   
   std::cout << "Reading " << m_numFrames << " frames in " << (ed-st)*stp << " file";
   if( (ed-st)*stp > 1) std::cout << "s";
   std::cout << "\n";
   
   //Allocate the storage
   m_typeSize = xrif_typesize(m_dataType);
   
   m_frames.resize(m_width, m_height, m_numFrames);   
   
   //Determine the order in which frames are copied 
   int findex = 0;
   int fed = m_frames.planes();
   if(stp == -1) 
   {
      findex = m_frames.planes()-1;
      fed = -1;
   }
   
   //Now de-compress and load the frames
   //Only decompressing the number of files needed, and only copying the number of frames needed
   for(long n=st; n != ed; n += stp)
   {
      FILE * fp_xrif = fopen(m_files[n].c_str(), "rb");
      size_t nr = fread(header, 1, XRIF_HEADER_SIZE, fp_xrif);
      if(nr != XRIF_HEADER_SIZE)
      {
         std::cerr << "Error reading header of " << m_files[n] << "\n";
         fclose(fp_xrif);
         return -1;
      }
      
      uint32_t header_size;
      xrif_read_header(m_xrif, &header_size , header);
      
      xrif_allocate_raw(m_xrif);
      xrif_allocate_reordered(m_xrif);

      nr = fread(m_xrif->raw_buffer, 1, m_xrif->compressed_size, fp_xrif);
      fclose(fp_xrif);
    
      if(nr != m_xrif->compressed_size)
      {
         std::cerr << "Error reading data from " << m_files[n] << "\n";
         return -1;
      }
   
      xrif_decode(m_xrif);
   
      mx::improc::eigenCube<short> tmpc( (short*) m_xrif->raw_buffer, m_xrif->width, m_xrif->height, m_xrif->frames);
      
      //Determine the order in which frames in tmpc are read
      long pst = 0;
      long ped = tmpc.planes();
      if(stp == -1)
      {
         pst = tmpc.planes()-1;
         ped = -1;
      }
      
      for( int p = pst; p != ped; p += stp)
      {
         m_frames.image(findex) = tmpc.image(p);
         findex += stp;
         if(findex == fed)
         {
            break;
         }
      }
   }

   //De-allocate xrif
   xrif_delete(m_xrif);
   m_xrif = nullptr; //This is so destructor doesn't choke
   

   //Now create share memory stream.
   
   uint32_t imsize[3];
   imsize[0] = m_width; 
   imsize[1] = m_height;
   imsize[2] = m_circBuffLength;
   
   std::cerr << "Creating stream: " << m_shmimName << "  (" << m_width << " x " << m_height << " x " << m_circBuffLength << ")\n";
      
   ImageStreamIO_createIm_gpu(&m_imageStream, m_shmimName.c_str(), 3, imsize, m_dataType, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL);
       
   m_imageStream.md->cnt1 = m_circBuffLength;
         
   
   
         
   //Begin streaming
   std::cout << m_fps << "\n";
   
   
   uint64_t next_cnt1 = 0; 
   char * next_dest = (char *) m_imageStream.array.raw;
   timespec * next_wtimearr = &m_imageStream.writetimearray[0];
   timespec * next_atimearr = &m_imageStream.atimearray[0];
   uint64_t * next_cntarr = &m_imageStream.cntarray[0];
      
   findex = 0;
   double lastSend = mx::get_curr_time();
   double delta = 0;
   while(1)
   {
      m_imageStream.md->write=1;
      
      memcpy(next_dest, m_frames.image(findex).data(), m_width*m_height*m_typeSize); //This is about 10 usec faster -- but we have to descramble.
      
      //Set the time of last write
      clock_gettime(CLOCK_REALTIME, &m_imageStream.md->writetime);
      m_imageStream.md->atime = m_imageStream.md->writetime;
   
      //Update cnt1
      m_imageStream.md->cnt1 = next_cnt1;
       
      //Update cnt0
      m_imageStream.md->cnt0++;
      
      *next_wtimearr = m_imageStream.md->writetime;
      *next_atimearr = m_imageStream.md->atime;
      *next_cntarr = m_imageStream.md->cnt0;
      
      //And post
      m_imageStream.md->write=0;
      ImageStreamIO_sempost(&m_imageStream,-1);
      
      //Now we increment pointers outside the time-critical part of the loop.
      next_cnt1 = m_imageStream.md->cnt1+1;
      if(next_cnt1 >= m_circBuffLength) next_cnt1 = 0;
      
      next_dest = (char *) m_imageStream.array.raw + next_cnt1*m_width*m_height*m_typeSize;
      next_wtimearr = &m_imageStream.writetimearray[next_cnt1];
      next_atimearr = &m_imageStream.atimearray[next_cnt1];
      next_cntarr = &m_imageStream.cntarray[next_cnt1];
      
      ++findex;
      if(findex >= m_frames.planes()) findex = 0;
      
      
      double ct = mx::get_curr_time();
      delta += 0.1 * (ct-lastSend - 1.0/m_fps);
      lastSend = ct;
            
            
      if(1./m_fps - delta > 0) microsleep( (1./m_fps - delta)*1e6 );
   } 
   
   
   ImageStreamIO_destroyIm( &m_imageStream );
   
   return 0;
}

