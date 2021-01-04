
#include "../../../tests/catch2/catch.hpp"

#include "../streamWriter.hpp"

namespace MagAOX
{
namespace app
{
struct streamWriter_test
{
   streamWriter * m_sw;
   
   streamWriter_test(streamWriter * sw)
   {
      m_sw = sw;
   }
   
   std::string rawimageDir(){ return m_sw->m_rawimageDir; }
   
   
   int setup_circbufs( int width, 
                       int height,
                       int dataType,
                       int circBuffLength
                     )
   {
      m_sw->m_width = width;
      m_sw->m_height = height;
      m_sw->m_dataType = dataType;
      m_sw->m_typeSize = ImageStreamIO_typesize(m_sw->m_dataType);
      std::cerr << m_sw->m_typeSize << "\n";
      
      m_sw->m_circBuffLength = circBuffLength;
                       
      return m_sw->allocate_circbufs();
   }
   
   // Sets m_writeChunkLength and calls allocate_xrif
   // Call this *only* after setup_circbufs.
   int setup_xrif( int writeChunkLength )
   {
      m_sw->m_writeChunkLength = writeChunkLength;
    
      m_sw->initialize_xrif();
      return m_sw->allocate_xrif();
   }
   
   //Allocates and populates the filename buffer
   int setup_fname()
   {
      m_sw->m_fnameBase = "/tmp/swtest_";
      
      m_sw->m_fnameSz = m_sw->m_fnameBase.size() + sizeof("YYYYMMDDHHMMSSNNNNNNNNN.xrif"); //the sizeof includes the \0
      m_sw->m_fname = (char*) malloc(m_sw->m_fnameSz);
      
      snprintf(m_sw->m_fname, m_sw->m_fnameSz, "%sYYYYMMDDHHMMSSNNNNNNNNN.xrif", m_sw->m_fnameBase.c_str());
      
      return 0;
   }
   
   //Fill the circular buffers.
   int fill_circbuf_uint16()
   {
      //fill in image data with increasing 256 bit vals.
      for(size_t pp =0; pp < m_sw->m_circBuffLength; ++pp)
      {
         uint16_t v = pp;
         for(size_t rr =0; rr < m_sw->m_width; ++rr)
         {
            for(size_t cc =0; cc < m_sw->m_height; ++cc)
            {
               ((uint16_t *)m_sw->m_rawImageCircBuff)[pp*m_sw->m_width*m_sw->m_height + rr*m_sw->m_height + cc] = v;
               ++v;
            }
         }
         
         //fitsFile<uint16_t> ff;
         //ff.write("cb.fits", m_sw->m_rawImageCircBuff);
                  
         //Fill in timing values with unique vals.
         uint64_t * curr_timing = m_sw->m_timingCircBuff + 5*pp;
         curr_timing[0] = pp; //image number
         curr_timing[1] = pp + 1000; //atime sec
         curr_timing[2] = pp + 2000; //atime nsec
         curr_timing[3] = pp + m_sw->m_circBuffLength + 1000; //wtime sec
         curr_timing[4] = pp + m_sw->m_circBuffLength + 2000; //wtime nsec
      }
      return 0;
   }
   
   int write_frames( int start, // arbitrary.
                     int stop  //should be a m_writeChunkLength boundary
                   )
   {
      m_sw->m_currSaveStart = start;
      m_sw->m_currSaveStop = stop;
      m_sw->m_currSaveStopFrameNo = stop;
      
      m_sw->m_writing = WRITING;
      return m_sw->doEncode();
   }
   
   //Read the xrif archive back in and compare the results.
   int comp_frames_uint16( size_t start,
                           size_t stop
                         )
   {
      
      std::cout << "Reading: " << m_sw->m_fname << "\n";
  
      xrif_t xrif;
      xrif_error_t xrv = xrif_new(&xrif);
      
      char header[XRIF_HEADER_SIZE];
      
      FILE * fp_xrif = fopen(m_sw->m_fname, "rb");
      size_t nr = fread(header, 1, XRIF_HEADER_SIZE, fp_xrif);
      
      if(nr != XRIF_HEADER_SIZE)
      {
         std::cerr << "Error reading header of " << m_sw->m_fname  << "\n";
         fclose(fp_xrif);
         return -1;
      }

      uint32_t header_size;
      xrif_read_header(xrif, &header_size , header);
      
      int rv = 0;
      if(xrif_width(xrif) != m_sw->m_width)
      {
         std::cerr << "width mismatch\n";
         rv = -1;
      }
      
      if(xrif_height(xrif) != m_sw->m_height)
      {
         std::cerr << "height mismatch\n";
         rv = -1;
      }
      
      if(xrif_depth(xrif) != 1)
      {
         std::cerr << "depth mismatch\n";
         rv = -1;
      }
      
      if(xrif_frames(xrif) != stop-start )
      {
         std::cerr << "frames mismatch\n";
         rv = -1;
      }
      
      xrif_allocate(xrif); 
      

      nr = fread(xrif->raw_buffer, 1, xrif->compressed_size, fp_xrif);
      
      if(nr != xrif->compressed_size)
      {
         std::cerr << "error reading compressed image buffer.\n";
         return -1;
      }
      
      xrv = xrif_decode(xrif); 
      if(xrv != XRIF_NOERROR)
      {
         std::cerr << "error decoding compressed image buffer. Code: " << xrv << "\n";
         return -1;
      }
      
      size_t badpix = 0;
      
      for(size_t n=0; n< m_sw->m_width*m_sw->m_height*m_sw->m_typeSize*(stop-start); ++n)
      {
         if( m_sw->m_rawImageCircBuff[start*m_sw->m_width*m_sw->m_height*m_sw->m_typeSize + n] != xrif->raw_buffer[n] ) ++badpix;
      }
      
      if(badpix > 0)
      {
         std::cerr << "Buffers don't match: " << badpix << " bad pixels.\n";
         return -1;
      }
      
      return rv;
   }
   
};
}
}

using namespace MagAOX::app;

SCENARIO( "streamWriter Configuration", "[streamWriter]" ) 
{
   GIVEN("A default constructed streamWriter")
   {
      streamWriter sw;
      streamWriter_test sw_test(&sw);
      
      WHEN("default configurations")
      {
         REQUIRE(sw_test.rawimageDir() == "");
         
         
      }
   }
}

SCENARIO( "streamWriter encoding data", "[streamWriter]" ) 
{
   GIVEN("A default constructed streamWriter and a 120x120 uint16 stream")
   {
      streamWriter sw;
      streamWriter_test sw_test(&sw);
      
      WHEN("writing full 1st chunk")
      {
         int circBuffLength = 10;
         int writeChunkLength = 5;
         REQUIRE(sw_test.setup_circbufs(120, 120, XRIF_TYPECODE_UINT16, circBuffLength) == 0);
         REQUIRE(sw_test.setup_xrif(writeChunkLength) == 0);
         REQUIRE(sw_test.setup_fname() == 0);
         
         REQUIRE(sw_test.fill_circbuf_uint16() == 0);
         
         REQUIRE(sw_test.write_frames(0,5) == 0);
         
         REQUIRE(sw_test.comp_frames_uint16(0,5) == 0);
      }
      
      WHEN("writing full 2nd chunk")
      {
         int circBuffLength = 10;
         int writeChunkLength = 5;
         REQUIRE(sw_test.setup_circbufs(120, 120, XRIF_TYPECODE_UINT16, circBuffLength) == 0);
         REQUIRE(sw_test.setup_xrif(writeChunkLength) == 0);
         REQUIRE(sw_test.setup_fname() == 0);
         
         REQUIRE(sw_test.fill_circbuf_uint16() == 0);
         
         REQUIRE(sw_test.write_frames(5,10) == 0);
         
         REQUIRE(sw_test.comp_frames_uint16(5,10) == 0);
      }
      
      WHEN("writing partial 1st chunk")
      {
         int circBuffLength = 10;
         int writeChunkLength = 5;
         REQUIRE(sw_test.setup_circbufs(120, 120, XRIF_TYPECODE_UINT16, circBuffLength) == 0);
         REQUIRE(sw_test.setup_xrif(writeChunkLength) == 0);
         REQUIRE(sw_test.setup_fname() == 0);
         
         REQUIRE(sw_test.fill_circbuf_uint16() == 0);
         
         REQUIRE(sw_test.write_frames(2,5) == 0);
         
         REQUIRE(sw_test.comp_frames_uint16(2,5) == 0);
      }
      
      WHEN("writing partial 2nd chunk")
      {
         int circBuffLength = 10;
         int writeChunkLength = 5;
         REQUIRE(sw_test.setup_circbufs(120, 120, XRIF_TYPECODE_UINT16, circBuffLength) == 0);
         REQUIRE(sw_test.setup_xrif(writeChunkLength) == 0);
         REQUIRE(sw_test.setup_fname() == 0);
         
         REQUIRE(sw_test.fill_circbuf_uint16() == 0);
         
         REQUIRE(sw_test.write_frames(5,8) == 0);
         
         REQUIRE(sw_test.comp_frames_uint16(5,8) == 0);
      }
   }
}
