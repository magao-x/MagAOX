/**
 * buffer.c - defines functions for the buffer structure
 * Copyright (C) 2021  Jacob Trzaska
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "buffer.hpp"


buffer::buffer()
: buffer_start { nullptr },
  blocks { nullptr },
  dataptr { nullptr },
  block_size { 0 },
  line_size { 0 }
{
}

buffer::buffer(size_t blocksize, size_t linesize, size_t numblocks)
{

   block_size = blocksize;
   line_size = linesize;
        
   buffer_start = (double *)malloc(blocksize * numblocks * sizeof(double));
   blocks = (double **)malloc(numblocks * sizeof(double *));

   dataptr = buffer_start;

   for (size_t i = 0; i <= numblocks; ++i)
      blocks[i] = buffer_start + (i * blocksize);

}

void buffer::buf_init(size_t blocksize, size_t linesize, size_t numblocks)
{
   if (buffer_start == nullptr && blocks == nullptr)
   {
      block_size = blocksize;
      line_size = linesize;
        
      buffer_start = (double *)malloc(blocksize * numblocks * sizeof(double));
      blocks = (double **)malloc(numblocks * sizeof(double *));

      dataptr = buffer_start;

      for (size_t i {0}; i <= numblocks; ++i)
         blocks[i] = buffer_start + (i * blocksize);
   }
   else
   {
      block_size = blocksize;
      line_size = linesize;
   
      free(buffer_start); free(blocks);

      buffer_start = (double *)malloc(blocksize * numblocks * sizeof(double));
      blocks = (double **)malloc(numblocks * sizeof(double*));
      
      dataptr = buffer_start;

      for (size_t i {0}; i <= numblocks; ++i)
         blocks[i] = buffer_start + (i * blocksize);
   }
}


void buffer::add_buf_line(double* src)
{
   memcpy(dataptr, src, line_size * sizeof(double));
   dataptr += line_size;
}


void buffer::add_buf_block(double* src)
{
   memcpy(dataptr, src, block_size * sizeof(double));
   dataptr += block_size;
}


void buffer::copy_buf_block(double* dest, double* src)
{
   memcpy(dest, src, block_size * sizeof(double));
}


void buffer::buf_free()
{
   free(buffer_start);
   free(blocks);
}



buffer::~buffer()
{
   free(buffer_start);
   free(blocks);
}
