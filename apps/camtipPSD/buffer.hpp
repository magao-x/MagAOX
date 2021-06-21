/* buffer.h
 *
 * Collection of tools for handling
 * input and output of large sets 
 * of 2D data
 *
 * Author: Jacob Trzaska
 */

#pragma once // BUFFER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/**
 * Data structure for holding values.
 */
class buffer {
   public:
      double*  buffer_start; 
      double** blocks;
      double*  dataptr;

      size_t block_size;
      size_t line_size;

      buffer();
      buffer(size_t blocksize, size_t linesize, size_t numblocks);
      ~buffer();

      void add_buf_line(double* src);
      void add_buf_block(double* src);
      void copy_buf_block(double* dest, double* src);

      void buf_init(size_t block_size, size_t linesize, size_t numblocks);
};
