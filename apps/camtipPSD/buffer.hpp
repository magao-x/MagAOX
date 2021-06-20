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
struct buffer {
        double*  buffer; 
        double** blocks;
        double* dataptr;

        size_t block_size;
        size_t line_size;
};

void add_buf_line(struct buffer* buf, double* src);

void add_buf_block(struct buffer* buf, double* src);

// blocksize and linesize are the number of elements, not bytes
struct buffer buf_init(size_t blocksize, size_t linesize, size_t numblocks);

void copy_buf_block(struct buffer* buf, double* dest, double* src);

void free_buf(struct buffer buf);
