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


struct buffer buf_init(size_t blocksize, size_t linesize, size_t numblocks)
{
        buffer buf;
        
        buf.block_size = blocksize;
        buf.line_size = linesize;
        
        buf.buffer = (double *)malloc(blocksize * numblocks * sizeof(double));
        buf.blocks = (double **)malloc(numblocks * sizeof(double *));

        buf.dataptr = buf.buffer;

        for (size_t i = 0; i <= numblocks; ++i)
                buf.blocks[i] = buf.buffer + (i * blocksize);

        return buf;
}


void add_buf_line(struct buffer* buf, double* src)
{
        memcpy(buf->dataptr, src, buf->line_size * sizeof(double));
        buf->dataptr += buf->line_size;
}


void add_buf_block(struct buffer* buf, double* src)
{
        memcpy(buf->dataptr, src, buf->block_size * sizeof(double));
        buf->dataptr += buf->block_size;
}


void copy_buf_block(struct buffer* buf, double* dest, double* src)
{
        memcpy(dest, src, buf->block_size * sizeof(double));
}


void free_buf(struct buffer buf)
{
        free(buf.buffer);
        free(buf.blocks);
}
