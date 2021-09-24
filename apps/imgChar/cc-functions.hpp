#pragma once

/**
 *   Program: cc-functions.h
 *   Purpose: Collection of functions for image analysis
 */
#pragma  once 

#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>
#include <cmath>
#include "reg_functions.hpp"
#include <x86intrin.h>
#include <ImageStruct.h>
#include <ImageStreamIO.h>

/**
 *  Fits a cross-correlation map to Gaussian function
 *  @param ROWS     number of rows in input image
 *  @param COLS     number of columns in input image
 *  @param input    pointer to input image
 *  @param SZ       determines crop size (2 * SZ + 1)
 *  @param results  2-elements array containing shifts
 */
void GaussFit(size_t ROWS, // number of rows in input 
              size_t COLS, // number of cols in input 
              double *input, // input image 
              size_t SZ,      // determines size of crop region
              double *result // holds results of the fit 
              );


/**
 *  Calculates the strehl ratio of a modulated Point Spread Function
 *  @param detector pointer to PSF image
 *  @param ROWS     number of rows in image
 *  @param COLS     number of columns in image
 *  @param xctr     approximate center of ring pattern in X
 *  @param yctr     approximate center of ring pattern in Y
 */
double getStrehlMod(double *detector, size_t ROWS, size_t COLS, size_t xctr, size_t yctr);


/**
 * @name copy_image
 * @brief copy images from shmem to input buffer
 *
 * @return n/a
 */
void copy_image(double* in, void * image, size_t rows, size_t cols, int datatype);


/**
 * @name copy_image0
 * @brief copy template image in memory and find image center
 *
 * @return n/a
 */
void copy_image0(double* in, void * image, size_t rows, size_t cols, 
                 int datatype, size_t * xctr, size_t * yctr);



/**
 * @name    strehlAmp
 * @brief   get the Fpk/Ftot value for a given PSF
 *
 * @return  Fpk/Ftot
 */
double strehlAmp(void * image, size_t IMEND, int datatype);
