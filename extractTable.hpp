
#pragma once


#ifndef __EXTRACT_TABLE_HPP__
#define __EXTRACT_TABLE_HPP__
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>

extern void extractTable();
extern void extractTable(std::string &file);
extern void AdaptiveFindThreshold(cv::Mat src, double *low, double *high, int aperture_size = 3);
extern void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high);
#endif