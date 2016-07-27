
#include "extractTable.hpp"

using namespace cv;
using namespace std;

vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
void extractTable(string &argv) {
	Mat src = imread(argv), dst;
	if (!src.data) {
		cout << "not picture" << endl;
	}
	Mat canny,gray,sobel, edge,erod, blur;

	double src_height=src.cols, src_width=src.rows;
	namedWindow("source", WINDOW_NORMAL);
	imshow("source", src);

	//先转为灰度图
	cvtColor(src, gray, COLOR_BGR2GRAY);


	//腐蚀（黑色区域变大）
	int erodeSize = src_height / 300;
	if (erodeSize % 2 == 0)
		erodeSize++;
	Mat element = getStructuringElement(MORPH_RECT, Size(erodeSize, erodeSize));
	erode(gray, erod, element);

	//高斯模糊化
	int blurSize = src_height / 200;
	if (blurSize % 2 == 0)
		blurSize++;
	GaussianBlur(erod, blur, Size(blurSize, blurSize), 0, 0);
	//namedWindow("GaussianBlur", WINDOW_NORMAL);
	//imshow("GaussianBlur", blur);
	
	//封装的二值化
	Mat thresh = gray.clone();
	adaptiveThreshold(~gray, thresh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, -2);
	imshow("thresh",thresh);

	/*
	这部分的思想是将线条从横纵的方向处理后抽取出来，再进行交叉，矩形的点，进而找到矩形区域的过程
	*/
	// Create the images that will use to extract the horizonta and vertical lines
	Mat horizontal = thresh.clone();
	Mat vertical = thresh.clone();

	int scale = 20; // play with this variable in order to increase/decrease the amount of lines to be detected

	// Specify size on horizontal axis
	int horizontalsize = horizontal.cols / scale;

	// Create structure element for extracting horizontal lines through morphology operations
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));

	// Apply morphology operations
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	// dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1)); // expand horizontal lines

	// Show extracted horizontal lines
	namedWindow("horizontal", WINDOW_NORMAL);
	imshow("horizontal", horizontal);

	// Specify size on vertical axis
	int verticalsize = vertical.rows / scale;

	// Create structure element for extracting vertical lines through morphology operations
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));

	// Apply morphology operations
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));
	//    dilate(vertical, vertical, verticalStructure, Point(-1, -1)); // expand vertical lines

	// Show extracted vertical lines
	namedWindow("vertical", WINDOW_NORMAL);
	imshow("vertical", vertical);

	Mat mask = horizontal + vertical;
	namedWindow("mask", WINDOW_NORMAL);
	imshow("mask", mask);

	// find the joints between the lines of the tables, we will use this information in order to descriminate tables from pictures (tables will contain more than 4 joints while a picture only 4 (i.e. at the corners))
	Mat joints;
	bitwise_and(horizontal, vertical, joints);
	namedWindow("joints", WINDOW_NORMAL);
	imshow("joints", joints);

	// Find external contours from the mask, which most probably will belong to tables or to images
	vector<Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Mat> rois;

	for (size_t i = 0; i < contours.size(); i++)
	{
		// find the area of each contour
		double area = contourArea(contours[i]);

		//        // filter individual lines of blobs that might exist and they do not represent a table
		if (area < 100) // value is randomly chosen, you will need to find that by yourself with trial and error procedure
			continue;

		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));

		// find the number of joints that each table has
		Mat roi = joints(boundRect[i]);

		vector<vector<Point> > joints_contours;
		findContours(roi, joints_contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

		// if the number is not more than 5 then most likely it not a table
		if (joints_contours.size() <= 4)
			continue;

		rois.push_back(src(boundRect[i]).clone());

		//        drawContours( rsz, contours, i, Scalar(0, 0, 255), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
		rectangle(src, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 1, 8, 0);
	}

	for (size_t i = 0; i < rois.size(); ++i)
	{
		/* Now you can do whatever post process you want
		* with the data within the rectangles/tables. */
		
		std::stringstream ss;
		ss << "roi" << i << "";
		namedWindow(ss.str(), WINDOW_NORMAL);
		imshow(ss.str(), rois[i]);
		waitKey();
	}
	namedWindow("contours", WINDOW_NORMAL);
	imshow("contours", src);

	waitKey(0);
	destroyAllWindows();
}



//自适应阈值的Canny，获取low，high两个参数。
void AdaptiveFindThreshold(const cv::Mat src, double *low, double *high, int aperture_size)
{
	const int cn = src.channels();
	cv::Mat dx(src.rows, src.cols, CV_16SC(cn));
	cv::Mat dy(src.rows, src.cols, CV_16SC(cn));

	cv::Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0);
	cv::Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0);

	CvMat _dx = dx, _dy = dy;
	_AdaptiveFindThreshold(&_dx, &_dy, low, high);

}

// 仿照matlab，自适应求高低两个门限                                              
void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high)
{
	CvSize size;
	IplImage *imge = 0;
	int i, j;
	CvHistogram *hist;
	int hist_size = 255;
	float range_0[] = { 0,256 };
	float* ranges[] = { range_0 };
	double PercentOfPixelsNotEdges = 0.7;
	size = cvGetSize(dx);
	imge = cvCreateImage(size, IPL_DEPTH_32F, 1);
	// 计算边缘的强度, 并存于图像中                                          
	float maxv = 0;
	for (i = 0; i < size.height; i++)
	{
		const short* _dx = (short*)(dx->data.ptr + dx->step*i);
		const short* _dy = (short*)(dy->data.ptr + dy->step*i);
		float* _image = (float *)(imge->imageData + imge->widthStep*i);
		for (j = 0; j < size.width; j++)
		{
			_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
			maxv = maxv < _image[j] ? _image[j] : maxv;

		}
	}
	if (maxv == 0) {
		*high = 0;
		*low = 0;
		cvReleaseImage(&imge);
		return;
	}

	// 计算直方图                                                            
	range_0[1] = maxv;
	hist_size = (int)(hist_size > maxv ? maxv : hist_size);
	hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(&imge, hist, 0, NULL);
	int total = (int)(size.height * size.width * PercentOfPixelsNotEdges);
	float sum = 0;
	int icount = hist->mat.dim[0].size;

	float *h = (float*)cvPtr1D(hist->bins, 0);
	for (i = 0; i < icount; i++)
	{
		sum += h[i];
		if (sum > total)
			break;
	}
	// 计算高低门限                                                          
	*high = (i + 1) * maxv / hist_size;
	*low = *high * 0.4;
	cvReleaseImage(&imge);
	cvReleaseHist(&hist);
}