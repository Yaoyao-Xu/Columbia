
#include "dji_linux_helpers.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "dji_vehicle.hpp"
#include "dji_perception.hpp"
#include <vector>
#include <iostream>
#include <cstdio>
#include <chrono>
#ifdef OPEN_CV_INSTALLED
  #include "opencv2/opencv.hpp"
  #include "opencv2/highgui/highgui.hpp"
using namespace cv;
#endif

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timer;
typedef std::chrono::duration<float> duration;

using namespace DJI::OSDK;
using namespace std;

Mat src, dst;
vector<Point> pointSet;
static int width = 20;
float calAngle(Vec2f a, Vec2f b)
{
	float cosr = (a[0] * b[0] + a[1] * b[1]) / (sqrt(a[0] * a[0] + a[1] * a[1]) * sqrt(b[0] * b[0] + b[1] * b[1]));
	return acos(cosr) * 180 / CV_PI;
}

bool isQua(Point paLeft, Point paRight, Point pbLeft, Point pbRight, Point p) 
{
	Vec2f vec1(paLeft.x - p.x, paLeft.y - p.y), vec2(paRight.x - p.x, paRight.y - p.y),
		  vec3(pbLeft.x - p.x, pbLeft.y - p.y), vec4(pbRight.x - p.x, pbRight.y - p.y);

	float angle = calAngle(vec1, vec2) + calAngle(vec2, vec3) + calAngle(vec3, vec4) + calAngle(vec4, vec1);
	return abs(angle - 360.f) <= 6;
}

cv::Point drawContours(Mat& mask, Mat& srcImage, vector<cv::Point>& pts)
{
	//通过最大连通域查找到叶片区域，且唯一
	//每一行从左向右查找，找到的第一个点为叶片边缘左极限点，最后一个点为叶片边缘右极限点
	int row = mask.rows;
	int col = mask.cols;
	vector<Point> leftContour, rightContour;

	for (int i = 0; i < row; ++i) {
		vector<Point> whiteLine;
		for (int j = 0; j < col; ++j) {
			if (mask.at<uchar>(i, j) == 255) {
				whiteLine.push_back(Point(j, i));
			}
		}
		if (!whiteLine.empty()) {
			leftContour.push_back(whiteLine.front());
			rightContour.push_back(whiteLine.back());
		}
	}

	//将leftContour和rightContour弹出边界点
	for (auto iter = leftContour.begin(); iter != leftContour.end();) {
		if (iter->x == 0 || iter->y == 0) {
			iter = leftContour.erase(iter);
		}
		else {
			++iter;
		}
	}

	for (auto iter = rightContour.begin(); iter != rightContour.end();) {
		if (iter->x == srcImage.cols - 1 || iter->y == srcImage.rows - 1) {
			iter = rightContour.erase(iter);
		}
		else {
			++iter;
		}
	}

	//y值从上到下，x数值先减后增从而分成两部分
	//根据点集拟合直线并且绘制左边缘直线
	Vec4f line_para1, line_para2, line_para3;
	for (auto point : leftContour) {
		cv::circle(srcImage, point, 1, Scalar(0, 255, 255));
	}

	fitLine(leftContour, line_para1, cv::DIST_L1, 0, 1e-2, 1e-2);
	double vx = line_para1[0];
	double vy = line_para1[1];
	double x = line_para1[2];
	double y = line_para1[3];
	if (vx < 0.01) {
		Vec4f line_para1;
		fitLine(leftContour, line_para1, cv::DIST_L2, 0, 1e-2, 1e-2);
		vx = line_para1[0];
		vy = line_para1[1];
		x = line_para1[2];
		y = line_para1[3];
	}

	double k1 = vy / vx;
	double angle = atan(k1);

	int left_y = int((-x * vy / vx) + y);
	int right_y = int(((srcImage.cols - x) * vy / vx) + y);
	cv::Point pLeft1, pLeft2;

	if (k1 > 0) {
		if (left_y < 0 && right_y > srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / (vy / vx));
			int down_x = int(x + (srcImage.rows - 1 - y) / (vy / vx));
			pLeft1 = Point(up_x, 0), pLeft2 = Point(down_x, srcImage.rows - 1);
		}
		else if (left_y < 0 && right_y < srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / (vy / vx));
			pLeft1 = Point(up_x, 0), pLeft2 = Point(srcImage.cols - 1, right_y);
		}
		else if (left_y > 0 && right_y > srcImage.rows - 1) {
			int down_x = int(x + (srcImage.rows - 1 - y) / (vy / vx));
			pLeft1 = Point(0, left_y), pLeft2 = Point(down_x, srcImage.rows - 1);
		}
		else if (left_y > 0 && right_y < srcImage.rows - 1) {
			pLeft1 = Point(0, left_y), pLeft2 = Point(srcImage.cols - 1, right_y);
		}
	}
	else {
		if (right_y < 0 && left_y > srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / (vy / vx));
			int down_x = int(x + (srcImage.rows - 1 - y) / (vy / vx));
			pLeft1 = Point(down_x, srcImage.rows - 1), pLeft2 = Point(up_x, 0);
		}
		else if (right_y < 0 && left_y < srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / (vy / vx));
			pLeft1 = Point(0, left_y), pLeft2 = Point(up_x, 0);
		}
		else if (right_y > 0 && left_y > srcImage.rows - 1) {
			int down_x = int(x + (srcImage.rows - 1 - y) / (vy / vx));
			pLeft1 = Point(down_x, srcImage.rows - 1), pLeft2 = Point(srcImage.cols - 1, right_y);
		}
		else if (right_y > 0 && left_y < srcImage.rows - 1) {
			pLeft1 = Point(0, left_y), pLeft2 = Point(srcImage.cols - 1, right_y);
		}
	}
	if (pLeft1.x == pLeft2.x) {
		cv::line(srcImage, pLeft1, pLeft2, cv::Scalar(0, 0, 255), 2);
	}
	else {
		cv::line(srcImage, Point(srcImage.cols - 1, right_y), Point(0, left_y), cv::Scalar(0, 0, 255), 2);
	}
	
	//根据点集绘制右边缘直线
	for (auto point : rightContour) {
		cv::circle(srcImage, point, 1, Scalar(255, 0, 0));
	}
	fitLine(rightContour, line_para2, cv::DIST_L1, 0, 1e-2, 1e-2);
	vx = line_para2[0];
	vy = line_para2[1];
	x = line_para2[2];
	y = line_para2[3];
	if (vx < 0.01) {
		Vec4f line_para2;
		fitLine(rightContour, line_para2, cv::DIST_L2, 0, 1e-2, 1e-2);
		vx = line_para2[0];
		vy = line_para2[1];
		x = line_para2[2];
		y = line_para2[3];
	}
	double k2 = vy / vx;
	angle = atan(k2);

	int left_y2 = int((-x * vy / vx) + y);
	int right_y2 = int(((srcImage.cols - x) * vy / vx) + y);
	cv::Point pRight1, pRight2;

	if (k2 > 0) {
		if (left_y2 < 0 && right_y2 > srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / (vy / vx));
			int down_x = int(x + (srcImage.rows - 1 - y) / (vy / vx));
			pRight1 = Point(up_x, 0), pRight2 = Point(down_x, srcImage.rows - 1);
		}
		else if (left_y2 < 0 && right_y2 < srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / (vy / vx));
			pRight1 = Point(up_x, 0), pRight2 = Point(srcImage.cols - 1, right_y2);
		}
		else if (left_y2 > 0 && right_y2 > srcImage.rows - 1) {
			int down_x = int(x + (srcImage.rows - 1 - y) / (vy / vx));
			pRight1 = Point(0, left_y2), pRight2 = Point(down_x, srcImage.rows - 1);
		}
		else if (left_y2 > 0 && right_y2 < srcImage.rows - 1) {
			pRight1 = Point(0, left_y2), pRight2 = Point(srcImage.cols - 1, right_y2);
		}
	}
	else {
		if (right_y2 < 0 && left_y2 > srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / (vy / vx));
			int down_x = int(x + (srcImage.rows - 1 - y) / (vy / vx));
			pRight1 = Point(down_x, srcImage.rows - 1), pRight2 = Point(up_x, 0);
		}
		else if (right_y2 < 0 && left_y2 < srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / (vy / vx));
			pRight1 = Point(0, left_y2), pRight2 = Point(up_x, 0);
		}
		else if (right_y2 > 0 && left_y2 > srcImage.rows - 1) {
			int down_x = int(x + (srcImage.rows - 1 - y) / (vy / vx));
			pRight1 = Point(down_x, srcImage.rows - 1), pRight2 = Point(srcImage.cols - 1, right_y2);
		}
		else if (right_y2 > 0 && left_y2 < srcImage.rows - 1) {
			pRight1 = Point(0, left_y2), pRight2 = Point(srcImage.cols - 1, right_y2);
		}
	}
	if (pRight1.x == pRight2.x) {
		cv::line(srcImage, pRight1, pRight2, cv::Scalar(0, 0, 255), 3);
	}
	else {
		cv::line(srcImage, Point(srcImage.cols - 1, right_y2), Point(0, left_y2), cv::Scalar(0, 0, 255), 3);
	}

	//如何计算中轴线？计算过程要先上后下
	Point p1, p2;
	float upWidth, downWidth;
	if (k1 * k2 < 0) {
		p1 = Point((pLeft2.x + pRight1.x) / 2, (pLeft2.y + pRight1.y) / 2);
		p2 = Point((pLeft1.x + pRight2.x) / 2, (pLeft1.y + pRight2.y) / 2);
		upWidth = sqrt((pLeft2.x - pRight1.x) * (pLeft2.x - pRight1.x) + ((pLeft2.y - pRight1.y) * (pLeft2.y - pRight1.y)));
		downWidth= sqrt((pLeft1.x - pRight2.x) * (pLeft1.x - pRight2.x) + ((pLeft1.y - pRight2.y) * (pLeft1.y - pRight2.y)));
	}
	else {
		p1 = Point((pLeft1.x + pRight1.x) / 2, (pLeft1.y + pRight1.y) / 2);
		p2 = Point((pLeft2.x + pRight2.x) / 2, (pLeft2.y + pRight2.y) / 2);
		upWidth = sqrt((pLeft1.x - pRight1.x) * (pLeft1.x - pRight1.x) + ((pLeft1.y - pRight1.y) * (pLeft1.y - pRight1.y)));
		downWidth = sqrt((pLeft2.x - pRight2.x) * (pLeft2.x - pRight2.x) + ((pLeft2.y - pRight2.y) * (pLeft2.y - pRight2.y)));
	}

	width = (upWidth + downWidth) / 2;
	width /= 4;
	double k3 = double(p1.y - p2.y) / double(p1.x - p2.x);
	x = double(p1.x + p2.x) / 2;
	y = double(p1.y + p2.y) / 2;
	int left_y3 = int((-x * k3) + y);
	int right_y3 = int(((srcImage.cols - x) * k3) + y);
	cv::Point pMid1, pMid2;
	if (k3 > 0) {
		if (left_y3 < 0 && right_y3 > srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / k3);
			int down_x = int(x + (srcImage.rows - 1 - y) / k3);
			pMid1 = Point(up_x, 0), pMid2 = Point(down_x, srcImage.rows - 1);
		}
		else if (left_y3 < 0 && right_y3 < srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / k3);
			pMid1 = Point(up_x, 0), pMid2 = Point(srcImage.cols - 1, right_y3);
		}
		else if (left_y3 > 0 && right_y3 > srcImage.rows - 1) {
			int down_x = int(x + (srcImage.rows - 1 - y) / k3);
			pMid1 = Point(0, left_y3), pMid2 = Point(down_x, srcImage.rows - 1);
		}
		else if (left_y3 > 0 && right_y3 < srcImage.rows - 1) {
			pMid1 = Point(0, left_y3), pMid2 = Point(srcImage.cols - 1, right_y3);
		}
	}
	else {
		if (right_y3 < 0 && left_y3 > srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / k3);
			int down_x = int(x + (srcImage.rows - 1 - y) / k3);
			pMid1 = Point(down_x, srcImage.rows - 1), pMid2 = Point(up_x, 0);
		}
		else if (right_y3 < 0 && left_y3 < srcImage.rows - 1) {
			int up_x = int(x + (0 - y) / k3);
			pMid1 = Point(0, left_y3), pMid2 = Point(up_x, 0);
		}
		else if (right_y3 > 0 && left_y3 > srcImage.rows - 1) {
			int down_x = int(x + (srcImage.rows - 1 - y) / k3);
			pMid1 = Point(down_x, srcImage.rows - 1), pMid2 = Point(srcImage.cols - 1, right_y3);
		}
		else if (right_y3 > 0 && left_y3 < srcImage.rows - 1) {
			pMid1 = Point(0, left_y3), pMid2 = Point(srcImage.cols - 1, right_y3);
		}
	}
	//cv::line(srcImage, Point((pLeft1.x + pRight1.x) / 2, (pLeft1.y + pRight1.y) / 2), Point((pLeft2.x + pRight2.x) / 2, (pLeft2.y + pRight2.y) / 2), cv::Scalar(0, 0, 255), 3)
	if (p1.x == p2.x) 
	{
		cv::line(srcImage, p1, p2, cv::Scalar(0, 0, 255), 3);
		pMid1 = p1, pMid2 = p2;
	}
	else 
	{
		cv::line(srcImage, pMid1, pMid2, cv::Scalar(0, 0, 255), 3);
	}
	
	//中点计算
	cv::Point midPoint((pMid1.x+pMid2.x)/2,(pMid1.y+pMid2.y)/2);

	Vec2f vec((pMid2.x - pMid1.x) / sqrt((pMid2.x - pMid1.x) * (pMid2.x - pMid1.x) + (pMid2.y - pMid1.y) * (pMid2.y - pMid1.y)), (pMid2.y - pMid1.y) / sqrt((pMid2.x - pMid1.x) * (pMid2.x - pMid1.x) + (pMid2.y - pMid1.y) * (pMid2.y - pMid1.y)));
	double length = sqrt((pMid2.x - pMid1.x) * (pMid2.x - pMid1.x) + (pMid2.y - pMid1.y) * (pMid2.y - pMid1.y));

	pts.push_back(Point(int(pMid1.x + length * vec[0] * 1 / 3), int(pMid1.y + length * vec[1] * 1 / 3)));
	pts.push_back(Point(int(pMid1.x + length * vec[0] * 2 / 3), int(pMid1.y + length * vec[1] * 2 / 3)));

	return midPoint;
}

int maxX(Point a, Point b, Point c, Point d)
{
	return max(a.x, max(b.x, max(c.x, d.x)));
}

int maxX(const vector<Point>& points)
{
	int maxVal = INT_MIN;
	for (auto point : points) {
		maxVal = std::max(maxVal, point.x);
	}
	return maxVal;
}

int maxY(Point a, Point b, Point c, Point d)
{
	return max(a.y, max(b.y, max(c.y, d.y)));
}

int maxY(const vector<Point>& points)
{
	int maxVal = INT_MIN;
	for (auto point : points) {
		maxVal = std::max(maxVal, point.y);
	}
	return maxVal;
}

int minX(Point a, Point b, Point c, Point d)
{
	return min(a.x, min(b.x, min(c.x, d.x)));
}

int minX(const vector<Point>& points)
{
	int minVal = INT_MAX;
	for (auto point : points) {
		minVal = std::min(minVal, point.x);
	}
	return minVal;
}

int minY(Point a, Point b, Point c, Point d)
{
	return min(a.y, min(b.y, min(c.y, d.y)));
}

int minY(const vector<Point>& points)
{
	int minVal = INT_MAX;
	for (auto point : points) {
		minVal = std::min(minVal, point.y);
	}
	return minVal;
}

//matlab的HSV空间各个通道的取值范围是[0,1)，而opencv中的取值范围是H:(0-180) S:(0-255) V:(0-255)
vector<cv::Point> hsvSegment(Mat& Image, const vector<Point>& pa_pb, cv::Point &midPoint)
{
	//将原图限制在20万左右的像素大小 
	//cv::Mat Image = imread("D:/Blades/19.jpg");
	double tick_start = getTickCount();
	Mat hsvImage, dst, hImage, sImage, vImage;
	cvtColor(Image, hsvImage, CV_BGR2HSV);//H:色调 S:饱和度 V：明度 

	hImage.create(hsvImage.size(), hsvImage.depth());
	sImage.create(hsvImage.size(), hsvImage.depth());
	vImage.create(hsvImage.size(), hsvImage.depth());

	//分离Hue/色相通道
	int ch[] = { 0, 0 };
	mixChannels(&hsvImage, 1, &hImage, 1, ch, 1);
	//分离Saturation/饱和度通道
	int ch1[] = { 1, 0 };
	mixChannels(&hsvImage, 1, &sImage, 1, ch1, 1);
	//分离Value/色调通道
	int ch2[] = { 2, 0 };
	mixChannels(&hsvImage, 1, &vImage, 1, ch2, 1);

	//根据hsv色彩空间的图片，计算H S V三个分量上的直方图
	int channels = 0;
	int histSize[] = { 256 };
	float midranges[] = { 0,255 };
	const float* ranges[] = { midranges };
	MatND dstHist;//要输出的直方图数组
	Mat h_drawImage = Mat::zeros(Size(256, 256), CV_8UC3);

	calcHist(&hsvImage, 1, &channels, Mat(), dstHist, 1, histSize, ranges);
	double g_dhistMaxVale;
	minMaxLoc(dstHist, 0, &g_dhistMaxVale);
	for (int i = 0; i < 256; ++i) {
		int value = cvRound(256 * 0.9 * (dstHist.at<float>(i) / g_dhistMaxVale));
		cv::line(h_drawImage, Point(i, h_drawImage.rows - 1), Point(i, h_drawImage.rows - 1 - value), Scalar(255, 0, 0));
	}
	channels = 1;
	Mat s_drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
	calcHist(&hsvImage, 1, &channels, Mat(), dstHist, 1, histSize, ranges);
	for (int i = 0; i < 256; ++i) {
		int value = cvRound(256 * 0.9 * (dstHist.at<float>(i) / g_dhistMaxVale));
		cv::line(s_drawImage, Point(i, s_drawImage.rows - 1), Point(i, s_drawImage.rows - 1 - value), Scalar(0, 255, 0));
	}
	channels = 2;
	Mat v_drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
	calcHist(&hsvImage, 1, &channels, Mat(), dstHist, 1, histSize, ranges);
	for (int i = 0; i < 256; ++i) {
		int value = cvRound(256 * 0.9 * (dstHist.at<float>(i) / g_dhistMaxVale));
		cv::line(v_drawImage, Point(i, v_drawImage.rows - 1), Point(i, v_drawImage.rows - 1 - value), Scalar(0, 0, 255));
	}

	//根据上一帧图片提取出来的中轴线信息，划定要分割的阈值范围
	Point pa = pa_pb[0], pb = pa_pb[1];//上一帧图像的中轴线端点，目前在当前图片来做实验

	//我们都确保pa在左边
	if (pa.x > pb.x) {
		Point temp = pa;
		pa = pb;
		pb = temp;
	}

	//计算单位向量，并计算相应的垂向量
	Vec2f vec((pb.x - pa.x) / sqrt((pb.x - pa.x) * (pb.x - pa.x) + (pb.y - pa.y) * (pb.y - pa.y)), (pb.y - pa.y) / sqrt((pb.x - pa.x) * (pb.x - pa.x) + (pb.y - pa.y) * (pb.y - pa.y)));
	Vec2f vecNormal(vec[1], -vec[0]);

	Point paLeft, paRight;//建议选取中间叶片长度的1/5或者1/4
	Point pbLeft, pbRight;

	if (abs(pa.x - pb.x) < 10) {
		paLeft.x = pa.x - width, paLeft.y = pa.y;
		paRight.x = pa.x + width, paRight.y = pa.y;
		pbLeft.x = pb.x - width, pbLeft.y = pb.y;
		pbRight.x = pb.x + width, pbRight.y = pb.y;
	}
	else if (abs(pa.y - pb.y) < 10) {
		paLeft.x = pa.x, paLeft.y = pa.y - width;
		paRight.x = pa.x, paRight.y = pa.y + width;
		pbLeft.x = pb.x, pbLeft.y = pb.y - width;
		pbRight.x = pb.x, pbRight.y = pb.y + width;
	}
	else if (pa.y > pb.y) {
		paLeft.x = pa.x - width * vecNormal[0], paLeft.y = pa.y - width * vecNormal[1];
		paRight.x = pa.x + width * vecNormal[0], paRight.y = pa.y + width * vecNormal[1];
		pbLeft.x = pb.x - width * vecNormal[0], pbLeft.y = pb.y - width * vecNormal[1];
		pbRight.x = pb.x + width * vecNormal[0], pbRight.y = pb.y + width * vecNormal[1];
	}
	else if (pa.y < pb.y) {
		paLeft.x = pa.x + width * vecNormal[0], paLeft.y = pa.y + width * vecNormal[1];
		paRight.x = pa.x - width * vecNormal[0], paRight.y = pa.y - width * vecNormal[1];
		pbLeft.x = pb.x + width * vecNormal[0], pbLeft.y = pb.y + width * vecNormal[1];
		pbRight.x = pb.x - width * vecNormal[0], pbRight.y = pb.y - width * vecNormal[1];
	}

	circle(Image, paLeft, 2, Scalar(0, 255, 0));
	circle(Image, paRight, 2, Scalar(0, 255, 0));
	circle(Image, pbLeft, 2, Scalar(0, 255, 0));
	circle(Image, pbRight, 2, Scalar(0, 255, 0));

	vector<vector<cv::Point>> points;
	vector<cv::Point> Points;

	//todo代码，会缺失一小段，修改代码来补充上
	for (int i = minY(paLeft, paRight, pbLeft, pbRight); i <= maxY(paLeft, paRight, pbLeft, pbRight); ++i) {
		vector<cv::Point> pointLine;
		for (int j = minX(paLeft, paRight, pbLeft, pbRight); j <= maxX(paLeft, paRight, pbLeft, pbRight); ++j) {
			if (isQua(paLeft, paRight, pbLeft, pbRight, Point(j, i))) {
				pointLine.push_back(Point(j, i));
			}
			if (j == paLeft.x && i == paLeft.y) {
				pointLine.push_back(paLeft);
			}
			else if (j == paRight.x && i == paRight.y) {
				pointLine.push_back(paRight);
			}
			else if (j == pbLeft.x && i == pbLeft.y) {
				pointLine.push_back(pbLeft);
			}
			else if (j == pbRight.x && i == pbRight.y) {
				pointLine.push_back(pbRight);
			}
		}
		points.push_back(pointLine);
	}
	//点的数量不多，不会对运行效率造成影响
	for (int i = 0; i < points.size(); ++i) {
		int min_x = minX(points[i]), max_x = maxX(points[i]);
		for (int j = min_x; j <= max_x; ++j) {
			int yVal = points[i][0].y;
			Points.push_back(Point(j, yVal));
		}
	}
	for (auto Point : Points) {
		circle(Image, Point, 1, Scalar(0, 255, 0));
	}
	int lowh = INT_MAX, maxh = INT_MIN;
	int lows = INT_MAX, maxs = INT_MIN;
	int lowv = INT_MAX, maxv = INT_MIN;
	for (auto Point : Points) {
		lowh = std::min(lowh, int(hsvImage.at<Vec3b>(Point)[0]));
		maxh = std::max(maxh, int(hsvImage.at<Vec3b>(Point)[0]));
		lows = std::min(lows, int(hsvImage.at<Vec3b>(Point)[1]));
		maxs = std::max(maxs, int(hsvImage.at<Vec3b>(Point)[1]));
		lowv = std::min(lowv, int(hsvImage.at<Vec3b>(Point)[2]));
		maxv = std::max(maxv, int(hsvImage.at<Vec3b>(Point)[2]));
	}
	const int offset = 20;
	lowh -= offset; lows -= offset; lowv -= offset;
	if (lowh < 0)    lowh = 0;
	if (lows < 0)    lows = 0;
	if (lowv < 0)    lowv = 0;
	maxh += offset; maxs += offset; maxv += offset;
	//得到初始的分割图像
	Mat mask = Mat::zeros(hsvImage.size(), CV_8UC1);
	for (int i = 0; i < hsvImage.rows; ++i) {
		for (int j = 0; j < hsvImage.cols; ++j) {
			if ((hsvImage.at<Vec3b>(i, j)[0] * 1 >= lowh && hsvImage.at<Vec3b>(i, j)[0] * 1 <= maxh) &&
				(hsvImage.at<Vec3b>(i, j)[1] * 1 >= lows && hsvImage.at<Vec3b>(i, j)[1] * 1 <= maxs) &&
				(hsvImage.at<Vec3b>(i, j)[2] * 1 >= lowv && hsvImage.at<Vec3b>(i, j)[2] * 1 <= maxv)) {
				mask.at<uchar>(i, j) = 255;
			}
		}
	}
	//对图像做先腐蚀再膨胀（开运算），消除杂点和影响数据点，去除噪声
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(9, 9), Point(-1, -1));
	erode(mask, mask, structureElement); //腐蚀
	structureElement = getStructuringElement(MORPH_RECT, Size(13, 13), Point(-1, -1));
	dilate(mask, mask, structureElement);

	//对图像先膨胀再腐蚀（闭运算），填充内部空洞
	structureElement = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
	dilate(mask, mask, structureElement);
	structureElement = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
	erode(mask, mask, structureElement); //腐蚀

	vector<vector<cv::Point>> contours;
	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// 寻找最大连通域  
	double maxArea = 0;
	vector<cv::Point> maxContour;
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			maxContour = contours[i];
		}
	}
	Mat MASK = Mat::zeros(hsvImage.size(), CV_8UC1);
	for (auto point : maxContour) {
		MASK.at<uchar>(point) = 255;
	}
	//找到最终的mask，分割出边缘线和中轴线
	vector<cv::Point> pts;
	midPoint=drawContours(MASK, Image, pts);
	//Image = MASK;
	double tick_end = getTickCount();
	cout << " " << (tick_end - tick_start) / getTickFrequency() * 1000 << "ms" << endl;
	return pts;
}

void on_mouse(int event, int x, int y, int flags, void* param)
{
	//注意为何使用static，首次调用时分配空间，之后无需重新分配内存空间
	static cv::Point pre_pt = { -1,-1 };
	static cv::Point cur_pt = { -1,-1 };

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		dst.copyTo(src);
		pre_pt = cv::Point(x, y);
		cv::circle(src, pre_pt, 3, cv::Scalar(255, 0, 0), CV_FILLED, CV_AA, 0);
		imshow("Live", src);
		src.copyTo(dst);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		dst.copyTo(src);
		cur_pt = cvPoint(x, y);
		cv::line(src, pre_pt, cur_pt, cv::Scalar(0, 255, 0), 1, CV_AA, 0);
		imshow("Live", src);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		dst.copyTo(src);
		cur_pt = cv::Point(x, y);
		cv::circle(src, cur_pt, 3, cv::Scalar(255, 0, 0), CV_FILLED, CV_AA, 0);
		cv::line(src, pre_pt, cur_pt, cv::Scalar(0, 255, 0), 1, CV_AA, 0);
		imshow("Live", src);
		src.copyTo(dst);
		pointSet.push_back(pre_pt);
		pointSet.push_back(cur_pt);
	}

}

/*
int main2(int argc, char* argv[])
{
	const char winName[] = "src";
	Mat frame;
	VideoCapture cap;

	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	cap.open(deviceID, apiID);
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	std::size_t nFrames = 0;

	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		 << "Press any key to terminate" << endl;
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		frame.copyTo(src);
		src.copyTo(dst);
		nFrames++;
		if (nFrames < 100) {
			cv::namedWindow("Live", CV_WINDOW_AUTOSIZE);
			imshow("Live", frame);
			if (waitKey(5) >= 0)
				break;
		}
		if(nFrames >= 100) {
			if (nFrames == 100) {
				cv::setMouseCallback("Live", on_mouse, 0);
				cv::waitKey(0);//按键暂停，等待按键处理
			}
			vector<cv::Point> pa_pb = hsvSegment(frame, pointSet);
			//imwrite("res.jpg", frame);
			pa_pb.swap(pointSet);
			cv::waitKey(10);//只为了展示使用，hsvSegment处理时间长，其他时间占比少，所以设置一小段时间缓冲一下
			imshow("Live", frame);
		}
	}
	return 0;
	
}
*/
int main(int argc,char **argv)
{
	// Setup OSDK
	//初始化OSDK相关的配置，初始化
	bool enableAdvancedSensing = true;
    LinuxSetup linuxEnvironment(argc, argv, enableAdvancedSensing);
    Vehicle* vehicle = linuxEnvironment.getVehicle();
    const char *acm_dev = linuxEnvironment.getEnvironment()->getDeviceAcm().c_str();
    vehicle->advancedSensing->setAcmDevicePath(acm_dev);
    if (vehicle == NULL)
    {
       std::cout << "Vehicle not initialized, exiting.\n";
       return -1;
    }

	int functionTimeout=1;
	vehicle->control->obtainCtrlAuthority(functionTimeout);

	bool mainCamResult = vehicle->advancedSensing->startMainCameraStream();
    if(!mainCamResult)
    {
       cout << "Failed to Open Camera" << endl;
       return 1;
    }
    CameraRGBImage mainImg;
	const char winName[]="src";
	char message1[100];
	char message2[100];

	std::size_t nFrames=0;
	for(;;)
	{
		if(vehicle->advancedSensing->newMainCameraImageReady())
		{
			int dx=0;
			int dy=0;			
			int yawRate=0;
			int pitchRate=0;
			timer gimbalTrackerStartTime,gimbalTrackerFinishTime;
			duration gimbalTrackerTimeDiff;

			vehicle->advancedSensing->getMainCameraImage(mainImg);
			cv::Mat frame(mainImg.height, mainImg.width, CV_8UC3, mainImg.rawData.data(), mainImg.width*3);
			resize(frame, frame, Size(), 0.45, 0.45);
			frame.copyTo(src);
			src.copyTo(dst);
			nFrames++;
			if (nFrames < 100) 
			{
				cv::namedWindow("Live", CV_WINDOW_AUTOSIZE);
				imshow("Live", frame);
				if (waitKey(5) >= 0)
				{
					break;
				}
			}
			if(nFrames >= 100) 
			{
				if (nFrames == 100) 
				{
					cv::setMouseCallback("Live", on_mouse, 0);
					cv::waitKey(0);//按键暂停，等待按键处理
				}
				cv::Point midPoint;
				vector<cv::Point> pa_pb = hsvSegment(frame, pointSet, midPoint);
				pa_pb.swap(pointSet);
				
				//云台跟踪相关
				dx=(int)(midPoint.x-frame.cols/2);
				dy=(int)(midPoint.y-frame.rows/2);
				yawRate  = dx;
				pitchRate=-dy;
				if(abs(dx)<10)    yawRate=0;
				if(abs(dy)<10)    pitchRate=0;

				DJI::OSDK::Gimbal::SpeedData gimbalSpeed;
				gimbalSpeed.roll=0;
				gimbalSpeed.pitch=pitchRate;
				gimbalSpeed.yaw=yawRate;
				gimbalSpeed.gimbal_control_authority=1;
				vehicle->gimbal->setSpeed(&gimbalSpeed);

				dx=midPoint.x-frame.cols/2;
				dy=midPoint.y-frame.rows/2;

				//图像正中心的位置
				cv::circle(frame,Point(frame.cols/2,frame.rows/2),5,cv::Scalar(255,0,0),2,8);
				//中轴线中点位置
				cv::circle(frame,Point(midPoint.x,midPoint.y),3,cv::Scalar(0,0,255),1,8);
				//两点之间差距
				cv::line(frame,Point(frame.cols/2,frame.rows/2),Point(midPoint.x,midPoint.y),cv::Scalar(0,255,255));
				
				cvtColor(frame, frame, COLOR_RGB2BGR);
      			sprintf(message1,"dx=%04d, dy=%04d",dx, dy);
      			putText(frame, message1, Point2f(20,30), FONT_HERSHEY_SIMPLEX, 1,  Scalar(0,255,0));
				cv::waitKey(10);//只为了展示使用，hsvSegment处理时间长，其他时间占比少，所以设置一小段时间缓冲一下
				imshow("Live", frame);
			}
		}
	}
	vehicle->advancedSensing->stopMainCameraStream();
}
