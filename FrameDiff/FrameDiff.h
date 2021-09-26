#pragma once

#include <vector>
#include "IOUT.h"

using namespace cv;



class Anomaly
{
	public:	
		Anomaly();
		explicit Anomaly(cv::Mat m_background);
		virtual ~Anomaly();
		void FindDiff(cv::Mat& CurrentFrame, std::vector<BoundingBox> &yolov5, std::vector<cv::Rect> &m_left);
		bool DetectShadow(cv::Mat &m_images);
		bool PointsinRegion(std::vector<cv::Point>& pt, const std::vector<cv::Point>& polygons);
		void UpdateBack(cv::Mat& background, bool update);
		void postprocess(std::vector<std::vector<LeftObjects>> &m_cadidateset);
	
	protected:
	private:
		
		cv::Mat imgback;
		bool updateornot;
		
	
};