#include "FrameDiff.h"

Anomaly::Anomaly()
	:updateornot(false)
	
{

}

Anomaly::~Anomaly()
{

}

Anomaly::Anomaly(cv::Mat m_background)
{
	imgback = m_background.clone();
	
}

void Anomaly::postprocess(std::vector<std::vector<LeftObjects>>& m_cadidateset)
{

	if (m_cadidateset.size()<2)
	{
		return;
	}
	for (int i =0 ; i <m_cadidateset.size();i++)
	{
		for (int j = 0; j < m_cadidateset[i].size(); j++)
		{
			//m_cadidateset[i][j]
		}
	}

}

bool Anomaly::DetectShadow(cv::Mat& src)
{
	
	if (!src.data)
	{
		printf("Err: Path error \ n");
		return false;
	}
	imshow("1 src", src);

	//Regulate the window size according to the image size
	int n = 11;

	// Do the maximum filtering of the original grayscale map (not expanded), get maxfiltermat_a
	Mat element = getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(n, n));
	int iteration = 1;
	

	// Do you minimize the size of maxfiltermat_a (isn't it corroded)
	Mat maxFilterMat_A;
	cv::morphologyEx(src, maxFilterMat_A, MORPH_DILATE, element, cv::Point(-1, -1), iteration, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
	imshow("2 Maximum Filter", maxFilterMat_A);

	// Do you minimize the size of maxfiltermat_a (isn't it corroded)
	Mat minFilterMat_B;
	cv::morphologyEx(maxFilterMat_A, minFilterMat_B, MORPH_ERODE, element, cv::Point(-1, -1), iteration, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
	imshow("3 first maximum and then minimum filtering", minFilterMat_B);

	


	// (First maximum and then minimum filtering) - Original grayscale
	Mat diffMat = minFilterMat_B - src;
	imshow("4 minus the original picture", diffMat);
	// Reverse, get a black word on white
	diffMat = ~diffMat;
	imshow("5 minus the original picture and then take anti-", diffMat);
	Mat normalizeMat;
	normalize(diffMat, normalizeMat, 0, 255, NORM_MINMAX);
	imshow("6 results normalized to 0-255", normalizeMat);
	cv::waitKey(0);
	return true;
}


bool Anomaly::PointsinRegion(std::vector<cv::Point>& pt, const std::vector<cv::Point>& polygons)
{
	if (pt.size()<4)
	{
		printf("input rect failure!!<<<<<<");
		return false;
	}
	int numofNonintersection = 0;

	for (int j = 0; j < pt.size(); j++)
	{
	int nCross = 0;    // 定义变量，统计目标点向右画射线与多边形相交次数
	for (int i = 0; i < polygons.size(); i++) 
	{   //遍历多边形每一个节点

		cv::Point p1;
		cv::Point p2;

		p1 = polygons[i];
		p2 = polygons[(i + 1) % polygons.size()];  // p1是这个节点，p2是下一个节点，两点连线是多边形的一条边
// 以下算法是用是先以y轴坐标来判断的

		if (p1.y == p2.y)
			continue;   //如果这条边是水平的，跳过

		
			if (pt[j].y < min(p1.y, p2.y)) //如果目标点低于这个线段，跳过
				continue;

			if (pt[j].y >= max(p1.y, p2.y)) //如果目标点高于这个线段，跳过
				continue;
			//那么下面的情况就是：如果过p1画水平线，过p2画水平线，目标点在这两条线中间
			double x = (double)(pt[j].y - p1.y) * (double)(p2.x - p1.x) / (double)(p2.y - p1.y) + p1.x;
			// 这段的几何意义是 过目标点，画一条水平线，x是这条线与多边形当前边的交点x坐标
			if (x > pt[j].x)
				nCross++; //如果交点在右边，统计加一。这等于从目标点向右发一条射线（ray），与多边形各边的相交（crossing）次数
		}

	if (nCross % 2 == 1) {

		return true; //如果是奇数，说明在多边形里
	}
	else {
		//numofNonintersection++;
		//return false; //否则在多边形外 或 边上
	}
	}
	return false;
}

 void Anomaly::UpdateBack(cv::Mat& background, bool update)
{
	if (!update)
	{
		return;
	}
	else
	{
		printf(">>>>>>Update background now!!!!<<<<<<<<<<< \n");
		printf(">>>>>>Update background now!!!!<<<<<<<<<<< \n");
		printf(">>>>>>Update background now!!!!<<<<<<<<<<< \n");
		printf(">>>>>>Update background now!!!!<<<<<<<<<<< \n");
		background.copyTo(imgback);
	}
}

 

 void Anomaly::FindDiff(cv::Mat& CurrentFrame, std::vector<BoundingBox>& yolov5, \
	 std::vector<cv::Rect>& m_left)
{
	if ((imgback.rows != CurrentFrame.rows) || (imgback.cols != CurrentFrame.cols))
	{
		if (imgback.rows > CurrentFrame.rows)
		{
			cv::resize(imgback, imgback, CurrentFrame.size(), 0, 0, cv::INTER_LINEAR);
		}
		else if (imgback.rows < CurrentFrame.rows)
		{
			cv::resize(CurrentFrame, CurrentFrame, imgback.size(), 0, 0, cv::INTER_LINEAR);
		}
	}

	cv::Mat image1_gary, image2_gary;
	if (imgback.channels() != 1)
	{
		cvtColor(imgback, image1_gary, cv::COLOR_BGR2GRAY);
	}
	if (CurrentFrame.channels() != 1)
	{
		cvtColor(CurrentFrame, image2_gary, cv::COLOR_BGR2GRAY);
	}

	cv::Mat frameDifference, absFrameDifferece;
	
	//图1减图2
	subtract(image1_gary, image2_gary, frameDifference, cv::Mat(), CV_16SC1);
	//subtract(image1_gary, image2_gary, frameDifference, cv::Mat(), CV_8UC1);
	//取绝对值
	absFrameDifferece = abs(frameDifference);

	//位深的改变
	absFrameDifferece.convertTo(absFrameDifferece, CV_8UC1, 1, 0);
	cv::imshow("absFrameDifferece", absFrameDifferece);
	cv::Mat segmentation;

	//阈值处理（这一步很关键，要调好二值化的值）
	threshold(absFrameDifferece, segmentation, 50, 255, cv::THRESH_BINARY);

	//中值滤波
	medianBlur(segmentation, segmentation, 5);

	//形态学处理(开闭运算)
	//形态学处理用到的算子
	cv::Mat morphologyKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
	morphologyEx(segmentation, segmentation, cv::MORPH_CLOSE, morphologyKernel, cv::Point(-1, -1), 2, cv::BORDER_REPLICATE);

	//显示二值化图片
	cv::imshow("segmentation", segmentation);


	//找边界
	std::vector< std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(segmentation, contours, hierarchy, 0, 2, cv::Point(0, 0));//CV_RETR_TREE
	std::vector< std::vector<cv::Point> > contours_poly(contours.size());

	
	// 需要在这里进行交并集去掉yolov5探测的目标

	std::vector<BoundingBox> v_temp;
	cv::Rect finalresult(0,0,0,0);
	char yichu[255];

	for (int index = 0; index < contours.size(); index++)
	{
		
		approxPolyDP(cv::Mat(contours[index]), contours_poly[index], 3, true);
		cv::Rect rect = cv::boundingRect(cv::Mat(contours_poly[index]));
		BoundingBox tempresults;
		tempresults.x = rect.x;
		tempresults.y = rect.y;
		tempresults.width = rect.width;
		tempresults.height = rect.height;


		int indexofmatch = highestIOU(tempresults, yolov5);
		// 判断是否运动物体与yolov5 结果粘连，如果是，则不计入最终追踪iou
		if (indexofmatch != -1 \
			&& intersectionOverUnion(tempresults, yolov5[indexofmatch]) >= 0.05)
		{
			tempresults.m_status = Ejected;
			/*rectangle(CurrentFrame,
				Point(tempresults.x, tempresults.y),
				Point(tempresults.x + tempresults.width,
					tempresults.y + tempresults.height),
				Scalar(0, 255, 0), 2, 8);*/

			sprintf(yichu, "Ejected");

			cv::putText(CurrentFrame, yichu,
				cv::Point((tempresults.x + tempresults.width - tempresults.width / 2) - 30,
					tempresults.y + tempresults.height + 10),
				1,
				1.2,
				Scalar(0, 255, 0),
				1.2, LINE_4);
			//v_temp.erase(v_temp.begin() + i);
		}
		else
		{
			tempresults.m_status = Suspected;
			finalresult.x = (int)tempresults.x;
			finalresult.y = (int)tempresults.y;
			finalresult.width = (int)tempresults.width;
			finalresult.height = (int)tempresults.height;
			if (finalresult.width > 10 && finalresult.height > 10)
			{
				
				m_left.push_back(finalresult);
				sprintf(yichu, "Suspected");
				/*rectangle(CurrentFrame,
					Point(tempresults.x, tempresults.y),
					Point(tempresults.x + tempresults.width,
						tempresults.y + tempresults.height),
					Scalar(0, 0, 255), 2, 8);*/
				cv::putText(CurrentFrame, yichu,
					cv::Point((tempresults.x + tempresults.width - tempresults.width / 2) - 35,
						tempresults.y + tempresults.height + 15),
					1,
					1.2,
					Scalar(0, 0, 255),
					2, LINE_4);
			}
			else
			{
				continue;
			}

		}
	}
}