
#include <iostream>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <memory>
#include <chrono>
#include <time.h>
//#include "detector.h"
//#include "cxxopts.hpp"
#include "UA-DETRAC.h"

#include <vector>
// iou relevant
#include "IOUT.h"

#include "../FrameDiff/FrameDiff.h"

#define yolov5 0
#define debug 0

#define RESIZE_WIDTH 960
#define RESIZE_HEIGHT 720
using namespace cv;
//using namespace std;


//Some constants for the algorithm
const double pi = 3.142;
const double cthr = 0.00001;
const double alpha = 0.002;
const double cT = 0.05;
const double covariance0 = 11.0;
const double cf = 0.1;
const double cfbar = 1.0 - cf;
const double temp_thr = 9.0 * covariance0 * covariance0;
const double prune = -alpha * cT;
const double alpha_bar = 1.0 - alpha;
//Temperory variable
int overall = 0;

//Structure used for saving various components for each pixel
struct gaussian
{
	double mean[3], covariance;
	double weight;								// Represents the measure to which a particular component defines the pixel value
	gaussian* Next;
	gaussian* Previous;
} *ptr, * start, * rear, * g_temp, * save, * next, * previous, * nptr, * temp_ptr;

struct MYNode
{
	gaussian* pixel_s;
	gaussian* pixel_r;
	int no_of_components;
	MYNode* Next;
} *N_ptr, * N_start, * N_rear;




struct Node1
{
	cv::Mat gauss;
	int no_of_comp;
	Node1* Next;
} *N1_ptr, * N1_start, * N1_rear;



//Some function associated with the structure management
 MYNode* Create_Node(double info1, double info2, double info3);
void Insert_End_Node(MYNode* np);
gaussian* Create_gaussian(double info1, double info2, double info3);

std::vector<std::string> LoadNames(const std::string& path) {
	// load class names
	std::vector<std::string> class_names;
	std::ifstream infile(path);
	if (infile.is_open()) {
		std::string line;
		while (getline(infile, line)) {
			class_names.emplace_back(line);
		}
		infile.close();
	}
	else {
		std::cerr << "Error loading the class names!\n";
	}

	return class_names;
}


void Demo(cv::Mat& img,
	const std::vector<std::tuple<cv::Rect, float, int>>& data_vec,
	const std::vector<std::string>& class_names,
	bool label = true) {
	for (const auto& data : data_vec) {
		cv::Rect box;
		float score;
		int class_idx;
		std::tie(box, score, class_idx) = data;

		cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

		if (label) {
			std::stringstream ss;
			ss << std::fixed << std::setprecision(2) << score;
			std::string s = class_names[class_idx] + " " + ss.str();

			auto font_face = cv::FONT_HERSHEY_DUPLEX;
			auto font_scale = 1.0;
			int thickness = 1;
			int baseline = 0;
			auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
			cv::rectangle(img,
				cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
				cv::Point(box.tl().x + s_size.width, box.tl().y),
				cv::Scalar(0, 0, 255), -1);
			cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
				font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
		}
	}
}


MYNode* Create_Node(double info1, double info2, double info3)
{
	N_ptr = new MYNode;
	if (N_ptr != NULL)
	{
		N_ptr->Next = NULL;
		N_ptr->no_of_components = 1;
		N_ptr->pixel_s = N_ptr->pixel_r = Create_gaussian(info1, info2, info3);
	}
	return N_ptr;
}

gaussian* Create_gaussian(double info1, double info2, double info3)
{
	ptr = new gaussian;
	if (ptr != NULL)
	{
		ptr->mean[0] = info1;
		ptr->mean[1] = info2;
		ptr->mean[2] = info3;
		ptr->covariance = covariance0;
		ptr->weight = alpha;
		ptr->Next = NULL;
		ptr->Previous = NULL;
	}
	return ptr;
}

void Insert_End_Node(MYNode* np)
{
	if (N_start != NULL)
	{
		N_rear->Next = np;
		N_rear = np;
	}
	else
		N_start = N_rear = np;
}

void Insert_End_gaussian(gaussian* nptr)
{
	if (start != NULL)
	{
		rear->Next = nptr;
		nptr->Previous = rear;
		rear = nptr;
	}
	else
		start = rear = nptr;
}

gaussian* Delete_gaussian(gaussian* nptr)
{
	previous = nptr->Previous;
	next = nptr->Next;
	if (start != NULL)
	{
		if (nptr == start && nptr == rear)
		{
			start = rear = NULL;
			delete nptr;
		}
		else if (nptr == start)
		{
			next->Previous = NULL;
			start = next;
			delete nptr;
			nptr = start;
		}
		else if (nptr == rear)
		{
			previous->Next = NULL;
			rear = previous;
			delete nptr;
			nptr = rear;
		}
		else
		{
			previous->Next = next;
			next->Previous = previous;
			delete nptr;
			nptr = next;
		}
	}
	else
	{
		std::cout << "Underflow........";
		//_getch();
		exit(0);
	}
	return nptr;
}

//CheckMode: 0代表去除黑区域，1代表去除白区域; NeihborMode：0代表4邻域，1代表8邻域;  
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //记录除去的个数  
	//记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		std::cout << "Mode: 去除小区域. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] < 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}
	else
	{
		std::cout << "Mode: 去除孔洞. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	std::vector<Point2i> NeihborPos;  //记录邻域点位置  
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
		std::cout << "Neighbor mode: 8邻域." << std::endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else std::cout << "Neighbor mode: 4邻域." << std::endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********  
				std::vector<cv::Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
				GrowBuffer.push_back(cv::Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  

				for (int z = 0; z < GrowBuffer.size(); z++)
				{

					for (int q = 0; q < NeihborCount; q++)                                      //检查四个邻域点  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows)  //防止越界  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查  
							}
						}
					}
				}
				if (GrowBuffer.size() > AreaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z < GrowBuffer.size(); z++)                         //更新Label记录  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********结束该点处的检查**********  


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}

	std::cout << RemoveCount << " objects removed." << std::endl;
}

//均值滤波
Mat myAverage(Mat& srcImage)
{
	
	Mat dstImage = Mat::zeros(srcImage.size(), srcImage.type());
	//Mat mask = Mat::ones(3, 3, srcImage.type());

	for (int k = 1; k < srcImage.rows - 1; k++)
	{
		for (int n = 1; n < srcImage.cols - 1; n++)
		{
			uchar f = 0;
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					f += srcImage.at<uchar>(k + i, n + j);

				}
			}
			dstImage.at<uchar>(k, n) = uchar(f / 9);
		}
	}
	return dstImage;
}

void removePepperNoise(Mat& mask)
{
	for (int y = 2; y < mask.rows - 2; ++y)
	{
		uchar* pThis = mask.ptr(y);
		uchar* pUp1 = mask.ptr(y - 1);
		uchar* pUp2 = mask.ptr(y - 2);
		uchar* pDown1 = mask.ptr(y + 1);
		uchar* pDown2 = mask.ptr(y + 2);

		pThis += 2; pUp1 += 2; pUp2 += 2; pDown1 += 2; pDown2 += 2;

		int x = 2;
		while (x < mask.cols - 2)
		{
			uchar v = *pThis;
			// 当前点为黑色
			if (v == 0)
			{
				// 5 * 5 邻域的外层
				bool allAbove = *(pUp2 - 2) && *(pUp2 - 1) && *(pUp2) && *(pUp2 + 1) && *(pUp2 + 2);
				bool allBelow = *(pDown2 - 2) && *(pDown2 - 1) && *(pDown2) && *(pDown2 + 1) && *(pDown2 + 2);
				bool allLeft = *(pUp1 - 2) && *(pThis - 2) && *(pDown1 - 2);
				bool allRight = *(pUp1 + 2) && *(pThis + 2) && *(pDown1 + 2);
				bool surroundings = allAbove && allBelow && allLeft && allRight;

				if (surroundings)
				{
					// 5*5 邻域的内层（3*3的小邻域）
					*(pUp1 - 1) = *(pUp1) = *(pUp1 + 1) = 255;
					*(pThis - 1) = *pThis = *(pThis + 1) = 255;
					*(pDown1 - 1) = *pDown1 = *(pDown1 + 1) = 255;
					//(*pThis) = ~(*pThis);
											// 0 ? 255
				}
				pUp2 += 2; pUp1 += 2; pThis += 2; pDown1 += 2; pDown2 += 2;
				x += 2;
			}
			++pThis; ++pUp2; ++pUp1; ++pDown1; ++pDown2; ++x;
		}
	}
}

#if 0
int main_01()
{
	// 开一个文件，将yolov5识别的结果写入
#if yolov5

	std::ofstream outfile("../results/yolov5.txt");

#else
	std::ifstream infile("../results/yolov5_out.txt");
	std::vector< std::vector<BoundingBox> > yolov5_detections;
	read_detections(infile, yolov5_detections);

#endif
	int i, j, k;
	i = j = k = 0;


	// Declare matrices to store original and resultant binary image
	cv::Mat orig_img, bin_img;
	
	//Declare a VideoCapture object to store incoming frame and initialize it
	cv::VideoCapture capture("../data/out.mp4");
	
	capture.read(orig_img);
	//orig_img = cv::imread("../data/back1.jpg");
	
	cv::resize(orig_img, orig_img, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), INTER_NEAREST);
	cv::cvtColor(orig_img, orig_img, cv::COLOR_BGR2YCrCb);
	//cv::GaussianBlur(orig_img, orig_img, cv::Size(3,3), 3.0);

	//Initializing the binary image with the same dimensions as original image
	bin_img = cv::Mat(orig_img.rows, orig_img.cols, CV_8U, cv::Scalar(0));

	double value[3];
	

	//Step 1: initializing with one gaussian for the first time and keeping the no. of models as 1
	cv::Vec3f val;
	uchar* r_ptr;
	uchar* b_ptr;
	for (i = 0; i < orig_img.rows; i++)
	{
		r_ptr = orig_img.ptr(i);
		for (j = 0; j < orig_img.cols; j++)
		{

			N_ptr = Create_Node(*r_ptr, *(r_ptr + 1), *(r_ptr + 2));
			if (N_ptr != NULL) {
				N_ptr->pixel_s->weight = 1.0;
				Insert_End_Node(N_ptr);
			}
			else
			{
				std::cout << "Memory limit reached... ";
				//_getch();
				exit(0);
			}
		}
	}



	int nL, nC;

	if (orig_img.isContinuous() == true)
	{
		nL = 1;
		nC = orig_img.rows * orig_img.cols * orig_img.channels();
	}

	else
	{
		nL = orig_img.rows;
		nC = orig_img.cols * orig_img.channels();
	}

	double del[3], mal_dist;
	double sum = 0.0;
	double sum1 = 0.0;
	int count = 0;
	bool close = false;
	int background;
	double mult;
	double duration, duration1, duration2, duration3;
	double temp_cov = 0.0;
	double weight = 0.0;
	double var = 0.0;
	double muR, muG, muB, dR, dG, dB, rVal, gVal, bVal;

	//Step 2: Modelling each pixel with Gaussian
	duration1 = static_cast<double>(cv::getTickCount());
	bin_img = cv::Mat(orig_img.rows, orig_img.cols, CV_8UC1, cv::Scalar(0));
	
	unsigned int count4tracker = 0;


	// 总帧率的探测结果
	std::vector< std::vector<BoundingBox>> vv_detections;
	

	// 总帧的追踪结果
	std::vector< Track > iou_tracks;
	float stationary_threshold = 0.90;		// low detection threshold,修改一下，这里改成大于这个的是静态物体
	float lazy_threshold = 0.70;
	float sigma_h = 0.7;		// high detection threshold,优选detection，检测的得分，其实这里面没有作用，不是通过classify的来的
	float sigma_iou = 0.2;	// IOU threshold
	float t_min = 3;		// minimum track length in frames

	std::cout<<"fps: "<<capture.get(cv::CAP_PROP_FPS)<<std::endl;
	
	// check if gpu flag is set
	bool is_gpu = false;

	// set device type - CPU/GPU

	torch::DeviceType device_type;
	if (torch::cuda::is_available() && is_gpu) {
		device_type = torch::kCUDA;
	}
	else {
		device_type = torch::kCPU;
	}

#if yolov5
	// 以下是运行深度网络yolov5
	// load class names from dataset for visualization
	std::vector<std::string> class_names = LoadNames("../weights/coco.names");
	if (class_names.empty()) {
		std::cout << "load className failed!" << std::endl;
		return -1;
	}
	else
	{
		std::cout << "load className success!" << std::endl;
	}
	// load network
	std::string weights = "../weights/yolov5s.torchscript.pt";
	auto detector = Detector(weights, device_type);

	// inference
	float conf_thres = 0.2f;
	float iou_thres = 0.5f;
	
#endif

	while (1)
	{
		duration3 = static_cast<double>(cv::getTickCount());
		std::vector<BoundingBox> v_bbnd;
		v_bbnd.clear();
		
		if (!capture.read(orig_img)) {
			break;
			capture.release();
			capture = cv::VideoCapture("../data/out.mp4");
			//capture = cv::VideoCapture("../data/test.avi");
			capture.read(orig_img);
		}
		else
		{
			cv::resize(orig_img, orig_img, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), INTER_NEAREST);
		}
		//break;
		int count = 0;
		int count1 = 0;


		N_ptr = N_start;
		duration = static_cast<double>(cv::getTickCount());
		for (i = 0; i < nL; i++)
		{
			r_ptr = orig_img.ptr(i);
			// 二值化的图的每个像素点的地址指针
			b_ptr = bin_img.ptr(i);

			for (j = 0; j < nC; j += 3)
			{
				sum = 0.0;
				sum1 = 0.0;
				close = false;
				background = 0;
				rVal = *(r_ptr++);
				gVal = *(r_ptr++);
				bVal = *(r_ptr++);
				start = N_ptr->pixel_s;
				rear = N_ptr->pixel_r;
				ptr = start;

				temp_ptr = NULL;

				if (N_ptr->no_of_components > 4)
				{
					Delete_gaussian(rear);
					N_ptr->no_of_components--;
				}

				for (k = 0; k < N_ptr->no_of_components; k++)
				{


					weight = ptr->weight;
					mult = alpha / weight;
					weight = weight * alpha_bar + prune;
					if (close == false)
					{
						muR = ptr->mean[0];
						muG = ptr->mean[1];
						muB = ptr->mean[2];

						dR = rVal - muR;
						dG = gVal - muG;
						dB = bVal - muB;

						/*del[0] = value[0]-ptr->mean[0];
						del[1] = value[1]-ptr->mean[1];
						del[2] = value[2]-ptr->mean[2];*/


						var = ptr->covariance;

						mal_dist = (dR * dR + dG * dG + dB * dB);

						if ((sum < cfbar) && (mal_dist < 16.0 * var * var))
							// 将背景高亮 
							background = 255;

						if (mal_dist < 9.0 * var * var)
						{
							weight += alpha;
							//mult = mult < 20.0*alpha ? mult : 20.0*alpha;

							close = true;

							ptr->mean[0] = muR + mult * dR;
							ptr->mean[1] = muG + mult * dG;
							ptr->mean[2] = muB + mult * dB;
							//if( mult < 20.0*alpha)
							//temp_cov = ptr->covariance*(1+mult*(mal_dist - 1));
							temp_cov = var + mult * (mal_dist - var);
							ptr->covariance = temp_cov < 5.0 ? 5.0 : (temp_cov > 20.0 ? 20.0 : temp_cov);
							temp_ptr = ptr;
						}

					}

					if (weight < -prune)
					{
						ptr = Delete_gaussian(ptr);
						weight = 0;
						N_ptr->no_of_components--;
					}
					else
					{
						//if(ptr->weight > 0)
						sum += weight;
						ptr->weight = weight;
					}

					ptr = ptr->Next;
				}



				if (close == false)
				{
					ptr = new gaussian;
					ptr->weight = alpha;
					ptr->mean[0] = rVal;
					ptr->mean[1] = gVal;
					ptr->mean[2] = bVal;
					ptr->covariance = covariance0;
					ptr->Next = NULL;
					ptr->Previous = NULL;
					//Insert_End_gaussian(ptr);
					if (start == NULL)
						// ??
						start = rear = NULL;
					else
					{
						ptr->Previous = rear;
						rear->Next = ptr;
						rear = ptr;
					}
					temp_ptr = ptr;
					N_ptr->no_of_components++;
				}

				ptr = start;
				while (ptr != NULL)
				{
					ptr->weight /= sum;
					ptr = ptr->Next;
				}

				while (temp_ptr != NULL && temp_ptr->Previous != NULL)
				{
					if (temp_ptr->weight <= temp_ptr->Previous->weight)
						break;
					else
					{
						//count++;
						next = temp_ptr->Next;
						previous = temp_ptr->Previous;
						if (start == previous)
							start = temp_ptr;
						previous->Next = next;
						temp_ptr->Previous = previous->Previous;
						temp_ptr->Next = previous;
						if (previous->Previous != NULL)
							previous->Previous->Next = temp_ptr;
						if (next != NULL)
							next->Previous = previous;
						else
							rear = previous;
						previous->Previous = temp_ptr;
					}

					temp_ptr = temp_ptr->Previous;
				}



				N_ptr->pixel_s = start;
				N_ptr->pixel_r = rear;

				//if(background == 1)
				//printf("current bin_image pixel's background %d \n", background);
				*b_ptr++ = background;
				//else
					//bin_img.at<uchar>(i,j) = 0;
				N_ptr = N_ptr->Next;
			}
		}


		// xuewei add some Morphology relevant processing
	
		//step one, filter tiny points
		//RemoveSmallRegion(bin_img, bin_img, 20, 0, 0);	
		
		// 有效验证中值滤波和闭操作性价比最高，也效果较好。
		// 中值滤波
		//cv::medianBlur(bin_img, bin_img, 3);

		// 闭操作呢
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
		cv::morphologyEx(bin_img, bin_img, CV_MOP_CLOSE, kernel);

		
		

		// 求得轮廓
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarcy;
		
		// 取反色，这样能下一步找到外轮廓
		cv::bitwise_not(bin_img, bin_img);


		// 再操作一把膨胀操作，将车内的空档连通起来,不要搞反了，膨胀就是对图像的高亮部分进行膨胀。
		cv::Mat dilatekernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::dilate(bin_img, bin_img, dilatekernel, Point(-1, -1), 1, 0);

		cv::findContours(bin_img, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		std::vector<RotatedRect> box(contours.size());
		std::vector<Rect> boundRect(contours.size());


		Point2f m_rect[4];

		
		
		
		BoundingBox m_BBtemp;
		memset(&m_BBtemp, 0, sizeof(BoundingBox));
		

		std::vector<BoundingBox> yolov5_currentobj;
		
#if yolov5
		// 先搞一步骤深度网络检测
		auto result = detector.Run(orig_img, conf_thres, iou_thres);


		// 将yolov5的结果写入txt
		bool ret = write2file(outfile, count4tracker, result);
		if (1) {
			Demo(orig_img, result, class_names,false);
		}
#else
		
		if (count4tracker< yolov5_detections.size())
		{
			yolov5_currentobj = yolov5_detections[count4tracker];
		}
		else
		{
			printf("it is not possible!, and yolov5 detections frames are less than videos!");
			return -1;
		}
#endif

		//  读取当前的深度网络探测结果
		std::vector<BoundingBox>::iterator iters_b = yolov5_currentobj.begin();
		std::vector<BoundingBox>::iterator iter_e = yolov5_currentobj.end();
		std::cout << "begin to draw yolov5 detections'results!!" << std::endl;
		for (; iters_b != iter_e; iters_b++)
		{
			rectangle(orig_img,
				Point((int)iters_b->x,
					(int)iters_b->y),
				Point((int)iters_b->x + (int)iters_b->width,
					(int)iters_b->y + (int)iters_b->height),
				Scalar(0, 0, 255),
				2,
				8);
		}
		
		// 以下是移动物体检测
		for (int i=0; i < contours.size();i++)
		{
			box[i] = minAreaRect(Mat(contours[i]));
			boundRect[i] = cv::boundingRect(Mat(contours[i]));
			if (box[i].size.width < 10 || box[i].size.height<10)
			{
				continue;
			}
			else
			{
				//rectangle(orig_img, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
				/*	m_BBtemp.x = boundRect[i].x;
					m_BBtemp.y = boundRect[i].y;
					m_BBtemp.w = boundRect[i].width;
					m_BBtemp.h = boundRect[i].height;
					m_BBtemp.score = 1;
					m_BBtemp.m_status = UnkownObj;
					v_bbnd.push_back(m_BBtemp);*/
				//circle(orig_img, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);
				box[i].points(m_rect);

				m_BBtemp.x = m_rect[0].x;
				m_BBtemp.y = m_rect[0].y;
				m_BBtemp.width = m_rect[1].x - m_rect[0].x;
				m_BBtemp.height = m_rect[2].y - m_rect[1].y;
				m_BBtemp.score = 1;
				m_BBtemp.m_status = UnkownObj;
				v_bbnd.push_back(m_BBtemp);


				// keep 最小外接矩形
				for (int j = 0; j < 4; j++)
				{
					//line(orig_img, m_rect[j], m_rect[(j + 1) % 4], Scalar(0, 255, 0), 2, 8);
				}
			}
		}

		// 做一把Yolo和背景法合并filter
		// step one ,先匹配遍历运动目标与yolo的探测

		char yichu[255];
		for (int i =0; i<v_bbnd.size();i++)
		{
			int indexofmatch = highestIOU(v_bbnd[i], yolov5_currentobj);
			// 判断是否运动物体与yolov5 结果粘连，如果是，则不计入最终追踪iou
			if (indexofmatch != -1 \
				&& intersectionOverUnion(v_bbnd[i], yolov5_currentobj[indexofmatch]) >= 0.05)
			{
				v_bbnd[i].m_status = Ejected;
				rectangle(orig_img,
					Point(v_bbnd[i].x, v_bbnd[i].y),
					Point(v_bbnd[i].x + v_bbnd[i].width,
						v_bbnd[i].y + v_bbnd[i].height),
					Scalar(0, 150, 50), 2, 8);

				sprintf(yichu, "Ejected");

				cv::putText(orig_img, yichu,
					cv::Point((v_bbnd[i].x + v_bbnd[i].width - v_bbnd[i].width / 2) - 30,
						v_bbnd[i].y + v_bbnd[i].height + 10),
					1,
					1.2,
					Scalar(0, 150, 50),
					1.2, LINE_4);

				v_bbnd.erase(v_bbnd.begin() + i);
			}
			else
			{
				v_bbnd[i].m_status = Suspected;
				rectangle(orig_img,
					Point(v_bbnd[i].x, v_bbnd[i].y),
					Point(v_bbnd[i].x + v_bbnd[i].width,
						v_bbnd[i].y + v_bbnd[i].height),
					Scalar(0, 150, 50), 2, 8);

				sprintf(yichu, "Suspected");

				cv::putText(orig_img, yichu,
					cv::Point((v_bbnd[i].x + v_bbnd[i].width - v_bbnd[i].width / 2) - 30,
						v_bbnd[i].y + v_bbnd[i].height + 10),
					1,
					1.2,
					Scalar(150, 0, 50),
					1.2, LINE_4);
			}
		}
		// 每帧循坏，将v_bbnd放入到嵌套vv中；
		vv_detections.push_back(v_bbnd);

		// 超过累计三个循坏开始 iou track

		if (count4tracker>3)
		{
			//begin to iou track
			iou_tracks = track_iou(stationary_threshold, lazy_threshold,sigma_h, sigma_iou, t_min, vv_detections);
			std::cout << "tracks'size" << iou_tracks.size() << std::endl;
			std::cout << "Last Track ID > " << iou_tracks.back().id << std::endl;
		}
		std::cout << "this is" << count4tracker << "frame" << std::endl;
		if (count4tracker==253)
		{
			cv::imwrite("../data/save1.jpg", orig_img);
		}

		

		char info[256];
		for (auto dt : iou_tracks)
		{
			int box_index = count4tracker - dt.start_frame;
			if (box_index < dt.boxes.size())
			{
				BoundingBox b = dt.boxes[box_index];
				cv::rectangle(orig_img, cv::Point(b.x, b.y), cv::Point(b.x + b.width, b.y + b.height), cv::Scalar(255, 0, 100), 2);

				std::string s_status;
				cv::Scalar blue(255, 0, 0);
				cv::Scalar red(0, 0, 255);
				cv::Scalar green(0, 255, 0);
				switch (dt.status)
				{
				case Moving:
					s_status = "Moving";
					sprintf_s(info, "ID:%d_AppearingT:%d_%s", dt.id, dt.total_appearing, s_status.c_str());
					//cv::putText(orig_img, info, cv::Point((b.x + b.w - b.w / 2) - 30, b.y + b.h - 5), 1, 1, blue, 1);
					break;
				case Stopping:
					s_status = "Stopping";
					sprintf_s(info, "ID:%d_AppearingT:%d_%s", dt.id, dt.total_appearing, s_status.c_str());
					//cv::putText(orig_img, info, cv::Point((b.x + b.w - b.w / 2) - 30, b.y + b.h - 5), 1, 1, green, 1);
					break;
				case Splittingobj:
					s_status = "Splittingobj";
					sprintf_s(info, "ID:%d_AppearingT:%d_%s", dt.id, dt.total_appearing, s_status.c_str());
					cv::putText(orig_img, info, cv::Point((b.x + b.width - b.width / 2) - 30, b.y + b.height - 5), 1, 1, red, 1);
					break;
				}
				
			}
			
		}

		count4tracker++;
		duration = static_cast<double>(cv::getTickCount()) - duration3;
		duration /= cv::getTickFrequency();
		
		std::cout << "\n per frame duration :" << duration;
		std::cout << "\n counts : " << count;
		cv::namedWindow("orig", CV_WINDOW_NORMAL);
		//cv::namedWindow("gp", CV_WINDOW_NORMAL);
		cv::imshow("orig", orig_img);
		//cv::imshow("gp", bin_img);
	
		
		cv::waitKey(5);
	}

	

#if yolov5
	outfile.close();
#endif

	system("PAUSE");
	return 0;
	//_getch();
}
#endif

int main()
{
	cv::Mat background = imread("../data/back.jpg");
	cv::Mat object = imread("../data/object.jpg");

	// 准备事先准备的探测结果

	std::ifstream infile("../results/yolov5_out.txt");
	std::vector< std::vector<BoundingBox> > yolov5_detections;
	read_detections(infile, yolov5_detections);


	cv::resize(background, background, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), INTER_NEAREST);
	// 这里使用copyto 速率更快，比clone好
	cv::Mat drawimg(RESIZE_HEIGHT, RESIZE_WIDTH, CV_8UC3);
	background.copyTo(drawimg);

	
	// 画出roi区域,960x720的区域，等比例变成 320x300
	double factorx = (double)RESIZE_WIDTH/960.0;
	double factory = (double)RESIZE_HEIGHT/720.0;
	
	

	cv::line(drawimg, cv::Point(424*factorx, 264*factory), cv::Point(524 * factorx, 264 * factory), cv::Scalar(0, 255, 0), 2, 0);
	cv::line(drawimg, cv::Point(424 * factorx, 264 * factory), cv::Point(331 * factorx, 474 * factory), cv::Scalar(0, 255, 0), 2, 0);
	cv::line(drawimg, cv::Point(524 * factorx, 264 * factory), cv::Point(764 * factorx, 474 * factory), cv::Scalar(0, 255, 0), 2, 0);
	cv::line(drawimg, cv::Point(331 * factorx, 474 * factory), cv::Point(2 * factorx, 474 * factory), cv::Scalar(0, 255, 0), 2, 0);
	cv::line(drawimg, cv::Point(764 * factorx, 474 * factory), cv::Point(958 * factorx, 474 * factory), cv::Scalar(0, 255, 0), 2, 0);
	cv::line(drawimg, cv::Point(2 * factorx, 474 * factory), cv::Point(2 * factorx, 718 * factory), cv::Scalar(0, 255, 0), 2, 0);
	cv::line(drawimg, cv::Point(958 * factorx, 474 * factory), cv::Point(958 * factorx, 718 * factory), cv::Scalar(0, 255, 0), 2, 0);
	cv::line(drawimg, cv::Point(2 * factorx, 718 * factory), cv::Point(958 * factorx, 718 * factory), cv::Scalar(0, 255, 0), 2, 0);

	// 设置单通道的掩码

	double duration, duration1,duration2;

	Mat mask = cv::Mat::zeros(drawimg.size(), CV_8UC1);

	Point p1(424*factorx, 264*factory); 
	Point p2(524*factorx, 264*factory);
	Point p8(331*factorx, 474 * factory); 
	Point p3(764 * factorx, 474 * factory);
	Point p7(2 * factorx, 474 * factory);  
	Point p4(958 * factorx, 474 * factory);
	Point p6(2 * factorx, 718 * factory); 
	Point p5(958 * factorx, 718 * factory);
	std::vector<Point> contour;
	contour.push_back(p1);
	contour.push_back(p2);
	contour.push_back(p3);
	contour.push_back(p4);
	contour.push_back(p5);
	contour.push_back(p6);
	contour.push_back(p7);
	contour.push_back(p8);




	std::vector<std::vector<Point> > contours;
	contours.push_back(contour);
	cv::drawContours(mask, contours, -1, cv::Scalar::all(255), CV_FILLED);
	cv::Mat backgroundroi(RESIZE_HEIGHT, RESIZE_WIDTH, CV_8UC1);
	
	// copyto 只花费 0.0003s,可以使用
	background.copyTo(backgroundroi, mask);
	
	cv::VideoCapture capture("../data/out.mp4");


	// Anomaly 类初始化
	
	Anomaly m_test(backgroundroi);

	//cv::Point p11(495, 404);

	std::vector<std::vector<cv::Rect>> cadidatesall(10);
	

	float stationary_threshold = 0.6;		// low detection threshold,修改一下，这里改成大于这个的是静态物体
	float vanish_threshold = 0.2;
	float t_min = 3;

	int framenum = 0;
	while (1)
	{
		std::vector< LeftObjects > small_track;
		small_track.clear();
		duration = static_cast<double>(cv::getTickCount());
		std::vector<BoundingBox> yolov5obj;
		yolov5obj.clear();

		std::vector<cv::Rect> cadidates;
		cadidates.clear();
		if (framenum < yolov5_detections.size())
		{
			yolov5obj = yolov5_detections[framenum];
		}
		else
		{
			printf("it is not possible!, and yolov5 detections frames are less than videos!");
			return -1;
		}


		cv::Mat frames;
		cv::Mat frameroi(RESIZE_HEIGHT, RESIZE_WIDTH, CV_8UC1);
		cv::Mat cleanframe(RESIZE_HEIGHT, RESIZE_WIDTH, CV_8UC1);

		// 存储可能的目标，单帧
		
		if (!capture.read(frames)) {
			break;
			capture.release();
			capture = cv::VideoCapture("../data/out.mp4");
			//capture = cv::VideoCapture("../data/test.avi");
			capture.read(frames);
		}
		else
		{
			cv::resize(frames, frames, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), INTER_NEAREST);
		}
		/*duration2 = static_cast<double>(cv::getTickCount()) - duration;
		duration2 /= cv::getTickFrequency();
		std::cout << "\n capture and resize per frame takes \t :" << duration2 << "\t" << "s" << std::endl;*/
		// 每一帧做一次roi 提取
		frames.copyTo(frameroi, mask);
		frames.copyTo(cleanframe, mask);

		double time1 = static_cast<double>(cv::getTickCount());
		duration1 = static_cast<double>(cv::getTickCount())-duration;
		double consumption  = duration1/ cv::getTickFrequency();

		std::cout << "time consumption duration1:\t" << consumption << std::endl;

		// 搞出来yolov5的探测结果, 注意： frameroi只是用来描画出来看的
		std::vector<BoundingBox>::iterator iters_b = yolov5obj.begin();
		std::vector<BoundingBox>::iterator iter_e = yolov5obj.end();
		std::vector<cv::Point> yolov5Points;
		char insidezifu[256];
		bool updateback = true;
		int iternum = 0;
		for (; iters_b != iter_e; iters_b++)
		{
			yolov5Points.clear();
			iternum++;
			cv::Point zuoshang, youxia;
			zuoshang.x = (int)iters_b->x;
			zuoshang.y = (int)iters_b->y;
			youxia.x = (int)iters_b->x + (int)iters_b->width;
			youxia.y = (int)iters_b->y + (int)iters_b->height;

			cv::Point topleft, topright, bottomleft, bottomright;
			topleft.x = (int)iters_b->x;
			topleft.y = (int)iters_b->y;
			topright.x = (int)iters_b->x + (int)iters_b->width;
			topright.y = (int)iters_b->y;
			bottomleft.x = (int)iters_b->x;
			bottomleft.y = (int)iters_b->y + (int)iters_b->height;
			bottomright.x = (int)iters_b->x + (int)iters_b->width;
			bottomright.y = (int)iters_b->y + (int)iters_b->height;

			yolov5Points.push_back(topleft);
			yolov5Points.push_back(topright);
			yolov5Points.push_back(bottomright);
			yolov5Points.push_back(bottomleft);

			double ticks = static_cast<double>(cv::getTickCount());
			bool insideornot = m_test.PointsinRegion(yolov5Points, contour);
			double ticks1 = static_cast<double>(cv::getTickCount()) - ticks;
			ticks1 /= cv::getTickFrequency();
			//	std::cout << "per Pointregion takes:\t" << ticks1 << std::endl;
			sprintf(insidezifu, "%s", insideornot ? "In" : "Out");
			if (insideornot)
			{

#if debug
				rectangle(frameroi,
					Point((int)iters_b->x,
						(int)iters_b->y),
					Point((int)iters_b->x + (int)iters_b->w,
						(int)iters_b->y + (int)iters_b->h),
					Scalar(0, 0, 255),
					2,
					8);
				cv::putText(frameroi,
					insidezifu,
					cv::Point(zuoshang.x - 20, zuoshang.y - 12),
					1,
					1.5,
					cv::Scalar(0, 0, 255),
					2);
#endif		
				updateback = false;
			}
			else
			{

#if debug
				rectangle(frameroi,
					Point((int)iters_b->x,
						(int)iters_b->y),
					Point((int)iters_b->x + (int)iters_b->w,
						(int)iters_b->y + (int)iters_b->h),
					Scalar(255, 0, 0),
					2,
					8);
				cv::putText(frameroi,
					insidezifu,
					cv::Point(zuoshang.x - 20, zuoshang.y - 12),
					1,
					1.5,
					cv::Scalar(255, 0, 0),
					2);
#endif
				
			}

		}
		double time2 = static_cast<double>(cv::getTickCount());
		duration2 = static_cast<double>(cv::getTickCount()) - time1;
		double consumption1 = duration2 / cv::getTickFrequency();
		std::cout << "time consumption duration2:\t" << consumption1 << std::endl;


		// 这个是debug使用的，正式版本要去除
#if debug
		imshow("Debugframe", frameroi);
		waitKey(3);
#endif
		// 注意： frameroi只是用来描画出来看的
		m_test.UpdateBack(cleanframe, updateback);
		// 做帧差显示,注意： frameroi只是用来描画出来看的

		double zhenctime = static_cast<double>(cv::getTickCount());

		if (framenum%25==0)
		{
			m_test.FindDiff(cleanframe, yolov5obj, cadidates);
			if (!cadidates.empty())
			{
				cadidatesall.push_back(cadidates);
			}
			else
			{
				cadidatesall.clear();
			}
		}

		// 保留最新的8个，暂定
		if (cadidatesall.size()>12)
		{
			auto iter = cadidatesall.erase(cadidatesall.begin(), cadidatesall.end() - 8);
		}

		small_track = smalltrack(stationary_threshold, vanish_threshold, t_min, cadidatesall);

		char info[256];
		for (auto dt : small_track)
		{
				
				cv::Rect b = dt.m_box.back();
				
				std::string s_status;
				cv::Scalar blue(255, 0, 0);
				cv::Scalar red(0, 0, 255);
				cv::Scalar green(0, 255, 0);
				switch (dt.status)
				{
				case Suspected:
					s_status = "Suspected";
					cv::rectangle(cleanframe, cv::Point(b.x, b.y), cv::Point(b.x + b.width, b.y + b.height), blue, 2);
					sprintf(info, "ID:%d_Ap:%d_%s", dt.m_ID, dt.count, s_status.c_str());
					cv::putText(cleanframe, info, cv::Point((b.x + b.width - b.width / 2) - 30, b.y + b.height - 5), 1, 1, blue, 1);
					break;
				case Splittingobj:
					s_status = "Splittingobj";
					cv::rectangle(cleanframe, cv::Point(b.x, b.y), cv::Point(b.x + b.width, b.y + b.height), red, 2);
					sprintf(info, "ID:%d_Ap:%d_%s", dt.m_ID, dt.count, s_status.c_str());
					cv::putText(cleanframe, info, cv::Point((b.x + b.width - b.width / 2) - 30, b.y + b.height - 5), 1, 1, red, 1);
					break;
				}
		}



		double zhenctime1 = static_cast<double>(cv::getTickCount()) - zhenctime;
		double zhenctimeconsumption = zhenctime1 / cv::getTickFrequency();
		std::cout << "zhencha time consumption:\t" << zhenctimeconsumption << std::endl;


		duration2 = static_cast<double>(cv::getTickCount()) - time2;
		double consumption2 = duration2 / cv::getTickFrequency();

		std::cout << "time consumption duration3:\t" << consumption2 << std::endl;

		duration1 = static_cast<double>(cv::getTickCount()) - duration;
		duration1 /= cv::getTickFrequency();
		std::cout << "\n process per frame takes \t :" << duration1<<"\t"<<"s"<<std::endl;
		//cv::namedWindow("效果图", 0);
		cv::imshow("效果图", cleanframe);
		cv::waitKey(1);
		framenum++;

	}


	/*imshow("backgroundroi", backgroundroi);
	cv::imwrite("../data/back_960_720.jpg", background);
	waitKey(0);*/
	system("pause");
	return 0;
}
