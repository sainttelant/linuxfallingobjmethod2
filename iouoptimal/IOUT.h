/******************************************************************************
* C++ Implementation of IOUT tracking algorithm
* More info on <http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf>
* Author Lucas Wals
******************************************************************************/
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/******************************************************************************
* STRUCTS
******************************************************************************/

enum ObjectStatus
{
	Suspected,
	SplittingObj,
	Ejected,
	UnkownObj
};
struct BoundingBox
{
	// x-component of top left coordinate
	float x;
	// y-component of top left coordinate
	float y;
	// width of the box
	float width;
	// height of the box
	float height;
	// score of the box;
	float score;
	ObjectStatus m_status;
};


enum TrackStatus
{
	Moving,
	Splittingobj,
	Stopping,
	Unkown,
	
};

struct Track
{
	std::vector<BoundingBox> boxes;
	float max_score;
	int start_frame;
	int id;
	int total_appearing;
	int stationary_count;
	TrackStatus status;
	bool first_stationary;
	float areasize;
};


struct LeftObjects
{
	std::vector<cv::Rect> m_box;
	unsigned int m_ID;
	ObjectStatus status;
	unsigned int count;
	bool first_detect;
};



// Return the IoU between two boxes
template <typename T>
inline float intersectionOverUnion(T box1, T box2);
// Returns the index of the bounding box with the highest IoU

template <typename T>
inline int highestIOU(T box, std::vector<T> boxes);
// Starts IOUT tracker
std::vector< Track > track_iou(float status_threshold,float lazy_threshold, float sigma_h, float sigma_iou, float t_min,
	std::vector< std::vector<BoundingBox> > &detections);


std::vector<LeftObjects> smalltrack(float status_threshold,
	float degrade_threshold,
	float t_min,
	std::vector< std::vector<cv::Rect>>& detections);

// Give an ID to the result tracks from "track_iou"
// Method useful the way IOU is implemented in Python
//void enumerate_tracks();