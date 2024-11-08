#pragma once
#ifndef SA_COOSAC_H
#define SA_COOSAC_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<double> SA_COOSAC(vector<Point2f>& source_match_pt, vector<Point2f>& target_match_pt, vector<int>& ground_truth, int repeat_time, Mat& sourceImg, Mat& targetImg);

#endif // !SA_COOSAC_H