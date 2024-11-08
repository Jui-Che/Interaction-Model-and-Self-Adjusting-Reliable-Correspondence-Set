#pragma once
#ifndef CALCULATE_H
#define CALCULATE_H

#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct confusion {
	int TP = 0, TN = 0, FP = 0, FN = 0;
};

struct cal_RMSE {
	double RMSE = 0, MAE = 0, MEE = 0;
};

static int len_pitch = 20; //180
static int angle_pitch = 5;
static int area_thres = 1000;


double ScottBinWidth(vector<double>& data);

int cal_pitch_frequency(vector<double>& input_info, const double pitch_dist, vector<double>& pitch_list);

vector<int> save_interval(vector<double>& inputs_info, vector<double>& pitch_list, int frequency, vector<int>& len_bin_idx);

vector<int> save_interval_second(vector<double>& inputs_info, vector<double>& pitch_list, int frequency, vector<int>& len_bin_idx,
	vector<int>& firstfilter_idx, vector<int>& compact_set, unordered_set<int>& compact_idx, vector<double>& length);

void cal_diff(vector<double>& input, vector<double>& pitch_list, vector<int>& save_first, vector<int>& save_second, vector<int>& bin_idx);

double calculateArea(const Point2f& ori_pt0, const Point2f& ori_pt1, const Point2f& tar_pt0, const Point2f& tar_pt1);

void computeError(const vector<Point2f>& src_pts, const vector<Point2f>& tar_pts, const Mat& H, vector<double>& residuals_error);

Mat OptimizeHomographyLM(const vector<Point2f>& src, const vector<Point2f>& dst, const Mat& initialH, const vector<int>& weights);

void self_adjusting(vector<double>& err_list, double& projErr, double& inlier_count, vector<int>& inlier_mask, vector<int>& weight, vector<bool>& inliers_outliers_mask,
	unordered_set<int>& compact_idx, vector<int>& ori_bin_idx, vector<int>& len_bin_idx, int& ori_bin_num,
	int& len_bin_num, pair<int, int> peak, int QR_times, int min_reduce_size, double& global_error);

Mat OPT_H(vector<Point2f>& src, vector<Point2f>& tar, vector<int>& weight);

confusion confusion_matrix(vector<int>& GT, vector<int>& Prediction);

cal_RMSE calculateErrorMetrics(const vector<int>& ground_truth, const vector<int>& inlier_mask, const vector<Point2f>& src_pts, const vector<Point2f>& tar_pts, const Mat& H);

#endif // !CALCULATE_H