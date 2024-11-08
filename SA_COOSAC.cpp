#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include "Calculate.h"
#include "COOSAC.h"

using namespace cv;
using namespace std;

void filter_compact(const unordered_set<int>& filter_idx, vector<Point2f>& init_src, vector<Point2f>& init_tar, vector<bool>& compact_mask) {
	int n = init_src.size();
	for (const auto& idx : filter_idx) {
		if (idx >= 0 && idx < n) {
			compact_mask[idx] = true;
		}
	}
}

vector<double> filter(const vector<int>& filter_idx, const vector<double>& inputs) {
	int n = filter_idx.size();
	vector<double> filter_result;
	filter_result.reserve(n);

	for (int i = 0; i < n; i++) {
		if (filter_idx[i] >= 0 && filter_idx[i] < inputs.size()) {
			filter_result.push_back(inputs[filter_idx[i]]);
		}
		else {
			filter_result.push_back(0.0);
		}
	}
	return filter_result;
}

unordered_set<int> OFTL(vector<double>& angle, vector<double>& length, vector<int>& ori_bin_idx, vector<int>& len_bin_idx, int& ori_bin_num,
	int& len_bin_num, int& angel_peak, int& length_peak, vector<int>& weight) {
	vector<double> first_infos, second_infos;
	double first_pitch, second_pitch;
	// angle first
	first_infos = angle;
	second_infos = length;
	first_pitch = angle_pitch;
	second_pitch = len_pitch;

	//	filiter first correspondences by histograms
	vector<double> range_first;
	vector<int> firstfilter_idx;
	vector<double> filter_first;
	int fre_first;
	first_pitch = ScottBinWidth(first_infos);
	fre_first = cal_pitch_frequency(first_infos, first_pitch, range_first);
	ori_bin_num = range_first.size() - 1;
	firstfilter_idx = save_interval(first_infos, range_first, fre_first, ori_bin_idx);
	filter_first = filter(firstfilter_idx, second_infos);
	angel_peak = fre_first;

	//	filiter second correspondences by histograms
	vector<double> range_second;
	vector<int> secondfilter_idx;
	vector<double> filter_second;
	int fre_second;
	second_pitch = ScottBinWidth(filter_first);
	fre_second = cal_pitch_frequency(filter_first, second_pitch, range_second);
	len_bin_num = range_second.size() - 1;
	vector<int> compact_set;
	unordered_set<int> compact_idx;
	secondfilter_idx = save_interval_second(filter_first, range_second, fre_second, len_bin_idx, firstfilter_idx, compact_set, compact_idx, length);
	length_peak = fre_second;

	cal_diff(filter_first, range_second, firstfilter_idx, compact_set, len_bin_idx);

	return compact_idx;
}

unordered_set<int> GH_filter(vector<Point2f>& src_pts, vector<Point2f>& tar_pts, vector<int>& ori_bin_idx, vector<int>& len_bin_idx, int& ori_bin_num,
	int& len_bin_num, int& angel_peak, int& length_peak, vector<int>& weight) {
	// Vi
	vector<Point2f> vec(src_pts.size());
	for (int i = 0; i < src_pts.size(); i++) {
		vec[i] = Point2f((tar_pts[i].x + 1080 - src_pts[i].x), (tar_pts[i].y - src_pts[i].y));
	}

	// length of Vi
	vector<double> vector_len(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double length = sqrt(pow(vec[i].x, 2) + pow(vec[i].y, 2));
		vector_len[i] = length;
	}
	// angle of Vi
	vector<double> vector_ang(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double theta = atan2(vec[i].y, vec[i].x);
		//theta = abs(theta * 180.0 / CV_PI);
		theta = theta * 180.0 / CV_PI; // effect better
		vector_ang[i] = theta;
	}
	unordered_set<int> compact = OFTL(vector_ang, vector_len, ori_bin_idx, len_bin_idx, ori_bin_num, len_bin_num, angel_peak, length_peak, weight);

	return compact;
}

vector<double> SA_COOSAC(vector<Point2f>& src_pts, vector<Point2f>& tar_pts, vector<int>& ground_truth, int repeat_time, Mat& sourceImg, Mat& targetImg)
{
	double default_Threshold = 4.5;
	double extractRate = 0.4;
	double recall_all = 0, precision_all = 0, f1_score_all = 0, time_all = 0, rmse_all = 0;

	for (int times = 0; times < repeat_time; times++) {
		vector<int> ori_bin_idx(src_pts.size(), 0);
		vector<int> len_bin_idx(src_pts.size(), INFINITY);
		vector<int> weight(src_pts.size(), 0);
		int angel_peak, length_peak, ori_bin_num = 0, len_bin_num = 0;
		clock_t start, stop;

		/*--------- Start counts ---------*/
		start = clock();

		// Extract OFTL histogram
		unordered_set<int> compact_idx = GH_filter(src_pts, tar_pts, ori_bin_idx, len_bin_idx, ori_bin_num, len_bin_num, angel_peak, length_peak, weight);

		double compact_rate = (double)compact_idx.size() / src_pts.size();
		double epsilon = 0.8 * compact_rate + 0.2 * (1 / (1 + exp(-20 * (compact_rate - 0.5))));
		double init_Threshold = default_Threshold + epsilon * default_Threshold;

		vector<bool> inliers_outliers_mask(src_pts.size(), false);
		filter_compact(compact_idx, src_pts, tar_pts, inliers_outliers_mask);

		pair<int, int> high_idx = { angel_peak , length_peak };

		// COOSAC main function
		homoinfo H_info = COOSAC(src_pts, tar_pts, ground_truth, inliers_outliers_mask, compact_idx, init_Threshold, extractRate, ori_bin_idx,
			len_bin_idx, ori_bin_num, len_bin_num, high_idx, compact_rate, weight, epsilon);
		stop = clock();
		/*--------- End counts ---------*/
		double total_time = ((double)stop - (double)start) / 1000;

		vector<int> inlier_mask = H_info.inliers;

		confusion classification = confusion_matrix(ground_truth, inlier_mask);
		double TP = classification.TP; double TN = classification.TN;
		double FP = classification.FP; double FN = classification.FN;
		double recall = TP / (TP + FN);
		double precision = TP / (TP + FP);
		double specificity = TN / (TN + FP);
		double f1_score;
		if (recall + precision == 0.0)
			f1_score = 0;
		else
			f1_score = 2 * precision * recall / (precision + recall);

		cal_RMSE cal_rmse = calculateErrorMetrics(ground_truth, inlier_mask, src_pts, tar_pts, H_info.H);
		
		recall_all += recall;
		precision_all += precision;
		f1_score_all += f1_score;
		time_all += total_time;
		rmse_all += cal_rmse.RMSE;

		cout << "[Iteration Result]\n";
		cout << "F1-score is " << f1_score << " and Specificity is " << specificity << endl;
		cout << "Recall is " << recall << " and Precision is " << precision << endl;
		cout << "RMSE: " << cal_rmse.RMSE << endl;
		cout << "Time: " << total_time << endl << endl;
	}

	recall_all /= repeat_time;
	precision_all /= repeat_time;
	f1_score_all /= repeat_time;
	time_all /= repeat_time;
	rmse_all /= repeat_time;

	cout << "[Final Result]\n";
	cout << "F1-score is " << f1_score_all << endl;
	cout << "Recall is " << recall_all << " and Precision is " << precision_all << endl;
	cout << "RMSE: " << rmse_all << endl;
	cout << "Time: " << time_all << endl << endl;

	vector<double> result;

	result.push_back(recall_all);
	result.push_back(precision_all);
	result.push_back(f1_score_all);
	result.push_back(time_all);
	result.push_back(rmse_all);

	return result;
}