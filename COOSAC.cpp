#include <random>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "COOSAC.h"
#include "Calculate.h"

using namespace std;
using namespace cv;

int maxIters = 200000;
int tiny_iteration = 0;
double maxConfidence = 0.995;
double inlierThreshHold = 4.5;
double best_tiny_inlier_ratio, tiny_confidence;
vector<int> best_inliers = vector<int>();
Mat final_H = Mat::eye(3, 3, CV_64F);
Mat best_tiny_H = Mat();
RNG rng((unsigned)time(NULL));

double cal_confidence(double inlier_ratio, int iteration) {
	return  1 - pow((1 - pow(inlier_ratio, 4)), iteration);
}

void findInliers(vector<double>& err, double projErr, double& inlier_count, vector<int>& inlier_mask, double& error) {
	double threshold = projErr * projErr;

	for (int i = 0; i < err.size(); i++) {
		double f = err[i] <= threshold;
		inlier_mask[i] = f;
		inlier_count += f;
		if(f)
			error += err[i];
	}
	error /= inlier_count;
}

bool check_iter_jump(vector<int>& tiny_indices, vector<Point2f>& pick_src, vector<Point2f>& pick_tar) {
	//CC nonsense method
	int size = tiny_indices.size();
	vector<double> v_res(size);
	computeError(pick_src, pick_tar, final_H, v_res);
	if (norm(v_res) < 1)
		return true;
	return false;
}

void randomPickIndex(int size, int pick_size, vector<int>& selected_indices) {
	for (int i = 0; i < pick_size; i++) {
		int idx_i;
		for (idx_i = rng.uniform(0, size); find(selected_indices.begin(), selected_indices.end(), idx_i) != selected_indices.end(); idx_i = rng.uniform(0, size)) {}
		selected_indices[i] = idx_i;
	}
}

void pick_TinyCorresponence(const vector<Point2f>& init_src_pts, const vector<Point2f>& init_tar_pts, vector<Point2f>& tiny_src, vector<Point2f>& tiny_tar,
	vector<int>& tiny_indices, unordered_set<int>& compact_idx, size_t smallWithdraw_size, vector<int>& weight, vector<int>& tiny_weight) {

	vector<int> selected_indices(smallWithdraw_size);
	//vector<int> compact_idx_vec(compact_idx.begin(), compact_idx.end());
	randomPickIndex(compact_idx.size(), smallWithdraw_size, selected_indices);

	auto it = compact_idx.begin();
	for (int idx : selected_indices) {
		advance(it, idx);
		int corr_idx = *it;
		tiny_src.push_back(init_src_pts[corr_idx]);
		tiny_tar.push_back(init_tar_pts[corr_idx]);
		tiny_indices.push_back(corr_idx);
		tiny_weight.push_back(weight[corr_idx]);
		it = compact_idx.begin(); // Reset the iterator for the next advance
	}
}

bool checkAllAreas(const vector<Point2f>& tiny_src, const vector<Point2f>& tiny_tar, vector<int>& indices,double area_threshold) {
	for (const auto& pair : COMBINATIONS) {
		int idx0 = indices[pair[0]];
		int idx1 = indices[pair[1]];
		double area = 0;

		area = calculateArea(tiny_src[idx0], tiny_src[idx1], tiny_tar[idx0], tiny_tar[idx1]);

		if (area < area_threshold) {
			return false;
		}
	}
	return true;
}

void pick_FourCorrespondence(const vector<Point2f>& tiny_src, const vector<Point2f>& tiny_tar, vector<int>& tiny_weight, vector<Point2f>& pick_src, vector<Point2f>& pick_tar, vector<int>& pick_weight) {
	int pick_size = 4, attempt_threshold = 1000, area_threshold = 1000, attempts = 0;

	while (attempts < attempt_threshold) {
		vector<int> indices(pick_size);
		randomPickIndex(tiny_src.size(), pick_size, indices);
		bool valid = true;

		//valid = checkAllAreas(tiny_src, tiny_tar, indices, area_threshold);

		if (valid) {
			for (int i = 0; i < pick_size; ++i) {
				pick_src.push_back(tiny_src[indices[i]]);
				pick_tar.push_back(tiny_tar[indices[i]]);
				pick_weight.push_back(tiny_weight[indices[i]]);
			}
			break;
		}
		attempts++;
		if (attempts >= attempt_threshold) {
			for (int i = 0; i < pick_size; ++i) {
				pick_src.push_back(tiny_src[indices[i]]);
				pick_tar.push_back(tiny_tar[indices[i]]);
				pick_weight.push_back(tiny_weight[indices[i]]);
			}
		}
	}
}

bool tinyCOOSAC(vector<Point2f>& tiny_src, vector<Point2f>& tiny_tar, vector<int>& tiny_indices, pair<int, int> high_idx, vector<int>& tiny_weight,	double sigmoid, int QR_times) {
	// Initialize the local parameters
	best_tiny_inlier_ratio = 0;
	tiny_confidence = 0;
	int tiny_size = tiny_src.size();
	vector<double> residuals_error(tiny_size);
	vector<int> mask(tiny_size);
	double best_tiny_error = INT_MAX;
	Mat tiny_H(3, 3, CV_64F);

	while (tiny_confidence < maxConfidence && tiny_iteration < maxIters) {
		double inlier_count = 0, tiny_error = 0.0;
		vector<Point2f> pick_src, pick_tar;
		vector<int> pick_weight;

		pick_FourCorrespondence(tiny_src, tiny_tar, tiny_weight, pick_src, pick_tar, pick_weight);

		if (QR_times != 0 && check_iter_jump( tiny_indices, pick_src, pick_tar)) {
			return true;
		}

		tiny_H = HomographyFourPointSolver::solve(pick_src, pick_tar, pick_weight);
	
		computeError(tiny_src, tiny_tar, tiny_H, residuals_error);
		findInliers(residuals_error, inlierThreshHold, inlier_count, mask, tiny_error);

		double inlier_ratio = inlier_count / tiny_size;
		if (inlier_ratio > best_tiny_inlier_ratio ) {
			best_tiny_inlier_ratio = inlier_ratio;
			tiny_H.copyTo(best_tiny_H);
			best_tiny_error = tiny_error;
		}

		tiny_confidence = cal_confidence(best_tiny_inlier_ratio, tiny_iteration);
		tiny_iteration++;
	}

	if (sigmoid > 0.75)
		best_tiny_H = OptimizeHomographyLM(tiny_src, tiny_tar, best_tiny_H, tiny_weight);

	return false;
}

homoinfo COOSAC(vector<Point2f>& init_src_pts, vector<Point2f>& init_tar_pts, vector<int>& ground_truth, vector<bool>& inliers_outliers_mask,
	unordered_set<int>& compact_idx, double inlierThresh, double extractRate, vector<int>& ori_bin_idx, vector<int>& len_bin_idx, int& ori_bin_num, int& len_bin_num,
	pair<int, int> high_idx, double compact_rate, vector<int>& weight, double sigmoid)
{
	// Initialize the parameters
	int final_iteration = 0;
	int all_size = init_src_pts.size();
	int QR_times = 0;
	int min_reduce_size = compact_idx.size() * compact_rate;
	double final_confidence = 0;
	double bestfinal_inlierRate = 0, best_global_error = INT_MAX;
	inlierThreshHold = inlierThresh;
	double gloabl_inlierThreshHold = inlierThresh;

	// Global RANSAC
	while (final_confidence <= maxConfidence && final_iteration < maxIters && QR_times < 10) {
		tiny_iteration = 0;
		
		int tinyWithdraw_size = round(compact_idx.size() * extractRate);
		vector<int> tiny_indices, tiny_weight;
		vector<Point2f> tiny_src, tiny_tar;

		// Extract the sub correspondences set
		pick_TinyCorresponence(init_src_pts, init_tar_pts, tiny_src, tiny_tar, tiny_indices, compact_idx, tinyWithdraw_size, weight, tiny_weight);

		// Local RANSAC
		bool earlyR = tinyCOOSAC(tiny_src, tiny_tar, tiny_indices, high_idx, tiny_weight, sigmoid, QR_times);

		// Update the global iteration
		if (earlyR)
			final_iteration = final_iteration * 2 + tiny_iteration;
		else
			final_iteration += tiny_iteration;

		double inlier_count = 0;
		double global_error = 0.0;
		vector<double> global_error_list(init_src_pts.size());
		vector<int> mask(all_size);

		computeError(init_src_pts, init_tar_pts, best_tiny_H, global_error_list);
		self_adjusting(global_error_list, gloabl_inlierThreshHold, inlier_count, mask, weight, inliers_outliers_mask,
			compact_idx, ori_bin_idx, len_bin_idx, ori_bin_num, len_bin_num, high_idx, QR_times, min_reduce_size, global_error);

		double final_inlierRate = inlier_count / all_size;

		if (final_inlierRate > bestfinal_inlierRate || best_global_error >= global_error) {
			best_tiny_H.copyTo(final_H);
			bestfinal_inlierRate = final_inlierRate;
			best_inliers = mask;
			best_global_error = min(best_global_error, global_error);
		}
		final_confidence = cal_confidence(bestfinal_inlierRate, final_iteration);
		QR_times++;
	}

	Mat initial_H = OPT_H(init_src_pts, init_tar_pts, weight);
	Mat eigen_H = OptimizeHomographyLM(init_src_pts, init_tar_pts, initial_H, weight);

	vector<double> eigen_error_list(all_size);
	vector<int> eigen_mask(all_size);
	computeError(init_src_pts, init_tar_pts, eigen_H, eigen_error_list);

	double eigen_inlier_count = 0, eigen_err_all = 0;
	findInliers(eigen_error_list, inlierThreshHold, eigen_inlier_count, eigen_mask, eigen_err_all);
	double eigen_inlierRate = eigen_inlier_count / all_size;

	if (eigen_inlierRate > bestfinal_inlierRate /*&& eigen_err_all <= best_global_error*/) {
		eigen_H.copyTo(final_H);
		bestfinal_inlierRate = eigen_inlierRate;
		best_inliers = eigen_mask;
		//cout << "------------- Weighted is Working -----------------\n";
	}

	homoinfo result;
	result.H = final_H;
	result.inliers = best_inliers;
	result.final_iteration = final_iteration;
	result.final_inlierRatio = bestfinal_inlierRate;

	return result;
}