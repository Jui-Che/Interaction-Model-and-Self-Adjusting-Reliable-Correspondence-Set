#include <vector>
#include <numeric>
#include <fstream>
#include <random>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "Calculate.h"

using namespace std;
using namespace cv;
using namespace Eigen;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);

struct HomographyElements {
	double h00, h01, h02; 
	double h10, h11, h12;
	double h20, h21;    

	explicit HomographyElements(const Mat& H) {
		h00 = H.at<double>(0, 0); h01 = H.at<double>(0, 1); h02 = H.at<double>(0, 2);
		h10 = H.at<double>(1, 0); h11 = H.at<double>(1, 1); h12 = H.at<double>(1, 2);
		h20 = H.at<double>(2, 0); h21 = H.at<double>(2, 1);
	}
};

class HomographyEstimator : public cv::MinProblemSolver::Function
{
public:
	HomographyEstimator(const vector<Point2f>& src, const vector<Point2f>& dst, const vector<int>& weights)
		: src_(src), dst_(dst), weights_(weights) {}

	int getDims() const override { return 9; }

	double calc(const double* x) const override
	{
		Mat H(3, 3, CV_64F, const_cast<double*>(x));
		vector<Point2f> src_projected;
		perspectiveTransform(src_, src_projected, H);

		double sum = 0;
		for (size_t i = 0; i < src_.size(); ++i)
		{
			double dx = dst_[i].x - src_projected[i].x;
			double dy = dst_[i].y - src_projected[i].y;
			sum += weights_[i] * (dx * dx + dy * dy);
		}
		return sum;
	}

private:
	const vector<Point2f>& src_;
	const vector<Point2f>& dst_;
	const vector<int>& weights_;
};


double ScottBinWidth(vector<double>& data) {
	// Scott's normal reference rule
	double sum = 0.0, mean = 0.0, standardDeviation = 0.0;
	int n = data.size();

	mean = accumulate(data.begin(), data.end(), 0.0) / n;

	double sq_sum = inner_product(data.begin(), data.end(), data.begin(), 0.0);
	double sigma = sqrt(sq_sum / n - mean * mean);
	return 3.49 * sigma * pow(n, -1.0 / 3);
}

void sort_index(vector<int>& frequency) {
	vector<pair<int, int>> vp;
	for (int i = 0; i < frequency.size(); i++)
		vp.push_back(make_pair(frequency[i], i));

	sort(vp.rbegin(), vp.rend());

	for (int i = 0; i < vp.size(); i++)
		frequency[i] = vp[i].second;
}

int cal_frequency_descending(const vector<double>& input, vector<double>& pitch_list) {

	// as np.histogram in python 
	int pitch_size = pitch_list.size();
	vector<int> frequency(pitch_size - 1, 0);

	for (const auto& value : input) {
		auto it = lower_bound(pitch_list.begin(), pitch_list.end(), value);
		if (it != pitch_list.begin() && it != pitch_list.end()) {
			++frequency[distance(pitch_list.begin(), it) - 1];
		}
	}
	//writeVectorsToCSV(frequency, pitch_list, "frequency.csv");
	
	// as np.argsort in python
	sort_index(frequency);

	// Return the index of the most frequent pitch
	return frequency[0];
}

int cal_pitch_frequency(vector<double>& input_info, const double pitch_dist, vector<double>& pitch_list) {

	auto minmax = minmax_element(input_info.begin(), input_info.end());
	double min_input = round(*minmax.first * 10) / 10.0;
	double max_input = round(*minmax.second * 10) / 10.0;

	pitch_list.push_back(min_input - pitch_dist / 2);
	while (min_input < (max_input + pitch_dist)) {
		pitch_list.push_back(min_input);
		min_input += pitch_dist;
	}

	int freq_idx = cal_frequency_descending(input_info, pitch_list);

	return freq_idx;
}

vector<int> save_interval(vector<double>& inputs_info, vector<double>& pitch_list, int frequency, vector<int>& ori_bin_idx) {
	vector<int> saveId;
	int range = 2;
	int min_index = max(0, frequency - range);
	int max_index = min(frequency + range + 1, static_cast<int>(pitch_list.size()) - 1);

	double low_bound = pitch_list[min_index];
	double high_bound = pitch_list[max_index];
	double bin_dist = pitch_list[1] - pitch_list[0];
	double start_from = pitch_list[0] - bin_dist / 2;

	for (int i = 0; i < inputs_info.size(); i++) {
		if (low_bound < inputs_info[i] && inputs_info[i] <= high_bound)
			saveId.push_back(i);
		int bin_idx = (inputs_info[i] - start_from) / bin_dist;
		ori_bin_idx[i] = bin_idx;
	}

	return saveId;
}

vector<int> save_interval_second(vector<double>& inputs_info, vector<double>& pitch_list, int frequency, vector<int>& len_bin_idx,
	vector<int>& firstfilter_idx, vector<int>& compact_set, unordered_set<int>& compact_idx, vector<double>& length) {

	vector<int> saveId;
	int range = 2;
	int min_index = max(0, frequency - range);
	int max_index = min(frequency + range + 1, static_cast<int>(pitch_list.size()) - 1);

	double low_bound = pitch_list[min_index];
	double high_bound = pitch_list[max_index];
	double pitch_dist = pitch_list[1] - pitch_list[0];
	double start_form = pitch_list[0] - pitch_dist / 2;

	int filiter_idx = 0;
	for (int i = 0; i < length.size(); i++) {
		if (filiter_idx < firstfilter_idx.size() && i == firstfilter_idx[filiter_idx]) {
			if (low_bound < inputs_info[filiter_idx] && inputs_info[filiter_idx] <= high_bound) {
				saveId.push_back(filiter_idx);
				compact_set.push_back(i);
				compact_idx.insert(i);
			}
			filiter_idx++;
		}
		int j = (length[i] - start_form) / pitch_dist;
		len_bin_idx[i] = j;
	}

	return saveId;
}

void cal_diff(vector<double>& input, vector<double>& pitch_list, vector<int>& save_first, vector<int>& save_second, vector<int>& bin_idx)
{
	sort(save_first.begin(), save_first.end());
	sort(save_second.begin(), save_second.end());

	vector<int> difference;

	// calculate the difference between two vectors
	set_difference(save_first.begin(), save_first.end(), save_second.begin(), save_second.end(), back_inserter(difference));

	unordered_map<int, int> diff_set;
	for (int i = 0; i < save_first.size(); i++) {
		diff_set[save_first[i]] = input[i];
	}

	// as np.histogram in python 
	int pitch_size = pitch_list.size();

	for (auto index : difference) {
		int val = diff_set[index];
		for (int i = 0; i < pitch_size - 1; i++) {
			if (pitch_list[i] < val && val <= pitch_list[i + 1]) {
				bin_idx[index] = i;
				break;
			}
		}
	}

}

double calculateArea(const Point2f& ori_pt0, const Point2f& ori_pt1, const Point2f& tar_pt0, const Point2f& tar_pt1) {
	return abs(
		ori_pt0.x * ori_pt1.y + ori_pt1.x * tar_pt1.y + tar_pt1.x * tar_pt0.y + tar_pt0.x * ori_pt0.y -
		ori_pt0.y * ori_pt1.x - ori_pt1.y * tar_pt1.x - tar_pt1.y * tar_pt0.x - tar_pt0.y * ori_pt0.x
	) / 2;
}

inline double computePointError(const Point2f& src_pt,
	const Point2f& tar_pt,
	const HomographyElements& H) {
	const double w = 1.0 / (H.h20 * src_pt.x + H.h21 * src_pt.y + 1.0);

	const double proj_x = (H.h00 * src_pt.x + H.h01 * src_pt.y + H.h02) * w;
	const double proj_y = (H.h10 * src_pt.x + H.h11 * src_pt.y + H.h12) * w;

	const double dx = proj_x - tar_pt.x;
	const double dy = proj_y - tar_pt.y;

	return dx * dx + dy * dy;
}

void computeError(const vector<Point2f>& src_pts, const vector<Point2f>& tar_pts, const Mat& H, vector<double>& residuals_error) {
	const size_t num_points = src_pts.size();

	const HomographyElements h_elements(H);
	for (int i = 0; i < num_points; ++i) {
		residuals_error[i] = computePointError(src_pts[i], tar_pts[i], h_elements);
	}
}

Mat OptimizeHomographyLM(const vector<Point2f>& src, const vector<Point2f>& dst, const Mat& initialH, const vector<int>& weights)
{
	Mat h = initialH.reshape(1, 9);
	vector<double> h_vec;
	h.copyTo(h_vec);

	Ptr<DownhillSolver> solver = DownhillSolver::create();
	Ptr<MinProblemSolver::Function> ptr_F = makePtr<HomographyEstimator>(src, dst, weights);

	solver->setFunction(ptr_F);

	Mat step = Mat::ones(9, 1, CV_64F) * 1e-4;
	solver->setInitStep(step);

	TermCriteria termcrit(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 1e-6);
	solver->setTermCriteria(termcrit);

	Mat optimized_h = Mat(h_vec).clone();
	solver->minimize(optimized_h);

	Mat optimizedH = optimized_h.reshape(1, 3);
	optimizedH /= optimizedH.at<double>(2, 2);

	return optimizedH;
}

bool adding_compact(unordered_set<int>& compact_idx, const vector<int>& ori_bin_idx, const vector<int>& len_bin_idx,
	const int ori_bin_num, const int len_bin_num, const pair<int, int> peak, const int idx) {
	double rand_value = dis(gen);
	int len_dist = abs(peak.second - len_bin_idx[idx]);
	int ori_dist = abs(peak.first - ori_bin_idx[idx]);

	double adding_val = exp(-(static_cast<double>(ori_dist) / ori_bin_num)) + exp(-(static_cast<double>(len_dist) / len_bin_num));

	if (adding_val >= rand_value) {
		compact_idx.insert(idx);

		return true;
	}
	return false;
}

bool MCMechanism(unordered_set<int>& compact_idx, vector<int>& weight, int idx) {
	double P_x_prime = exp(-weight[idx]);
	double rand_value = dis(gen);

	if (rand_value <= P_x_prime) {
		compact_idx.erase(idx);

		return true;
	}

	return false;
}

void self_adjusting(vector<double>& err_list, double& projErr, double& inlier_count, vector<int>& inlier_mask, vector<int>& weight, vector<bool>& inliers_outliers_mask,
	unordered_set<int>& compact_idx, vector<int>& ori_bin_idx, vector<int>& len_bin_idx, int& ori_bin_num,
	int& len_bin_num,  pair<int, int> peak, int QR_times, int min_reduce_size, double& global_error) {

	double threshold = projErr * projErr;
	int remove_nums = 0 , add_nums = 0;
	int part = 10, ignore_num = 5;
	double  max_sigma = 0;

	for (int i = 0; i < err_list.size(); i++) {
		int f = err_list[i] <= threshold;
		bool isChanged = false;

		inlier_mask[i] = f;
		inlier_count += f;
		weight[i] += max(min(part, part - static_cast<int>(floor(err_list[i] / (threshold / 3)))), 0);

		if (f == 1) {	// now is inlier
			global_error += err_list[i];
			max_sigma = max(max_sigma, err_list[i]);
			if (!inliers_outliers_mask[i] && weight[i] <= ignore_num) {	// not in compact set
				// check should add to compact set
				if (adding_compact( compact_idx, ori_bin_idx, len_bin_idx, ori_bin_num, len_bin_num, peak, i)) {
					inliers_outliers_mask[i] = true;
					isChanged = true;
				}
				add_nums++;
			}
		}

		if (QR_times > 0 && f != 1 && inliers_outliers_mask[i] && !isChanged) {
			if (weight[i] <= ignore_num && min_reduce_size > remove_nums && compact_idx.size() > min_reduce_size) {
				if (MCMechanism(compact_idx, weight, i)) {
					remove_nums++;
					inliers_outliers_mask[i] = false;
				}
			}
		}
	}
	if (max_sigma != 0)
		projErr = sqrt((max_sigma + threshold) / 2);

	global_error /= inlier_count;

	//cout << "QR_Round" << QR_times << ", LCL-RANSAC: " << compact_idx.size() << endl;
	 
	//cout << "Inlier count: " << inlier_count << endl;
	//cout << "Add " << add_nums << " correspondences" << endl;
	//cout << "Remove " << remove_nums << " correspondences" << endl;
	//cout << "Compact set size: " << compact_idx.size() << endl;
}

Mat OPT_H(vector<Point2f>& src, vector<Point2f>& tar, vector<int>& weight) {
	const int points = src.size();
	MatrixXd A(2 * points, 9);
//# pragma omp parallel for num_threads(16) schedule(dynamic)
	for (int i = 0; i < points; ++i) {
		double w = weight[i];
		double x = src[i].x, y = src[i].y;
		double xp = tar[i].x, yp = tar[i].y;
		double wx = w * x, wy = w * y;
		int row1 = 2 * i, row2 = 2 * i + 1;

		A.row(row1) << wx, wy, w, 0, 0, 0, -wx * xp, -wy * xp, -w * xp;
		A.row(row2) << 0, 0, 0, wx, wy, w, -wx * yp, -wy * yp, -w * yp;
	}

	JacobiSVD<MatrixXd> svd(A, ComputeThinV);
	VectorXd h = svd.matrixV().col(8);

	Mat H(3, 3, CV_64F);
	double inv_h22 = 1.0 / h(8);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			H.at<double>(i, j) = h(i * 3 + j) * inv_h22;
		}
	}

	return H;
}

confusion confusion_matrix(vector<int>& GT, vector<int>& Prediction) {
	confusion cm;

	if (GT.size() != Prediction.size()) {
		cout << "different" << endl;
		cout << "GT: " << GT.size() << endl;
		cout << "Prediction: " << Prediction.size() << endl;
	}

	for (size_t i = 0; i < GT.size(); ++i) {
		if (GT[i] == 1 && Prediction[i] == 1)
			cm.TP++;
		else if (GT[i] == 0 && Prediction[i] == 0)
			cm.TN++;
		else if (GT[i] == 0 && Prediction[i] == 1)
			cm.FP++;
		else if (GT[i] == 1 && Prediction[i] == 0)
			cm.FN++;
	}

	return cm;
}

cal_RMSE calculateErrorMetrics(const vector<int>& ground_truth,
	const vector<int>& inlier_mask,
	const vector<Point2f>& src_pts,
	const vector<Point2f>& tar_pts,
	const Mat& H) {
	vector<float> errors;
	float sum_squared_error = 0.0f;
	float max_error = 0.0f;
	cal_RMSE cr;

	for (size_t i = 0; i < inlier_mask.size(); ++i) {
		if ( ground_truth[i] == 1) {
			Mat src_pt = (Mat_<double>(3, 1) << src_pts[i].x, src_pts[i].y, 1.0);
			Mat dst_pt = H * src_pt;
			dst_pt /= dst_pt.at<double>(2);

			float dx = dst_pt.at<double>(0, 0) - tar_pts[i].x;
			float dy = dst_pt.at<double>(1, 0) - tar_pts[i].y;
			float error = sqrt(dx * dx + dy * dy);

			errors.push_back(error);
			sum_squared_error += dx * dx + dy * dy;
			max_error = max(max_error, error);
		}
	}

	// Calculate RMSE
	cr.RMSE = sqrt(sum_squared_error / errors.size());
	//cr.RMSE = sum_squared_error / errors.size();

	// Calculate MAE
	cr.MAE = max_error;
	// Calculate MEE
	sort(errors.begin(), errors.end());
	cr.MEE = errors[errors.size() / 2];

	//cout << "RMSE: " << cr.RMSE << endl;
	//cout << "MAE: " << cr.MAE << endl;
	//cout << "MEE: " << cr.MEE << endl;
	return cr;
}