#pragma once
#ifndef COOSAC_H
#define COOSAC_H

#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

class HomographyFourPointSolver {
public:
	static cv::Mat solve(const std::vector<cv::Point2f>& src_pts,
		const std::vector<cv::Point2f>& dst_pts,
		const std::vector<int>& weights) {
		if (src_pts.size() != dst_pts.size() || src_pts.size() < 4) {
			throw std::invalid_argument("Invalid input: need at least 4 point pairs");
		}

		std::vector<double> adjusted_weights;
		adjustWeights(weights, adjusted_weights);

		if (src_pts.size() == 4) {
			return solveMinimal(src_pts, dst_pts, adjusted_weights);
		}
		else {
			return solveNonMinimal(src_pts, dst_pts, adjusted_weights);
		}
	}

private:
	static void adjustWeights(const std::vector<int>& original_weights, std::vector<double>& adjusted_weights) {
		adjusted_weights.clear();
		adjusted_weights.reserve(original_weights.size());

		bool all_zero = std::all_of(original_weights.begin(), original_weights.end(), [](int w) { return w == 0; });

		if (all_zero) {
			adjusted_weights.assign(original_weights.size(), 1.0);
		}
		else {
			for (int w : original_weights) {
				adjusted_weights.push_back(w == 0 ? 1e-6 : static_cast<double>(w));
			}
		}
	}
	static cv::Mat solveMinimal(const std::vector<cv::Point2f>& src_pts,
		const std::vector<cv::Point2f>& dst_pts,
		const std::vector<double>& weights) {
		Eigen::Matrix<double, 8, 9> A;

		for (int i = 0; i < 4; ++i) {
			double weight = weights.empty() ? 1.0 : static_cast<double>(weights[i]);
			double x1 = src_pts[i].x, y1 = src_pts[i].y;
			double x2 = dst_pts[i].x, y2 = dst_pts[i].y;

			A.row(i * 2) << weight * x1, weight* y1, weight, 0, 0, 0, -weight * x2 * x1, -weight * x2 * y1, -weight * x2;
			A.row(i * 2 + 1) << 0, 0, 0, weight* x1, weight* y1, weight, -weight * y2 * x1, -weight * y2 * y1, -weight * y2;
		}

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
		Eigen::VectorXd h = svd.matrixV().col(8);

		cv::Mat H(3, 3, CV_64F);
		double inv_h22 = 1.0 / h(8);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				H.at<double>(i, j) = h(i * 3 + j) * inv_h22;
			}
		}

		return H;
	}

	static cv::Mat solveNonMinimal(const std::vector<cv::Point2f>& src_pts,
		const std::vector<cv::Point2f>& dst_pts,
		const std::vector<double>& weights) {
		int n = src_pts.size();
		Eigen::MatrixXd A(2 * n, 9);
		Eigen::VectorXd b(2 * n);

		for (int i = 0; i < n; ++i) {
			double weight = weights.empty() ? 1.0 : std::max(1e-8, static_cast<double>(weights[i]));
			double x1 = src_pts[i].x, y1 = src_pts[i].y;
			double x2 = dst_pts[i].x, y2 = dst_pts[i].y;
			A.row(i * 2) << weight * x1, weight* y1, weight, 0, 0, 0, -weight * x2 * x1, -weight * x2 * y1, -weight * x2;
			A.row(i * 2 + 1) << 0, 0, 0, weight* x1, weight* y1, weight, -weight * y2 * x1, -weight * y2 * y1, -weight * y2;
			b(i * 2) = 0;
			b(i * 2 + 1) = 0;
		}
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
		Eigen::VectorXd h = svd.matrixV().col(8);

		cv::Mat H(3, 3, CV_64F);
		double inv_h22 = 1.0 / h(8);
		if (std::abs(h(8)) < 1e-8) {
			std::cout << "Warning: h(8) is close to zero. Value: " << h(8) << std::endl;
			inv_h22 = 0;
		}

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				H.at<double>(i, j) = h(i * 3 + j) * inv_h22;
				if (std::isnan(H.at<double>(i, j)) || std::isinf(H.at<double>(i, j))) {
					std::cout << "Warning: NaN or Inf detected in H at (" << i << "," << j << ")" << std::endl;
				}
			}
		}

		return H;
	}
};
struct homoinfo
{
	Mat H;
	vector<int> inliers;
	double final_iteration = 0;
	double final_inlierRatio = 0;
};

constexpr std::array<std::array<int, 2>, 6> COMBINATIONS = { {
	{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
} };

homoinfo COOSAC(vector<Point2f>& init_src_pts, vector<Point2f>& init_tar_pts, vector<int>& ground_truth, vector<bool>& inliers_outliers_mask,
	unordered_set<int>& compact_idx, double inlierThresh, double extractRate, vector<int>& ori_bin_idx, vector<int>& len_bin_idx, int& ori_bin_num, int& len_bin_num,
	pair<int, int> high_idx, double compact_rate, vector<int>& weight, double sigmoid);

#endif // !COOSAC_H