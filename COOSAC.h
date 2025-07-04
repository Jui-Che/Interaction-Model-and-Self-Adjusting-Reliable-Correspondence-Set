#pragma once
#ifndef COOSAC_H
#define COOSAC_H

#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

class OptimizedFundamentalMatrixSolver {
public:
    static cv::Mat solve(const std::vector<cv::Point2f>& src_pts,
        const std::vector<cv::Point2f>& dst_pts,
        const std::vector<int>& weights) {
        if (src_pts.size() < 8) {
            throw std::invalid_argument("Need at least 8 points for fundamental matrix");
        }

        return solveFastOpenCV(src_pts, dst_pts, weights);
    }

    static void computeEpipolarErrorFast(const std::vector<cv::Point2f>& src_pts,
        const std::vector<cv::Point2f>& dst_pts,
        const cv::Mat& F,
        std::vector<double>& errors) {
        errors.clear();
        errors.reserve(src_pts.size());

        if (F.empty() || F.rows != 3 || F.cols != 3) {
            errors.assign(src_pts.size(), 1000.0);
            return;
        }

        for (size_t i = 0; i < src_pts.size(); ++i) {
            double x1 = src_pts[i].x, y1 = src_pts[i].y;
            double x2 = dst_pts[i].x, y2 = dst_pts[i].y;

            double l2_x = F.at<double>(0, 0) * x1 + F.at<double>(0, 1) * y1 + F.at<double>(0, 2);
            double l2_y = F.at<double>(1, 0) * x1 + F.at<double>(1, 1) * y1 + F.at<double>(1, 2);
            double l2_z = F.at<double>(2, 0) * x1 + F.at<double>(2, 1) * y1 + F.at<double>(2, 2);

            double denominator = sqrt(l2_x * l2_x + l2_y * l2_y);

            if (denominator > 1e-8) {
                double d = abs(l2_x * x2 + l2_y * y2 + l2_z) / denominator;
                errors.push_back(d);
            }
            else {
                errors.push_back(1000.0);  
            }
        }
    }

private:
    static cv::Mat solveFastOpenCV(const std::vector<cv::Point2f>& src_pts,
        const std::vector<cv::Point2f>& dst_pts,
        const std::vector<int>& weights) {
        try {
            std::vector<cv::Point2f> weighted_src, weighted_dst;

            for (size_t i = 0; i < src_pts.size(); ++i) {
                int weight = weights.empty() ? 1 : std::max(1, weights[i]);
                weight = std::min(weight, 3);

                for (int w = 0; w < weight; ++w) {
                    weighted_src.push_back(src_pts[i]);
                    weighted_dst.push_back(dst_pts[i]);
                }
            }

            cv::Mat F = cv::findFundamentalMat(weighted_src, weighted_dst,
                cv::FM_8POINT,
                1.55, 0.995);

            if (F.empty() || F.rows != 3 || F.cols != 3) {
                F = cv::findFundamentalMat(src_pts, dst_pts, cv::FM_8POINT);
            }

            return F;
        }
        catch (const std::exception& e) {
            return cv::Mat();
        }
    }
};

struct geometryinfo
{
    Mat F; 
    vector<int> inliers;
    double final_iteration = 0;
    double final_inlierRatio = 0;
};

constexpr std::array<std::array<int, 2>, 28> COMBINATIONS = { {
    {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7},
    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7},
    {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7},
    {3, 4}, {3, 5}, {3, 6}, {3, 7},
    {4, 5}, {4, 6}, {4, 7},
    {5, 6}, {5, 7},
    {6, 7}
} };

geometryinfo COOSAC(vector<Point2f>& init_src_pts, vector<Point2f>& init_tar_pts, vector<int>& ground_truth, vector<bool>& inliers_outliers_mask,
    unordered_set<int>& compact_idx, double inlierThresh, double extractRate, vector<int>& ori_bin_idx, vector<int>& len_bin_idx, int& ori_bin_num, int& len_bin_num,
    pair<int, int> high_idx, double compact_rate, vector<int>& weight, double sigmoid);

#endif // !COOSAC_H