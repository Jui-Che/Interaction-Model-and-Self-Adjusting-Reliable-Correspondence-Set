#pragma once
#ifndef READDATA_H
#define READDATA_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

void ModifyInlier(vector<Point2f>& all_source_match_pt, vector<Point2f>& all_target_match_pt, vector<int>& all_ground_truth,
	string filename, vector<Point2f>& source_match_pt, vector<Point2f>& target_match_pt, vector<int>& ground_truth);

void LoadData(string& datasetName, string& sourceName, string& targetName,	vector<Point2f>& all_source_match_pt, vector<Point2f>& all_target_match_pt,	vector<int>& all_ground_truth);

void ReadImage(Mat& sourceImg, Mat& targetImg, string dataset_name, string source, string target);

void ReadDataName(string datasetName, vector<string>& dataset_1, vector<string>& dataset_2);

#endif // !READDATA_H