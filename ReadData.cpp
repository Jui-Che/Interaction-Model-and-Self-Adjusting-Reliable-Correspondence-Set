#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <numeric>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<int> readRemovalList(const string& filename) {
	ifstream file(filename);
	if (!file.is_open()) {
		throw runtime_error("Unable to open file: " + filename);
	}
	vector<int> remove_list;
	string line;
	while (getline(file, line)) {
		istringstream lineStream(line);
		string data;
		if (getline(lineStream, data, ',')) {
			remove_list.push_back(stoi(data));
		}
	}
	file.close();
	return remove_list;
}

void adjustMatches(vector<Point2f>& source_match_pt, vector<Point2f>& target_match_pt,
	vector<int>& ground_truth, const vector<int>& remove_list) {
	for (int i = remove_list.size() - 1; i >= 0; --i) {
		int index = remove_list[i];
		if (index < source_match_pt.size() && index < target_match_pt.size() && index < ground_truth.size()) {
			source_match_pt.erase(source_match_pt.begin() + index);
			target_match_pt.erase(target_match_pt.begin() + index);
			ground_truth.erase(ground_truth.begin() + index);
		}
	}
}

void ModifyInlier(vector<Point2f>& all_source_match_pt, vector<Point2f>& all_target_match_pt, vector<int>& all_ground_truth,
	string filename, vector<Point2f>& source_match_pt, vector<Point2f>& target_match_pt, vector<int>& ground_truth) {

	auto remove_list = readRemovalList(filename);

	adjustMatches(source_match_pt, target_match_pt, ground_truth, remove_list);
	double gt_inlier = accumulate(ground_truth.begin(), ground_truth.end(), 0.0);
	double inlier_rate = gt_inlier / ground_truth.size();

}

vector<string> splittoken(string& s, char delimiter) {
	vector<string> tokens;
	string token;
	istringstream tokenStream(s);
	while (getline(tokenStream, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}

void readFileToPoints(string& filename, vector<Point2f>& points) {
	ifstream file(filename);
	if (!file.is_open()) {
		throw runtime_error("Failed to open file: " + filename);
	}
	string line;
	while (getline(file, line)) {
		auto tokens = splittoken(line, ',');
		points.emplace_back(std::stof(tokens[0]), std::stof(tokens[1]));
	}
}

void readFileToTruth(string& filename, vector<int>& truths) {
	ifstream file(filename);
	if (!file.is_open()) {
		throw runtime_error("Failed to open file: " + filename);
	}
	string line;
	auto toInt = [](const std::string& val) -> int {return val == "False" ? 0 : 1; };
	while (getline(file, line)) {
		auto tokens = splittoken(line, ',');
		for (const auto& token : tokens) {
			truths.push_back(toInt(token));
		}
	}
}

void LoadData(string& datasetName, string& sourceName, string& targetName,
	vector<Point2f>& all_source_match_pt, vector<Point2f>& all_target_match_pt,
	vector<int>& all_ground_truth) {

	string basePath = ".\\" + datasetName + "\\";
	string sourcePtsFile = basePath + sourceName.substr(4, 4) + "_pts.csv";
	string targetPtsFile = basePath + targetName.substr(4, 4) + "_pts.csv";
	string groundTruthFile = basePath + sourceName.substr(4, 4) + "_" + targetName.substr(4, 4) + ".csv";

	readFileToPoints(sourcePtsFile, all_source_match_pt);
	readFileToPoints(targetPtsFile, all_target_match_pt);
	readFileToTruth(groundTruthFile, all_ground_truth);
}

void ReadImage(Mat& sourceImg, Mat& targetImg, string dataset_name, string sourceName, string targetName) {
	string basePath = ".\\" + dataset_name + "\\";
	string srcPath = basePath + sourceName;
	string tarPath = basePath + targetName;
	sourceImg = imread(srcPath);
	targetImg = imread(tarPath);
	if (dataset_name != "UAV" && dataset_name != "VGG") {
		resize(sourceImg, sourceImg, Size(1080, 720), INTER_LINEAR);
		resize(targetImg, targetImg, Size(1080, 720), INTER_LINEAR);
	}
}

void load_file(string root, vector<string>& src_data, vector<string>& tar_data) {
	// load delete list
	set<string> delete_list;
	ifstream f_delete(root + "delete_list.txt");
	string line;
	while (getline(f_delete, line))
		delete_list.insert(line);
	//delete_list.insert("IMG_0129.JPG");
	//delete_list.insert("IMG_0123.JPG");


	//delete_list.clear();
	// load data number except delete data
	for (int i = 0; i < 200; i++) {
		char src_idx[5];
		char tar_idx[5];
		sprintf_s(src_idx, "%04d", i);
		strcpy_s(tar_idx, sizeof(tar_idx), src_idx);
		tar_idx[0] = '1';
		ifstream f(root + "IMG_" + string(src_idx) + ".JPG");
		if (f.good() && delete_list.find("IMG_" + string(src_idx) + ".JPG") == delete_list.end()) {
			src_data.push_back("IMG_" + string(src_idx) + ".JPG");
			tar_data.push_back("IMG_" + string(tar_idx) + ".JPG");
		}
	}

	return;
}

void ReadDataName(string datasetName, vector<string>& dataset_1, vector<string>& dataset_2) {

	if (datasetName == "Airport") {
		dataset_1 = { "IMG_0061.JPG","IMG_0116.JPG","IMG_0177.JPG","IMG_0282.JPG","IMG_3479.JPG" };
		dataset_2 = { "IMG_0062.JPG","IMG_0117.JPG","IMG_0178.JPG","IMG_0283.JPG","IMG_3480.JPG" };
	}
	else if (datasetName == "Small_Village") {
		dataset_1 = { "IMG_0924.JPG","IMG_0970.JPG","IMG_1011.JPG","IMG_1113.JPG","IMG_1204.JPG" };
		dataset_2 = { "IMG_0925.JPG","IMG_0971.JPG","IMG_1012.JPG","IMG_1114.JPG","IMG_1205.JPG" };
	}
	else if (datasetName == "University_Campus") {
		dataset_1 = { "IMG_0060.JPG","IMG_0098.JPG","IMG_0172.JPG","IMG_0333.JPG","IMG_0403.JPG" };
		dataset_2 = { "IMG_0061.JPG","IMG_0099.JPG","IMG_0173.JPG","IMG_0334.JPG","IMG_0404.JPG" };
	}
	else if (datasetName == "UAV") {
		load_file(".\\UAV\\", dataset_1, dataset_2);
	}
	else if (datasetName == "VGG") {
		load_file(".\\VGG\\", dataset_1, dataset_2);
	}
	else {
		cout << "You need to choose a dataset." << endl;
	}
}