#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <fstream>
#include <numeric>

#include "ReadData.h"
#include "SA_COOSAC.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char* argv[]) {
	/*------------ Select dataset & Load data ------------*/ 
	// "Airport" "Small_Village" "University_Campus" "UAV" "VGG"
	string dataset_name = "VGG";

	// Dataset inlier rate
	//vector<float> adjust_inlier_rate = { 0.1 };
	vector<float> adjust_inlier_rate = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };

	// Repeat times
	int repeat_time = 10;

	// Select data in dataset
	vector<string> source_name, target_name;
	ReadDataName(dataset_name, source_name, target_name);

	// wirte result to file
	ofstream ofs;
	ofs.open(".\\" + dataset_name + "\\total_ours_fianl_tmp.csv", fstream::out);
	ofs << "GT inlier" << "," << "Total time" << "," << "Recall" << "," << "Precision" << "," << "F1-score" << "," << "RMSE" << "," << "MAE" << "," << "MEE" << endl;

	for (int adjustIdx = 0; adjustIdx < adjust_inlier_rate.size(); adjustIdx++) {
		double GT_inlier = 0, correspondence_all = 0, adjust_all = 0, reduce_all = 0;
		double	time_all = 0, recall_all = 0, precision_all = 0, f1_score_all = 0, rmse_all = 0;

		int datasize = source_name.size();
		vector<double> rmse_list(datasize, 0);
		for (int datasetIdx = 0; datasetIdx < datasize; datasetIdx++) {
			Mat sourceImg, targetImg;
			vector<Point2f> all_source_match_pt, all_target_match_pt;
			vector<int> all_ground_truth, remove_list;

			// Read image
			//ReadImage(sourceImg, targetImg, dataset_name, source_name[datasetIdx], target_name[datasetIdx]);

			// Load data correspondences
			LoadData(dataset_name, source_name[datasetIdx], target_name[datasetIdx], all_source_match_pt, all_target_match_pt, all_ground_truth);

			vector<Point2f> source_match_pt(all_source_match_pt);
			vector<Point2f> target_match_pt(all_target_match_pt);
			vector<int> ground_truth(all_ground_truth);

			// Modify GT correspondences
			stringstream stream;
			stream.precision(1);
			stream << fixed;
			stream << adjust_inlier_rate[adjustIdx];
			string str_adjust = stream.str();
			string combin_src_tar = source_name[datasetIdx].substr(4, 4) + "_" + target_name[datasetIdx].substr(4, 4);
			string filename = ".\\" + dataset_name + "\\adjust\\" + combin_src_tar + "_" + str_adjust + ".csv";
			// modify the inlier rate of the correspondences
			ModifyInlier(all_source_match_pt, all_target_match_pt, all_ground_truth, filename, source_match_pt, target_match_pt, ground_truth);
			GT_inlier += accumulate(ground_truth.begin(), ground_truth.end(), 0.0) / source_match_pt.size();

			vector<double> result = SA_COOSAC(target_match_pt, source_match_pt, ground_truth, repeat_time, sourceImg, targetImg);

			recall_all += result[0];
			precision_all += result[1];
			f1_score_all += result[2];
			time_all += result[3];
			rmse_all += result[4];
			rmse_list[datasetIdx] = result[4];
		}

		sort(rmse_list.begin(), rmse_list.end());

		cout << "=============================================\n";
		cout << "[DEBUG] Total result\n";
		cout << "GT inlier rate: " << GT_inlier / datasize << endl;
		cout << "Total time: " << time_all / datasize << endl;
		cout << "Recall: " << recall_all / datasize << endl;
		cout << "Precision: " << precision_all / datasize << endl;
		cout << "F1-score: " << f1_score_all / datasize << endl;
		cout << "RMSE: " << rmse_all / datasize << endl;
		cout << "=============================================\n";

		ofs << GT_inlier / datasize << "," << time_all / datasize << "," << recall_all / datasize << "," << precision_all / datasize << ","
			<< f1_score_all / datasize << "," << rmse_all / datasize << "," << rmse_list[datasize - 1] << "," << rmse_list[datasize / 2] << endl;
	}

	return 0;
}