#include "Message.h"
#include "ReadData.h"
#include "NPDFeature.h"
#include <opencv2/opencv.hpp>
#include "Model.h"
using namespace cv;
void showSamples(const Samples & samples_){
    
	for (int k = 0; k < samples_.samples.size(); k++)
	{
		cv::Mat image = cv::Mat(samples_.height, samples_.width, CV_8UC1);
		OneSample sample_ = samples_.samples[k];
		for (int j = 0; j < samples_.height; j++)
		{
			for (int i = 0; i < samples_.width; i++) {
				image.at<uchar>(j, i) = sample_.grayImage[j][i];
			}
		}
		cv::imshow("Image", image);
		cv::waitKey(0);
	}
}
int main()
{
    Samples pos("F:\\aflw\\face\\",200000,"positive samples");
	Samples neg("F:\\non_face\\", 200000, "negative samples");
	//showSamples(pos);
	Model npdModel("D:\\Projects\\NPDFace\\Data\\Model\\");
	Configuration config;
	
	npdModel.train(pos,neg,"D:\\Projects\\NPDFace\\Data\\nonFaceBig.txt", config);
	return 0;
}




