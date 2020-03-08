#ifndef NPDFACE_READDATA_H
#define NPDFACE_READDATA_H
#include <string>
#include <vector>
#include "Message.h"
#include <opencv2/opencv.hpp>
#include "Configuration.h"
typedef unsigned int uint;
class OneSample
{
public:
    unsigned char grayImage[SAMPLE_SIZE][SAMPLE_SIZE];
    bool useable;
    float weight;
    float fit;
	bool reject;
public:
    OneSample(std::string imagePath_)
        :weight(1.0), fit(0.0),useable(false),reject(false)
    {
		int width= SAMPLE_SIZE;
		int height = SAMPLE_SIZE;
        if(!imagePath_.empty())
        {
            cv::Mat image_=cv::imread(imagePath_,0);
            if(image_.empty())
            {
                Message("Samples::OneSample",imagePath_ +" load fail!");
            }
            else
            {
                cv::Mat image;
				if (image_.cols != SAMPLE_SIZE || image_.rows != SAMPLE_SIZE)
					cv::resize(image_, image, cv::Size(SAMPLE_SIZE, SAMPLE_SIZE));
				else
					image = image_;
                //memory copy
                for (int j=0; j<height; j++) {
                    for (int i=0; i<width; i++) {
                        grayImage[j][i] = image.at<uchar>(j,i);
                    }
                }
                useable=true;
            }
        }
    }
	OneSample(cv::Mat& image ,int x,int y,int size,float fit_)
		:weight(1.0), fit(fit_), useable(true), reject(false)
	{
		for (int j = 0; j < SAMPLE_SIZE; j++)
		{
			for (int i = 0; i < SAMPLE_SIZE; i++)
			{
				grayImage[j][i] = image.at<uchar>(y + j, x + i);
			}
		}
		
	}
    ~OneSample()
    {
    }
    bool empty()
    {
        return useable;
    }
};

class Samples
{
public:
    std::vector<OneSample> samples;
	std::vector<OneSample*> samplePtrs;
	int width;
	int height;
    uint initialSize;
public:
    Samples(const char * samplesPath ,int size,const char * sampelsName);
public:
	void removeRejectSamples();
	void removeRejectSamples(float fit);
};
#endif
