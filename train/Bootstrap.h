#ifndef NPDFACE_BOOTSTRAP_H
#define NPDFACE_BOOTSTRAP_H
#include <string>
#include "ReadData.h"
#include "Model.h"
class BootStrap 
{
public:
	std::vector<std::string> negativeBigImagePaths;
	int currentImageId;
	int currentScale;
	std::vector<OneSample> grid(Model model,int sampleNum);
	std::vector<OneSample> scan(Model model,int sampleNum ,int step);
	float facter;
	int maxScaleTimes;
	std::vector<float> scaleArray;
	float randomSelectRatio;
public:
	BootStrap(std::string negativeBigImagePath_, 
		float facter_ = 0.9, int maxSacleTimes_ = 20, float randomSelectRatio=0.99);
	std::vector<OneSample> miningSamples(Model model, int sampleNum);
};

#endif