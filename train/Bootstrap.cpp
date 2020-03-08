#include "Bootstrap.h"
#include <vector>
#include <algorithm> 
std::vector<OneSample> BootStrap::grid(Model model,int sampleNum)
{
	
	return scan(model, sampleNum, SAMPLE_SIZE);
}

std::vector<OneSample> BootStrap::scan(Model model,int sampleNum , int step)
{
	std::vector<OneSample> scanSamples;
	
	int minedNum = 0;
	extern NPDfeature npdFea;
	while (minedNum <= sampleNum)
	{
		
		std::string fileName = negativeBigImagePaths[currentImageId];
		cv::Mat image = cv::imread(fileName, 0);
		if (image.data == NULL)
		{
			char message[512];
			sprintf(message, "%s%s", fileName.c_str()," is empty!");
			Message("BootStrap::scan",message);
			currentImageId++;
			continue;
		}
		for (int fliptimes = 0; fliptimes<4; fliptimes++)
		{
			if (fliptimes % 4 == 1)
				cv::flip(image, image, -1);
			else if (fliptimes % 4 == 2)
				cv::flip(image, image, 0);
			else if (fliptimes % 4 == 3)
				cv::flip(image, image, 1);
			for (int s = currentScale; s < maxScaleTimes; s++)
			{
				int resizeRows = image.rows*scaleArray[s];
				int resizeCols = image.cols*scaleArray[s];
				if (resizeRows <= SAMPLE_SIZE || resizeCols <= SAMPLE_SIZE)
					break;
				cv::Mat imageResize;
				cv::resize(image, imageResize, cv::Size(resizeCols, resizeRows));
				int j = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
				for (j = 0; j < resizeRows - SAMPLE_SIZE; j += step)
				{
					for (int i = 0; i < resizeCols - SAMPLE_SIZE; i += step)
					{
						float fit = 0.0;
						bool pass = true;
						for (int t = 0; t < model.treeArray.size(); t++)
						{
							DQTreeNode *node = model.treeArray[t].root;
							while (node->nodeType != LEAF)
							{
								int pixel_1 = imageResize.at<uchar>(node->featureID.y_1 + j,
									node->featureID.x_1 + i);
								int pixel_2 = imageResize.at<uchar>(node->featureID.y_2 + j,
									node->featureID.x_2 + i);

								unsigned char fea = npdFea.NPDtable[pixel_1][pixel_2];
								if (fea<node->cutPoint[0] || fea>node->cutPoint[1])
								{
									node = node->leftRoot;
								}
								else
								{
									node = node->rightRoot;
								}
							}
							fit += node->fit;
							if (fit < model.treeArray[t].threshold)
							{

								pass = false;
								break;
							}
						}
#ifdef _OPENMP
#pragma omp critical 
#endif			
						if (pass)
						{
							OneSample sample(imageResize, i, j, SAMPLE_SIZE, fit);
							scanSamples.push_back(sample);
							minedNum++;
						}
					}
				}
			}
		}
	
		currentImageId++;
		char message[512];
		sprintf(message, "|%d samples to be found",  sampleNum- minedNum);
		Message("BootStrap::scan", negativeBigImagePaths[currentImageId]+std::string(message));
	}
	return scanSamples;
}

BootStrap::BootStrap(std::string negBigImagePath_, 
	float facter_,int maxSacleTimes_,
	float randomSelectRatio_)
	:currentImageId(0),currentScale(0)
	, facter(facter_),maxScaleTimes(maxSacleTimes_),randomSelectRatio(randomSelectRatio_)
{
	// read big negative images
	char imagePath[256];
	FILE* negBigImagePathFile = 
		fopen(negBigImagePath_.c_str(), "r");
	if (negBigImagePathFile == NULL)
	{
		Message("BootStrap::BootStrap", "file open fail!");
	}
	while (!feof(negBigImagePathFile))
	{

		fscanf(negBigImagePathFile, "%s\n", imagePath);
		negativeBigImagePaths.push_back(std::string(imagePath));
	}
	fclose(negBigImagePathFile);
	//get scale factors
	float scale =2.0;
	for (int i = 0; i < maxSacleTimes_; i++) 
	{
		scaleArray.push_back(scale);
		scale *= facter_;
	}	
}

std::vector<OneSample> BootStrap::miningSamples(Model model, int sampleNum)
{
	char message[512];
	sprintf(message, "|mining %d samples", sampleNum );
	Message("BootStrap::miningSamples", std::string(message));
	int samplePoolNum = sampleNum / randomSelectRatio;
	std::vector<OneSample> samples;
	//std::vector<OneSample> gridSamples = grid(model, samplePoolNum*0.3);
	std::vector<OneSample> scanSamples = scan(model, samplePoolNum ,SAMPLE_SIZE/2);
	std::vector<OneSample> tempVector;
	//tempVector.insert(tempVector.end(), gridSamples.begin(), gridSamples.end());
	tempVector.insert(tempVector.end(), scanSamples.begin(), scanSamples.end());
	random_shuffle(tempVector.begin(), tempVector.end());
	for (int i = 0; i < sampleNum; i++)
	{
		samples.push_back(tempVector[i]);
	}
	return samples;
}
