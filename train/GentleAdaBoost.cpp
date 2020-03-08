#include "GentleAdaBoost.h"
#ifdef _OPENMP
#include <omp.h>
#endif


std::vector<DQTree> Gadaboost::learn(Samples & pos, Samples & neg, const Configuration & config, Model & model)
{
	std::vector<DQTree> newStage;
	int treeNumInStage = 0;
	while (model.treeArray.size()<config.maxNumWeaks)
	{
		if (neg.samples.size() < config.minSamples)
		{
			Message("Gadaboost::learn", "No enough Negative samples!");
			break;
		}

		int posTrimNum = Gadaboost::trim(pos, config.trimFrac);
		int negTrimNum = Gadaboost::trim(neg, config.trimFrac);
		if (posTrimNum)
		{
			char message[256];
			sprintf(message, "%d Positive samples are trimmed!", posTrimNum);
			Message("Model::train", message);
		}
		if (negTrimNum)
		{
			char message[256];
			sprintf(message, "%d Negative samples are trimmed!", negTrimNum);
			Message("Model::train", message);
		}
		{
			char message[256];
			Message("", "********************************************");
			sprintf(message, "%d Tree begin to learn.....", treeNumInStage+1);
			Message("Model::train", message);
			Message("", "********************************************");
		}
		{
			char message[256];
			sprintf(message, "%d (%d)negative remine.....", neg.samples.size(),neg.samplePtrs.size());
			Message("Model::train", message);
			
		}

		DQTree newTree(pos, neg, model.stageNum, treeNumInStage++,config);
		neg.removeRejectSamples(newTree.threshold);

		if (newTree.root == NULL)
		{
			Message("Gadaboost::learn", "no more tree be learned!");
			break;
		}
		newStage.push_back(newTree);
		float far = float(neg.samples.size()) / float(neg.initialSize);
		if (far < config.maxFAR)
		{
			Message("Gadaboost::learn", "far is approaching maxFar!Train can be stoped!");
			break;
		}
		if (neg.samples.size() < config.minSamples|| 
			neg.samples.size() < pos.initialSize*config.minNegRatio)
		{
			Message("Gadaboost::learn", "No enough Negative samples!");
			break;
		}
		Gadaboost::GentleAdaboostCalcWeight(pos, 1, config.maxWeight);
		Gadaboost::GentleAdaboostCalcWeight(neg, -1, config.maxWeight);
	}
	return newStage;
}

void Gadaboost::GentleAdaboostCalcWeight(Samples&  trainSample, int label, float maxWeight)
{
	if (trainSample.samples.size() == 0)
	{
		Message("Model::GentleAdaboostCalcWeight", "Samples is empty");
	}
	float weightSum = 0.0;
	bool MaxWeightReached = false;
	int MaxWeightReached_num = 0;
	const float LOG_2_0 = log(2.0);
	const float E_2_0 = exp(2.0);
	int i = 0;

	for (; i < trainSample.samples.size(); i++)
	{
		float currentFit = 0.0;
		currentFit = -1*label*(trainSample.samples)[i].fit;
		if (currentFit <= 2.0)
		{
			(trainSample.samples)[i].weight = exp(currentFit);
		}
		else
		{
			(trainSample.samples)[i].weight = log(currentFit) - LOG_2_0 + E_2_0;
		}
		if ((trainSample.samples)[i].weight > maxWeight)
		{
			MaxWeightReached = true;
			MaxWeightReached_num++;
			trainSample.samples[i].weight = maxWeight;
		}
		weightSum += (trainSample.samples)[i].weight;
	}
	if (MaxWeightReached)
	{
		char message[512];
		if(label==1)
			sprintf(message, "Positave:%d Samples Reach Max Weight!", MaxWeightReached_num);
		else
			sprintf(message, "Negative:%d Samples Reach Max Weight!", MaxWeightReached_num);
		Message("Model::GentleAdaboostCalcWeight",message);
	}
	if (weightSum != 0)
	{
		int i = 0;
		for (i = 0; i < trainSample.samples.size(); i++)
		{
			(trainSample.samples)[i].weight /= weightSum;
		}
	}
	else
	{
		int i = 0;
		for (; i < trainSample.samples.size(); i++)
		{
			(trainSample.samples)[i].weight = 1.0 / trainSample.samples.size();
		}
	}
}
/*
void Gadaboost::GentleAdaboostCalcWeight(Samples&  trainSample, int label, float maxWeight,float currentThreshold)
{
	if (trainSample.samples.size() == 0)
	{
		Message("Model::GentleAdaboostCalcWeight", "Samples is empty");
	}
	float weightSum = 0.0;
	bool MaxWeightReached = false;
	int MaxWeightReached_num = 0;
	for (int i = 0; i < trainSample.samples.size(); i++)
	{
		float currentFit = 0.0;
		currentFit = -1.0*label*((trainSample.samples)[i].fit - currentThreshold);

		(trainSample.samples)[i].weight = exp(currentFit);
		if ((trainSample.samples)[i].weight > maxWeight)
		{
			MaxWeightReached = true;
			MaxWeightReached_num++;
		}
		(trainSample.samples)[i].weight = ((trainSample.samples)[i].weight> maxWeight ?
			maxWeight : (trainSample.samples)[i].weight);
		weightSum += (trainSample.samples)[i].weight;
	}
	if (MaxWeightReached)
	{
		char message[512];
		if (label == 1)
			sprintf(message, "Positave:%d Samples Reach Max Weight!", MaxWeightReached_num);
		else
			sprintf(message, "Negative:%d Samples Reach Max Weight!", MaxWeightReached_num);
		Message("Model::GentleAdaboostCalcWeight", message);
	}
	if (weightSum != 0)
	{
		int i = 0;
#ifdef _OPENMP
#pragma parallel omp for
#endif
		for (i = 0; i < trainSample.samples.size(); i++)
		{
			(trainSample.samples)[i].weight /= weightSum;
		}
	}
	else
	{
		for (int i = 0; i < trainSample.samples.size(); i++)
		{
			(trainSample.samples)[i].weight = 1.0 / trainSample.samples.size();
		}
	}
}
*/
int Gadaboost::trim(Samples & samples, float trimFrac)
{
	int trimNum = 0;
	for (int i = 0; i < samples.samples.size(); i++)
	{
		if (samples.samples[i].weight >= trimFrac)
		{
			samples.samples[i].reject = true;
			trimNum++;
		}
	}
	return trimNum;
}

