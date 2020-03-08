#ifndef NPDFACE_CONFIGURATION_HPP
#define NPDFACE_CONFIGURATION_HPP
#include "Message.h"
#define SAMPLE_SIZE 24
#define FEATURE_SIZE (SAMPLE_SIZE*SAMPLE_SIZE*(SAMPLE_SIZE*SAMPLE_SIZE - 1) / 2)
#define GRAY_LEVEL 256
#define STORAGE_ACCURACY 5 //
#define STORAGE_ACCURACY_CARRY 1e-5

typedef enum { SPLIT = 0,LEAF=1 } NodeType;
class Configuration 
{
public:
	int treeLevel;					// the maximal depth of the DQT trees to be learned
	int maxNumWeaks;				// maximal number of weak classifiers to be learned
	float minDR;					// minimal detection rate required
	float maxFAR;					// maximal FAR allowed; stop the training if reached
	int minSamples;					// minimal samples required to continue training
	float minNegRatio;				// minimal fraction of negative samples required to remain,
									// w.r.t.the total number of negative samples.This is a signal of
									// requiring new negative sample bootstrapping.Also used to avoid
									// overfitting.
	float trimFrac;				// weight trimming in AdaBoost
	float samFrac;					// the fraction of samples randomly selected in each iteration
									// for training; could be used to avoid overfitting.
	float minLeafFrac;				//minimal sample fraction w.r.t.the total number of
									// samples required in each leaf node.This is used to avoid overfitting.
	int minLeaf;					// minimal samples required in each leaf node.This is used to avoid overfitting.
	float maxWeight ;					// maximal sample weight in AdaBoost; used to ensure numerical stability.
	
	
public:
	Configuration() :
		treeLevel(8),
		maxNumWeaks(10000),
		minDR(1.0),
		maxFAR(1e-16),
		minSamples(10000),
		minNegRatio(0.6),	
		trimFrac(0.05),		
		samFrac(1.0),			
		minLeafFrac (0.01),		
		minLeaf(100),
		maxWeight(100)
	{
	}

	bool setConfigValue(const std::string valueName, float value)
	{
		if (valueName == "treeLevel")
		{
			treeLevel = (int)value;
		}
		else if (valueName == "maxNumWeaks")
		{
			maxNumWeaks = (int)value;
		}
		else if (valueName == "minDR")
		{
			minDR = value;
		}
		else if (valueName == "maxFAR")
		{
			maxFAR = value;
		}
		else if (valueName == "minSamples")
		{
			minSamples = int(value);
		}
		else if (valueName == "minNegRatio")
		{
			minNegRatio = value;
		}
		else if (valueName == "trimFrac")
		{
			trimFrac = value;
		}
		else if (valueName == "samFrac")
		{
			samFrac = value;
		}
		else if (valueName == "minLeafFrac")
		{
			minLeafFrac = value;
		}
		else if (valueName == "minLeaf")
		{
			minLeaf = int(value);
		}
		else if (valueName == "maxWeight")
		{
			maxWeight = value;
		}
		else 
		{
			Message("Configuration::setConfigValue", "wrong value name!");
			return false;
		}
		return true;
	}

};
#endif//NPDFACE_CONFIGURATION_HPP
