#ifndef GENTLEADABOOST_H
#define GENTLEADABOOST_H
#include "Model.h"
#include "ReadData.h"
#include "Model.h"
#include "Configuration.h"
class Gadaboost
{
public:
	static std::vector<DQTree> learn(Samples& pos, Samples& neg, const Configuration& config, Model & model);
	static void GentleAdaboostCalcWeight(Samples& samples, int label, float maxWeight);
	//static void GentleAdaboostCalcWeight(Samples&  trainSample, int label, float maxWeight, float currentThreshold);
	static int trim(Samples& samples, float trimFrac);

	
};
#endif//GENTLEADABOOST_H