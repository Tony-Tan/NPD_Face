#include "Model.h"
#include <fstream>
#include "GentleAdaBoost.h"
#include "Bootstrap.h"
static bool isFileExists(std::string filePath)
{
	std::fstream file;
	file.open(filePath.c_str(), std::ios::in);
	if (!file)
	{
		return false;
	}
	else
	{
		file.close();
		return true;
	}
	return false;
}
Model::Model(std::string modelPath_)
	:stageNum(0)
{
	modelPath = modelPath_;
	modelIndex = modelPath + std::string("modelIndex.txt");
	if (isFileExists(modelIndex))
	{
		loadModel();
	}
}

void Model::loadModel()
{
	FILE* modelIndexFile = fopen(modelIndex.c_str(), "r");
	if (NULL == modelIndexFile)
	{
		Message("Model::loadModel", "model index file open faile");
		stageNum = 0;
		return;
	}
	while (!feof(modelIndexFile))
	{
		int stageId = 0;
		int treeId = 0;
		char treePath[100];
		fscanf(modelIndexFile,"stage:%d tree:%d %s\n",
			&stageId, &treeId,treePath);
		DQTree tree(modelPath + std::string(treePath),stageId,treeId);
		appendTree(tree);
		stageNum = stageId + 1;
	}
	fclose(modelIndexFile);
}

void Model::writeModel()
{
}

void Model::train(Samples & pos, Samples & neg, std::string negBigImageFile, Configuration config)
{
	BootStrap bootStrap(negBigImageFile);
	if (stageNum != 0)
	{
		initSamples(pos);
		initSamples(neg);
		pos.removeRejectSamples();
		neg.removeRejectSamples();
	}
	
	while (true)
	{
		{
			char message[256];
			Message("","********************************************");
			sprintf(message,"Stage : %d",stageNum);
			Message("Model::train", message);
			Message("", "********************************************");
		}
		int numBeforeTrim = pos.samples.size();
		Gadaboost::GentleAdaboostCalcWeight(pos, 1, config.maxWeight);
		Gadaboost::trim(pos, config.trimFrac);
		pos.removeRejectSamples();
		if(numBeforeTrim-pos.samples.size()>0)
		{
			char message[256];
			sprintf(message, "Positive trim %d samples", numBeforeTrim - pos.samples.size());
			Message("Model::train", message);
		}
		
		if (neg.samples.size() < pos.samples.size())
		{
			std::vector<OneSample> negMiningSamples = 
				bootStrap.miningSamples(*this,neg.initialSize - neg.samples.size());
			//std::cout << "negative remine :" << neg.samples.size() << std::endl;
			//std::cout << "negative samples ptr remine :" << neg.samplePtrs.size() << std::endl;
			char message[256];
			sprintf(message, "%d Negative mined!", negMiningSamples.size());
			Message("Model::train", message);
			for (int i = 0, samplesIndex = neg.samples.size();
				i < negMiningSamples.size(); i++, samplesIndex++)
			{
				neg.samples.push_back(negMiningSamples[i]);
				neg.samplePtrs.push_back(&(neg.samples[samplesIndex]));
			}

		}
		Gadaboost::GentleAdaboostCalcWeight(pos, 1 , config.maxWeight);
		Gadaboost::GentleAdaboostCalcWeight(neg, -1, config.maxWeight);

		std::vector<DQTree> newTrees=Gadaboost::learn(pos, neg, config, *this);

		if (newTrees.size() == 0)
		{
			Message("Model::train", "No more DQT tree create,train finish!");
			break;
		}
		else 
		{
			merge(newTrees);
			save(newTrees);
			stageNum++;
		}
	}
}

inline void Model::appendTree(DQTree newTree_)
{
	treeArray.push_back(newTree_);
	stageNum = newTree_.stageId ;
}

inline int Model::test(unsigned char grayImage[SAMPLE_SIZE][SAMPLE_SIZE],float* fit)
{
	extern NPDfeature npdFea;
	
	int i = 0;
	for ( i = 0;i < treeArray.size(); i++)
	{
		DQTreeNode *node = treeArray[i].root;
		while (node->nodeType != LEAF) 
		{
			unsigned char fea = npdFea.NPDtable[grayImage[node->featureID.y_1][node->featureID.x_1]]
											[grayImage[node->featureID.y_2][node->featureID.x_2]];
			if (fea<node->cutPoint[0] || fea>node->cutPoint[1])
				node = node->leftRoot;
			else
				node = node->rightRoot;
		}
		(*fit) += node->fit;
		if ((*fit) < treeArray[i].threshold)
			break;
	}
	return i;
}

void Model::initSamples(Samples& sample)
{
	int T = treeArray.size();
	
	for (int i = 0; i < sample.samples.size(); i++)
	{
		int testStage = test(sample.samples[i].grayImage, &(sample.samples[i].fit));
		if (T > testStage)
		{
			sample.samples[i].reject = true;
		}
		else if (T < testStage)
		{
			Message("Model::initSamples", "Logical error occurrence");
		}
	}
}

void Model::merge(std::vector<DQTree> newTrees_)
{
	for (int i = 0; i < newTrees_.size(); i++)
	{
		appendTree(newTrees_[i]);
	}
}

void Model::save(std::vector<DQTree> stage)
{
	std::ofstream modelIndexFile;
	
	modelIndexFile.open(modelIndex.c_str(), std::ios::app);
	if (!modelIndexFile) //检查文件是否正常打开
	{
		Message("Model::save", "file open fail!");
	}
	else
	{
		//stage:0 tree:1 stage_0_tree1.npd
		for (int i = 0; i < stage.size(); i++)
		{
			char fileName[256];
			sprintf(fileName, "stage_%d_tree%d.npd",stage[i].stageId,stage[i].treeId);
			modelIndexFile << "stage:" << stage[i].stageId 
				<<" tree:"<<stage[i].treeId<<" "<< fileName <<std::endl;
			stage[i].write(modelPath);
		}
		modelIndexFile.close();
	}
}
