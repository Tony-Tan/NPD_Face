#ifndef NPDFACE_MODEL_H
#define NPDFACE_MODEL_H
#include "DQTree.h"
#include "ReadData.h"
#include "NPDFeature.h"
class Model
{
public:
	std::vector<DQTree> treeArray;
	int stageNum;
	std::string modelIndex;
	std::string modelPath;
	
public:
	Model(std::string modlepath_);
public:
	void loadModel();
	void writeModel();
	void train(Samples& pos,Samples& neg,std::string negBigImageFile,Configuration config);
	inline void appendTree(DQTree newTree_);
	inline int test(unsigned char grayImage[SAMPLE_SIZE][SAMPLE_SIZE],float* fit );
	void initSamples(Samples& pos);
	void merge(std::vector<DQTree> newTrees_);
	void save(std::vector<DQTree> stage);
};
#endif//NPDFACE_MODEL_H