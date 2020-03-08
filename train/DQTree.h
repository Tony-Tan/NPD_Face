#ifndef NPDFACE_QDTREE_H
#define NPDFACE_QDTREE_H
#include "ReadData.h"


class DQTreeNode
{
public:
	class FeatureID 
	{
	public:
		int x_1,y_1,x_2,y_2;
		FeatureID()
		{
			x_1 = 0;
			y_1 = 0; 
			x_2 = 0;
			y_2 = 0;
		}
	}featureID;
	unsigned char cutPoint[2];
	NodeType nodeType;
	float fit;
	DQTreeNode* leftRoot;
	DQTreeNode* rightRoot;
public:
	DQTreeNode();

};
class DQTree
{
public:
	DQTreeNode * root;
	float threshold;
	int stageId;// stage num
	int treeId; // tree id in stage
public:
	DQTree(Samples & pos, Samples & neg, int stageId_,
		int treeId_, const Configuration & config);
	DQTree(std::string treeFilePath,int stageId,int treeId);
public:
	void learnThreshold(std::vector<OneSample*> posX);
	void read(const std::string treeFilePath);
	void write(const std::string treeFilePath);
	bool destoryDQTree(DQTree & root);
private:
	void learnDQTree(std::vector<OneSample*>&  posX,
		std::vector<OneSample*>& negX, float paretFit,
		const Configuration & config);

};
#endif//NPDFACE_QDTREE_H