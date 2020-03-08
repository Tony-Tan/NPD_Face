#include "DQTree.h"
#include <fstream>
#include "NPDFeature.h"
#include "minMSE.h"
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif
NPDfeature npdFea;

static void writeNode(DQTreeNode* root, std::ofstream * file)
{
	if (root == NULL)
		return;
	if (root->nodeType == SPLIT)
		*file << root->nodeType << " " << int(root->cutPoint[0])
		<< " " << int(root->cutPoint[1])
		<< " (" << root->featureID.x_1 << ","
		<< root->featureID.y_1 << ")"
		<< " (" << root->featureID.x_2 << ","
		<< root->featureID.y_2 << ")" << std::endl;
	else if (root->nodeType == LEAF)
		*file << root->nodeType << " " <<setiosflags(std::ios::fixed) << std::setprecision(STORAGE_ACCURACY) << root->fit+ STORAGE_ACCURACY_CARRY << std::endl;
	writeNode(root->leftRoot, file);
	writeNode(root->rightRoot, file);
}


static void learnDQTreeNode(DQTreeNode ** root_, std::vector<OneSample*>& posX,
	std::vector<OneSample*>& negX, float paretFit, int depth, const Configuration & config)
{
	*root_ = new DQTreeNode;
	float fit[2];
	float minCost;
	//calc minCost
	float weightSum = 0.0;
	for (int i = 0; i < posX.size(); i++)
		weightSum += posX[i]->weight;

	minCost = weightSum*(1 - paretFit)*(1 - paretFit);
	weightSum = 0.0;
	for (int i = 0; i < negX.size(); i++)
		weightSum += negX[i]->weight;

	minCost += weightSum*(1 + paretFit)*(1 + paretFit);
	if (posX.size() == 0 || negX.size() == 0 ||
		posX.size() + negX.size() < 2 * config.minLeaf ||
		depth>=config.treeLevel)
	{
		(*root_)->nodeType = LEAF;
		(*root_)->fit = paretFit;
		{
			char message[256];
			sprintf(message, "LEAF  |Sample:%6d(pos:%4d,neg:%4d)|Fit:%f",
				posX.size() + negX.size(), posX.size(), negX.size(), paretFit);
			Message("learnDQTreeNode", message);
		}
		for (int i = 0; i < posX.size(); i++)
		{
			posX[i]->fit += paretFit;
		}
		for (int i = 0; i < negX.size(); i++)
		{
			negX[i]->fit += paretFit;
		}
		return;
	}
	gpuMSE(root_, posX, negX, config, fit, minCost);
	int x_1 = (*root_)->featureID.x_1;
	int x_2 = (*root_)->featureID.x_2;
	int y_1 = (*root_)->featureID.y_1;
	int y_2 = (*root_)->featureID.y_2;
	{
		char message[256];
		sprintf(message, "SPLITE|Sample:%6d(pos:%4d,neg:%4d)|Fit:%f",
			posX.size() + negX.size(), posX.size(), negX.size(), paretFit);
		Message("learnDQTreeNode", message);
		sprintf(message, "SPLITE|Point1(%2d,%2d)|Point2(%2d,%2d)|cutPoint1:%2d|cutPoint2:%2d",
			x_1, y_1, x_2, y_2, (*root_)->cutPoint[0], (*root_)->cutPoint[1]);
		Message("learnDQTreeNode", message);
	}
	std::vector<OneSample* > posXLeft;
	std::vector<OneSample* > posXRight;
	std::vector<OneSample* > negXLeft;
	std::vector<OneSample* > negXRight;
	for (int i = 0; i < posX.size(); i++)
	{
		unsigned char fea = (unsigned char)npdFea.NPDtable[(posX[i]->grayImage)[y_1][x_1]]
			[(posX[i]->grayImage)[y_2][x_2]];
		if (fea<(*root_)->cutPoint[0] || fea>(*root_)->cutPoint[1])
		{
			posXLeft.push_back(posX[i]);
		}
		else
		{
			posXRight.push_back(posX[i]);
		}
	}
	//
	for (int i = 0; i < negX.size(); i++)
	{
		unsigned char fea = (unsigned char)npdFea.NPDtable[(negX[i]->grayImage)[y_1][x_1]]
			[(negX[i]->grayImage)[y_2][x_2]];
		if (fea<(*root_)->cutPoint[0] || fea>(*root_)->cutPoint[1])
		{
			negXLeft.push_back(negX[i]);
		}
		else
		{
			negXRight.push_back(negX[i]);
		}
	}
	learnDQTreeNode(&((*root_)->leftRoot), posXLeft, negXLeft, fit[0], depth + 1, config);
	learnDQTreeNode(&((*root_)->rightRoot), posXRight, negXRight, fit[1], depth + 1, config);
}

DQTreeNode::DQTreeNode()
	:featureID(), leftRoot(NULL), rightRoot(NULL),fit(0.0)
{
}

DQTree::DQTree(Samples & pos, Samples & neg ,int stageId_,
	int treeId_, const Configuration & config) :
	root(NULL), threshold(0.0), treeId(treeId_),stageId(stageId_)
{
	learnDQTree(pos.samplePtrs, neg.samplePtrs, 0.0 ,config);
	
}

DQTree::DQTree(std::string treeFilePath,int stageId_,int treeId_)
	:stageId(stageId_),treeId(treeId_), threshold(0.0)
{
	read(treeFilePath);
}






void DQTree::learnDQTree(std::vector<OneSample*>& posX, std::vector<OneSample*>& negX,
	float paretFit, const Configuration & config)
{
	learnDQTreeNode(&root, posX, negX, 0.0, 0, config);
	learnThreshold(posX);
}

void DQTree::learnThreshold(std::vector<OneSample*> posX)
{
	float minFit = FLT_MAX;
	for (int i = 0; i <posX.size(); i++)
	{
		if (posX[i]->fit < minFit)
		{
			minFit = posX[i]->fit;
		}
	}
	threshold = minFit;
	char message[256];
	sprintf(message, "DQTree threshold:%f", minFit);
	Message("DQTree::learnThreshol", message);
}
/*
*
*
*
*/
static void readNode(DQTreeNode** root, FILE* treeDefineFile)
{
	*root = new DQTreeNode();

	fscanf(treeDefineFile, "%d ", &((*root)->nodeType));
	if ((*root)->nodeType == SPLIT)
	{
		int left_cut, right_cut;
		fscanf(treeDefineFile, "%d %d (%d,%d) (%d,%d)\n",
			&(left_cut), &(right_cut),&((*root)->featureID.x_1), 
			&((*root)->featureID.y_1),&((*root)->featureID.x_2),
			&((*root)->featureID.y_2));
		(*root)->cutPoint[0] = (unsigned char)(left_cut);
		(*root)->cutPoint[1] = (unsigned char)(right_cut);
	}
	else if ((*root)->nodeType == LEAF)
	{
		fscanf(treeDefineFile, "%f\n", &((*root)->fit));
		return;
	}
	readNode(&((*root)->leftRoot), treeDefineFile);
	readNode(&((*root)->rightRoot), treeDefineFile);
}
void DQTree::read(const std::string treeFilePath)
{
	FILE * treeFile = fopen(treeFilePath.c_str(), "r");
	if (treeFile == NULL)
	{
		Message("DQTree::read", "read tree define file fail:" + treeFilePath);
		exit(1);
	}
	
	fscanf(treeFile, "%f\n", &threshold);
	readNode(&root,treeFile);
	fclose(treeFile);
}


void DQTree::write(const std::string treeFilePath)
{
	
	char file_name[256];
	sprintf(file_name, "stage_%d_tree%d.npd", stageId, treeId);
	std::ofstream treeFile(std::string(treeFilePath + file_name).c_str());
	treeFile << setiosflags(std::ios::fixed) << std::setprecision(STORAGE_ACCURACY) << threshold+ STORAGE_ACCURACY_CARRY << std::endl;
	writeNode(root, &treeFile);
	treeFile.close();
}

bool DQTree::destoryDQTree(DQTree & root)
{
	return false;
}
