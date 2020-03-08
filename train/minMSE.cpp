#include "minMSE.h"

extern NPDfeature npdFea;
void weightHist(float* weightHist_, int* count,
	std::vector<OneSample*> sampleX, int x_1, int y_1, int x_2, int y_2)
{
	memset(weightHist_, 0.0, sizeof(float)*GRAY_LEVEL);
	for (int i = 0; i < sampleX.size(); i++)
	{
		unsigned char fea = (unsigned char)npdFea.NPDtable[(sampleX[i]->grayImage)[y_1][x_1]]
			[(sampleX[i]->grayImage)[y_2][x_2]];
		weightHist_[fea] += sampleX[i]->weight;
		count[fea]++;

	}
}

void findMinMSE(DQTreeNode ** root_, float* fit, float *mse, int *thr0, int * thr1,
	float* fit0, float* fit1, float minCost)
{
	int minIndex = 0;
	
	for (int i = 0; i < FEATURE_SIZE; i++)
	{
		if (mse[i] < minCost && thr0[i] != -1)
		{
			minCost = mse[i];
			minIndex = i;

		}

	}
	(*root_)->featureID.x_1 = npdFea.feaPoints[minIndex * 4 + 0];
	(*root_)->featureID.y_1 = npdFea.feaPoints[minIndex * 4 + 1];
	(*root_)->featureID.x_2 = npdFea.feaPoints[minIndex * 4 + 2];
	(*root_)->featureID.y_2 = npdFea.feaPoints[minIndex * 4 + 3];
	(*root_)->cutPoint[0] = (unsigned char)thr0[minIndex];
	(*root_)->cutPoint[1] = (unsigned char)thr1[minIndex];
	(*root_)->nodeType = SPLIT;
	fit[0] = fit0[minIndex];
	fit[1] = fit1[minIndex];
}
void cpuMSE(DQTreeNode ** root_, std::vector<OneSample*>& posX,
	std::vector<OneSample*>& negX, const Configuration & config, float *fit, float minCost)
{
	int feaId = 0;
	float* mse = new float[FEATURE_SIZE];
	float* fit0 = new float[FEATURE_SIZE];
	float* fit1 = new float[FEATURE_SIZE];
	int* thr0 = new int[FEATURE_SIZE];
	int* thr1 = new int[FEATURE_SIZE];

#ifdef _OPENMP
	omp_set_num_threads(16);
#pragma omp parallel for 
#endif
	for (feaId = 0; feaId < FEATURE_SIZE; feaId++)
	{
		int x_1 = npdFea.feaPoints[feaId * 4 + 0];
		int y_1 = npdFea.feaPoints[feaId * 4 + 1];
		int x_2 = npdFea.feaPoints[feaId * 4 + 2];
		int y_2 = npdFea.feaPoints[feaId * 4 + 3];
		int count[256];
		float posWHist[256];
		float negWHist[256];
		memset(count, 0, 256 * sizeof(int));
		weightHist(posWHist, count, posX, x_1, y_1, x_2, y_2);
		weightHist(negWHist, count, negX, x_1, y_1, x_2, y_2);
		/***********************************************************/
		//copy from source code
		float posWSum = 0.0;
		float negWSum = 0.0;
		for (int bin = 0; bin < 256; bin++)
		{
			posWSum += posWHist[bin];
			negWSum += negWHist[bin];
		}

		int totalCount = posX.size() + negX.size();
		float wSum = posWSum + negWSum;
		float minMSE = FLT_MAX;
		int localThr0 = -1, localThr1;
		float localFit0 = FLT_MAX, localFit1 = FLT_MAX;

		for (int v = 0; v < 256; v++) // lower threshold
		{
			int rightCount = 0;
			float rightPosW = 0;
			float rightNegW = 0;
			for (int u = v; u < 256; u++) // upper threshold
			{
				rightCount += count[u];
				rightPosW += posWHist[u];
				rightNegW += negWHist[u];
				if (rightCount < config.minLeaf)
				{
					continue;
				}
				int leftCount = totalCount - rightCount;
				if (leftCount < config.minLeaf)
				{
					break;
				}
				float leftPosW = posWSum - rightPosW;
				float leftNegW = negWSum - rightNegW;
				float leftFit, rightFit;
				if (leftPosW + leftNegW <= 0)
				{
					leftFit = 0.0f;
				}
				else
				{
					leftFit = (leftPosW - leftNegW) / (leftPosW + leftNegW);
				}

				if (rightPosW + rightNegW <= 0)
				{
					rightFit = 0.0f;
				}
				else
				{
					rightFit = (rightPosW - rightNegW) / (rightPosW + rightNegW);
				}
				float leftMSE = leftPosW * (leftFit - 1) * (leftFit - 1) +
					leftNegW * (leftFit + 1) * (leftFit + 1);
				float rightMSE = rightPosW * (rightFit - 1) * (rightFit - 1) +
					rightNegW * (rightFit + 1) * (rightFit + 1);
				float localMSE = leftMSE + rightMSE;
				if (localMSE < minMSE)
				{
					minMSE = localMSE;
					localThr0 = v;
					localThr1 = u;
					localFit0 = leftFit;
					localFit1 = rightFit;
				}
			}
		}
		/*
		#pragma omp critical // modify the record by a single thread
		if (minMSE < minCost)
		{
		{
		minCost = minMSE;
		(*root_)->featureID.x_1 = npdFea.feaPoints[feaId * 4 + 0];
		(*root_)->featureID.y_1 = npdFea.feaPoints[feaId * 4 + 1];
		(*root_)->featureID.x_2 = npdFea.feaPoints[feaId * 4 + 2];
		(*root_)->featureID.y_2 = npdFea.feaPoints[feaId * 4 + 3];
		(*root_)->cutPoint[0] = (unsigned char)thr0;
		(*root_)->cutPoint[1] = (unsigned char)thr1;
		(*root_)->nodeType = SPLIT;
		fit[0] = localFit0;
		fit[1] = localFit1;
		}
		}*/
		
		mse[feaId] = minMSE;
		fit0[feaId] = localFit0;
		fit1[feaId] = localFit1;
		thr0[feaId] = localThr0;
		thr1[feaId] = localThr1;

	}
	/***********************************************************/
	findMinMSE(root_, fit, mse, thr0, thr1, fit0, fit1, minCost);
	delete mse;
	delete fit0;
	delete fit1;
	delete thr0;
	delete thr1;
	/***********************************************************/
}


