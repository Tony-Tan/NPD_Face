#include "minMSE.h"


__global__ void mseKernel(unsigned char * posXGrayImage, unsigned char * negXGrayImage, unsigned char *feaPoints,
	unsigned char *NPDtable, float *posXweight, float *negXweight, int posX_size, int negX_size, int minLeaf,
	float *mse, float *fit0, float * fit1, int *thr0, int *thr1)
{
	const unsigned int  feaId = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	unsigned char  x_1 = feaPoints[feaId * 4 + 0];
	unsigned char y_1 = feaPoints[feaId * 4 + 1];
	unsigned char x_2 = feaPoints[feaId * 4 + 2];
	unsigned char y_2 = feaPoints[feaId * 4 + 3];
	int count[256];
	float posWHist[256];
	float negWHist[256];
	for (int i = 0; i < 256; i++)
	{
		count[i] = 0;
		posWHist[i] = 0;
		negWHist[i] = 0;
	}
	for (int i = 0; i < posX_size; i++)
	{
		unsigned char *image = posXGrayImage + i*SAMPLE_SIZE*SAMPLE_SIZE;
		unsigned char fea = NPDtable[image[y_1*SAMPLE_SIZE + x_1] * GRAY_LEVEL +
			image[y_2*SAMPLE_SIZE + x_2]];
		posWHist[fea] += posXweight[i];
		count[fea]++;

	}
	for (int i = 0; i < negX_size; i++)
	{
		unsigned char * image = negXGrayImage + i*SAMPLE_SIZE*SAMPLE_SIZE;
		unsigned char fea = NPDtable[image[y_1*SAMPLE_SIZE + x_1] * GRAY_LEVEL +
			image[y_2*SAMPLE_SIZE + x_2]];
		negWHist[fea] += negXweight[i];
		count[fea]++;

	}

	float posWSum = 0.0;
	float negWSum = 0.0;
	for (int bin = 0; bin < 256; bin++)
	{
		posWSum += posWHist[bin];
		negWSum += negWHist[bin];
	}

	int totalCount = posX_size + negX_size;
	float wSum = posWSum + negWSum;
	float minMSE = FLT_MAX;
	int localThr0 = -1, localThr1;
	float localFit0, localFit1;

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
			if (rightCount < minLeaf)
			{
				continue;
			}
			int leftCount = totalCount - rightCount;
			if (leftCount < minLeaf)
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
	if (localThr0 == -1)
	{
		mse[feaId] = FLT_MAX;
	}
	else
	{
		mse[feaId] = minMSE;
		fit0[feaId] = localFit0;
		fit1[feaId] = localFit1;
		thr0[feaId] = localThr0;
		thr1[feaId] = localThr1;
	}

}

void gpuMSE(DQTreeNode ** root_, std::vector<OneSample*>& posX,
	std::vector<OneSample*>& negX, const Configuration & config,
	float *fit, float minCost)
{
	/**********************************************************************************************/
	uchar * posXArray_local = new unsigned char[SAMPLE_SIZE*SAMPLE_SIZE*posX.size()];
	float * posXWeight_local = new float[posX.size()];
	for (int i = 0; i < posX.size(); i++)
	{
		memcpy(posXArray_local + SAMPLE_SIZE*SAMPLE_SIZE*i, posX[i]->grayImage,
			sizeof(uchar)*SAMPLE_SIZE*SAMPLE_SIZE);
		posXWeight_local[i] = posX[i]->weight;
	}
	
	//
	uchar * posXArray_dev = NULL;
	cudaMalloc((void **)&posXArray_dev, sizeof(uchar)*SAMPLE_SIZE*SAMPLE_SIZE*posX.size());
	cudaMemcpy(posXArray_dev, posXArray_local,
		sizeof(uchar)*SAMPLE_SIZE*SAMPLE_SIZE*posX.size(), cudaMemcpyHostToDevice);
	//
	float * posXWeight_dev = NULL;
	cudaMalloc((void **)&posXWeight_dev, sizeof(float)*posX.size());
	cudaMemcpy(posXWeight_dev, posXWeight_local,
		sizeof(float)*posX.size(), cudaMemcpyHostToDevice);
	/**********************************************************************************************/
	uchar * negXArray_local = new unsigned char[SAMPLE_SIZE*SAMPLE_SIZE*negX.size()];
	float * negXWeight_local = new float[negX.size()];
	for (int i = 0; i < negX.size(); i++)
	{
		memcpy(negXArray_local + SAMPLE_SIZE*SAMPLE_SIZE*i, negX[i]->grayImage,
			sizeof(uchar)*SAMPLE_SIZE*SAMPLE_SIZE);
		negXWeight_local[i] = negX[i]->weight;
	}
	//for (int pixel_i = 0; pixel_i < SAMPLE_SIZE*SAMPLE_SIZE; pixel_i++)
	//{
	//	printf("pixel1:%d pixel2:%d\n", (int)(negXArray_local + SAMPLE_SIZE*SAMPLE_SIZE*(negX.size() - 1))[pixel_i],
	//		(int)(negX[(negX.size() - 1)]->grayImage)[0][pixel_i]);
	//}
	//
	uchar * negXArray_dev = NULL;
	cudaMalloc((void **)&negXArray_dev, sizeof(uchar)*SAMPLE_SIZE*SAMPLE_SIZE*negX.size());
	cudaMemcpy(negXArray_dev, negXArray_local,
		sizeof(uchar)*SAMPLE_SIZE*SAMPLE_SIZE*negX.size(), cudaMemcpyHostToDevice);
	//
	float * negXWeight_dev = NULL;
	cudaMalloc((void **)&negXWeight_dev, sizeof(float)*negX.size());
	cudaMemcpy(negXWeight_dev, negXWeight_local,
		sizeof(float)*negX.size(), cudaMemcpyHostToDevice);
	/**********************************************************************************************/
	//
	uchar *feaPoints_dev = NULL;
	cudaMalloc((void **)&feaPoints_dev, sizeof(uchar)*FEATURE_SIZE*4);
	cudaMemcpy(feaPoints_dev, npdFea.feaPoints,
		sizeof(uchar)*FEATURE_SIZE*4, cudaMemcpyHostToDevice);
	//
	uchar *NPDtable_dev = NULL;
	cudaMalloc((void **)&NPDtable_dev, sizeof(uchar)*GRAY_LEVEL*GRAY_LEVEL);
	cudaMemcpy(NPDtable_dev, npdFea.NPDtable,
		sizeof(uchar)*GRAY_LEVEL*GRAY_LEVEL, cudaMemcpyHostToDevice);
	//unsigned char * posXGrayImage,unsigned char * negXGrayImage, unsigned char *feaPoints,
	//	unsigned char *NPDtable, float *posXweight,float *negXweight, int posX_size, int negX_size, int minLeaf,
	//	float *mse, float *fit0, float * fit1, float *thr0, float *thr1

	float *mse_dev = NULL;
	cudaMalloc((void **)&mse_dev, sizeof(float)*FEATURE_SIZE);
	float *fit0_dev = NULL;
	cudaMalloc((void **)&fit0_dev, sizeof(float)*FEATURE_SIZE);
	float * fit1_dev = NULL;
	cudaMalloc((void **)&fit1_dev, sizeof(float)*FEATURE_SIZE);
	int *thr0_dev = NULL;
	cudaMalloc((void **)&thr0_dev, sizeof(int)*FEATURE_SIZE);
	int *thr1_dev = NULL;
	cudaMalloc((void **)&thr1_dev, sizeof(int)*FEATURE_SIZE);


	//dim3 thread_rect(SAMPLE_SIZE/2, SAMPLE_SIZE);
	//dim3 block_rect();
	mseKernel<<<SAMPLE_SIZE*SAMPLE_SIZE - 1, SAMPLE_SIZE*SAMPLE_SIZE / 2 >>>
		(posXArray_dev, negXArray_dev, feaPoints_dev, NPDtable_dev, posXWeight_dev,
		negXWeight_dev,posX.size(), negX.size(),  config.minLeaf,mse_dev, fit0_dev,
		fit1_dev, thr0_dev, thr1_dev);
	float* mse = new float[FEATURE_SIZE];
	float* fit0 = new float[FEATURE_SIZE];
	float* fit1 = new float[FEATURE_SIZE];
	int* thr0 = new int[FEATURE_SIZE];
	int* thr1 = new int[FEATURE_SIZE];

	cudaMemcpy(mse, mse_dev, sizeof(float)*FEATURE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(fit0, fit0_dev, sizeof(float)*FEATURE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(fit1, fit1_dev, sizeof(float)*FEATURE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(thr0, thr0_dev, sizeof(int)*FEATURE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(thr1, thr1_dev, sizeof(int)*FEATURE_SIZE, cudaMemcpyDeviceToHost);

	
	
	findMinMSE(root_, fit, mse, thr0, thr1, fit0, fit1, minCost);

	delete mse;
	delete fit0;
	delete fit1;
	delete thr0;
	delete thr1;

	delete posXArray_local;
	delete negXArray_local;
	delete posXWeight_local;
	delete negXWeight_local;
	

	cudaFree(posXArray_dev);
	cudaFree(negXArray_dev);
	cudaFree(posXWeight_dev);
	cudaFree(negXWeight_dev);
	cudaFree(feaPoints_dev);
	cudaFree(NPDtable_dev);
	

	cudaFree(mse_dev);
	cudaFree(fit0_dev);
	cudaFree(fit1_dev);
	cudaFree(thr0_dev);
	cudaFree(thr1_dev);


}
