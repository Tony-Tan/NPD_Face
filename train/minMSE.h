#ifndef NPDFACE_MINMSE_H
#define NPDFACE_MINMSE_H
#include "Configuration.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "NPDFeature.h"
#include <vector>
#include "ReadData.h"
#include "DQTree.h"
#include "Configuration.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

void cpuMSE(DQTreeNode ** root_, std::vector<OneSample*>& posX,
	std::vector<OneSample*>& negX, const Configuration & config,
	float *fit, float minCost);
void gpuMSE(DQTreeNode ** root_, std::vector<OneSample*>& posX,
	std::vector<OneSample*>& negX, const Configuration & config,
	float *fit, float minCost);
extern NPDfeature npdFea;
void weightHist(float* weightHist_, int* count,
	std::vector<OneSample*> sampleX, int x_1, int y_1, int x_2, int y_2);
void findMinMSE(DQTreeNode ** root_, float* fit, float *mse, int *thr0, int * thr1,
	float* fit0, float* fit1, float minCost);
#endif//NPDFACE_MINMSE_H