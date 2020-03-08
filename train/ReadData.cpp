#include "ReadData.h"
#include <string>
#include <sys/stat.h>
#ifdef _OPENMP
#include <omp.h>
#endif
Samples::Samples(const char * samplesPath,int size,const char *samplesName):
width(SAMPLE_SIZE),height(SAMPLE_SIZE)
{
	Message("Samples::Samples", "Constructing " + std::string(samplesName));
	int sampleIdex = 0;
#ifdef _OPENMP
omp_set_num_threads(16);
#pragma omp parallel for
#endif
	for (sampleIdex = 0; sampleIdex < size; sampleIdex++)
	{
		char imageName[256];
		sprintf(imageName, "%s%d.jpg", samplesPath, sampleIdex);
		OneSample sample(imageName);
#ifdef _OPENMP
#pragma omp critical
#endif
		if (sample.useable)
		{
			samples.push_back(sample);
			if (samples.size() % 1000 == 0)
			{
				char message[256];
				sprintf(message, "Reading data complete:%f%%", samples.size() / float(size)*100.0);
				Message("Samples::Samples", message);
			}
		}
		
	}
	Message("Samples::Samples", "Constructing samples\'s pointer vector" );
	//create ptrs
	for (sampleIdex = 0; sampleIdex < size; sampleIdex++)
	{
		samplePtrs.push_back(&(samples[sampleIdex]));
	}
	Message("Samples::Samples", "Constructing completed!");
	initialSize = samples.size();
}


void Samples::removeRejectSamples()
{
	
	Message("Samples::removeRejectSamples", "Remove rejected samples!");
	for (std::vector<OneSample>::iterator iter = samples.begin();
		iter != samples.end();)
	{
		if (iter->reject)
		{
			iter = samples.erase(iter);
			
		}
		else
		{
			iter++;
		}
	}
	samplePtrs.clear();
	for (int i = 0; i < samples.size(); i++)
	{
		samplePtrs.push_back(&(samples[i]));
	}
}

void Samples::removeRejectSamples(float threshold)
{
	for (std::vector<OneSample>::iterator iter = samples.begin();
		iter != samples.end();)
	{
		if (iter->fit<threshold)
		{
			iter = samples.erase(iter);
		}
		else
		{
			iter++;
		}
	}
	samplePtrs.clear();
	for (int i = 0; i < samples.size(); i++)
	{
		samplePtrs.push_back(&(samples[i]));
	}
	{
		char message[256];
		sprintf(message, "Remove %d samples | %d samples(%d samples ptr) remine", initialSize- samples.size(),samples.size(),samplePtrs.size() );
		Message("Samples::removeRejectSamples", message);
	}
}
