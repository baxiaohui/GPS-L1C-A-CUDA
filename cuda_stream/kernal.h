#ifndef __KERNAL_H_
#define __KERNAL_H_

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "gpssim.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void GPUMemoryInit(Table* Tb, int ch_num);
void produce_samples_withCuda(Table* Tb, channel_t* channel, int fs, float* parameters, float* dev_parameters, int* sum, int* dev_sum, float* dev_i_buff, int satnum, float* dev_noise, double* db_para, double* dev_db_para);
void GPUMemroy_delete(Table* Tb);
void navdata_update(Table* Tb);
void Storedata(channel_t* channel, int count_call, int LoopNumber, int* sum, int satnum, int fs, float* parameters, double* db_para);
void CudaStream(Table* Tb, float* parameters, double* db_para, float* dev_noise, int* sum, int fs, int satnum);
void CudaStream_firstSec(Table* Tb, float* parameters, double* db_para, float* dev_noise, int* sum, int fs, int satnum);
#endif