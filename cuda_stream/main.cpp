#include "gpssim.h"
#include <stdio.h>
#include <complex>
#include "kernal.h"
#include <string.h>


int main()
{
	const char* rfile= "..//brdc2017_0660.17n";
	const char* tfile = "..//stable_.csv";
	FILE* fp = NULL,*fpw=NULL;
	fp = fopen("gpssim_.bin", "wb+");
	fpw = fopen("gpssim.bin", "wb+");


	typedef std::complex<float> complexf;
	float runTime = 0.0f;
	int samp_freq = 120e6;
	int simu_time = 10; 
	int time_all = 300 ;  
	int buff_size = (samp_freq * simu_time / 10);
	complexf* buff = new complexf[buff_size];
	int* qua_buff = (int*)malloc(sizeof(int) * buff_size / 5);
	int ibit = 0;
	transfer_parameter tp;
	gpstime_t g0;
	tp.g0.week = -1;
	g0.sec = 10.0;
	tp.xyz = (double(*)[3]) malloc(sizeof(double) * time_all * 3);
	tp.navbit = (char*)malloc(sizeof(char) * MAX_SAT * 1800);
	tp.neph = readRinexNavAll(tp.eph, &tp.ionoutc, rfile);
	readUserMotion(tp.xyz, tfile, time_all);
	float run_time[1000];

	Table GPSL1table;
	int* CAcode;
	CAcode = (int*)malloc(sizeof(int) * MAX_SAT * 1023);
	for (int i = 0; i < MAX_SAT; i++)
	{
		codegen((CAcode + i * 1023), i + 1);
	}
	GPSL1table.CAcode = CAcode;
	//GPSL1table.i_buff = (float*)malloc(2 * sizeof(float) * buff_size);
	checkCuda(cudaHostAlloc(((void**)&GPSL1table.i_buff), 2 * sizeof(float) * buff_size , cudaHostAllocDefault));
	float *dev_i_buff=NULL;
	//checkCuda(cudaHostAlloc((void**)&dev_i_buff, buff_size * sizeof(float) * 2, cudaHostAllocDefault));
	for (int i = 0; i < (int)(time_all / simu_time); i++)
	{
		memset(qua_buff, 0, sizeof(int) * buff_size / 5);
		runTime = gps_sim(&GPSL1table,&tp, buff, buff_size, samp_freq, 0, simu_time,dev_i_buff);
		run_time[i] = runTime;
	}
	double sum_time = 0;
	for (int i = 0; i < (int)(time_all / simu_time); i++)
	{
		sum_time = sum_time + run_time[i];
	}
	sum_time = sum_time / (int)(time_all / simu_time);
	printf("程序产生1s数据平均用时：%f\n\n", sum_time);

	delete[]buff;
	//fclose(feph);
	fclose(fpw);
	fclose(fp);
	free(qua_buff);
	free(tp.navbit);
	free(tp.xyz);
	checkCuda(cudaFreeHost(GPSL1table.i_buff));
	int aa=getchar();
}