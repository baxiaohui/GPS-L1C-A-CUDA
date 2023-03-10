#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "kernal.h"
#include "cuda_texture_types.h"//??????ʶ??texture
//__constant__  int sinTable512[] = {
//	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
//	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
//	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
//	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
//	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
//	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
//	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
//	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250,
//	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
//	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
//	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
//	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
//	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
//	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
//	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
//	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
//	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
//	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
//	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
//	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
//	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
//	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
//	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
//	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
//	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
//	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
//	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
//	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
//	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
//	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
//	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
//	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2
//};
//
//__constant__ int cosTable512[] = {
//	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
//	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
//	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
//	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
//	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
//	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
//	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
//	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
//	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
//	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
//	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
//	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
//	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
//	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
//	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
//	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
//	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
//	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
//	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
//	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
//	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
//	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
//	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
//	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2,
//	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
//	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
//	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
//	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
//	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
//	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
//	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
//	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250
//};



//#define SIZE 1024
#define BLOCK_SIZES 1024
#define GRID_SIZES 256
#define L 64

texture<int> t_sinTable;
texture<int> t_cosTable;
texture<int, 1> t_CAcode;
texture<char, 1> t_navbit;
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s error code %d\n", cudaGetErrorString(result), result);
		getchar();
		//assert(result == cudaSuccess);
	}
#endif
	return result;
}



/*
* ????:???뾭??ʹ?õĲ?ѯ???ͱ???
* sin/cos table: texture memory
* pseudorandom code:texture memory
* navigation data:
* i_buff/q_buff:page-locked memory
* amplititude: texture memory
* input: sinTable,cosTable,CAcode
*/
void GPUMemoryInit(Table* Tb, int ch_num)
{
	cudaChannelFormatDesc coschannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaMallocArray(&(Tb->cu_cosTable), &coschannelDesc, 512, 1);
	cudaMemcpyToArray(Tb->cu_cosTable, 0, 0, Tb->cosTable, sizeof(int) * 512, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(t_cosTable, Tb->cu_cosTable);//?????ұ?????Ϊ?????ڴ?

	cudaChannelFormatDesc sinchannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaMallocArray(&(Tb->cu_sinTable), &sinchannelDesc, 512, 1);
	cudaMemcpyToArray(Tb->cu_sinTable, 0, 0, Tb->sinTable, sizeof(int) * 512, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(t_sinTable, Tb->cu_sinTable);//?????ұ?????Ϊ?????ڴ?

	checkCuda(cudaMalloc((void**)&Tb->dev_CAcode, sizeof(int) * 1023 * MAX_SAT));
	cudaMemcpy(Tb->dev_CAcode, Tb->CAcode, sizeof(int) * 1023 * MAX_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_CAcode, Tb->dev_CAcode);//??CA??????Ϊ?????ڴ?

	checkCuda(cudaMalloc((void**)&Tb->dev_navdata, sizeof(char) * 1800 * MAX_SAT));//1800????,һ??֡????5????֡??һ????֡ʮ???֣?һ????30???أ?һ??????ռһ??char,?洢??ʱ??????һ֡?ĵ?????֡Ҳ?????????????????õ?ʱ?򷽱?һ??
	cudaMemcpy(Tb->dev_navdata, Tb->navdata, sizeof(char) * 1800 * MAX_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_navbit, Tb->dev_navdata);//?????????İ???Ϊ?????ڴ?
}

void navdata_update(Table* Tb)
{
	checkCuda(cudaMalloc((void**)&Tb->dev_navdata, sizeof(char) * 1800 * MAX_SAT));
	cudaMemcpy(Tb->dev_navdata, Tb->navdata, sizeof(char) * 1800 * MAX_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_navbit, Tb->dev_navdata);//?????????İ???Ϊ?????ڴ?
}

__global__ void cudaBPSK(float* dev_parameters, float* dev_i_buff, int* dev_sum, float* dev_noise,double *dev_db_para)//carrier_step
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x+1;//ch_num=y,sample_num=x idx????????ʱ??û?ж?????λ?????ۼӣ?ʵ????Ӧ???ۼӵ?
	int idy = threadIdx.y;
	int prn = dev_parameters[idy*p_n+2]-1;//?ڱ???prn?Ǵ?0??ʼ??
	double codephase = (idx * dev_db_para[idy * pd_n + 1] + dev_db_para[idy * pd_n + 0]);
	codephase -= (int)(codephase);
	int CurrentCodePhase = (int)(idx * dev_db_para[idy * pd_n + 1] + dev_db_para[idy * pd_n + 0])%1023 ;//????λ=????????*????λ????+??ʼ????λ ???ܻᳬ??1023????????Ҫ??1023ȡ??
	codephase+= CurrentCodePhase;
	double CarrierPhase = (idx * dev_db_para[idy * pd_n + 3] + dev_db_para[idy * pd_n + 2]); //??λ?ĵ?λ????(2*pi) ?ز???λ=????????*?ز???λ????+??ʼ?ز???λ
	int cph = 0;
	if(CarrierPhase<0)
		 cph = (CarrierPhase - (int)CarrierPhase) * 512+512;//????С??????
	else
		 cph = (CarrierPhase - (int)CarrierPhase) * 512;    //????С??????  ????Ƶ?????³????˸?ֵ
	//int cph = (CarrierPhase >> 16) & 511;
	int temp = (int)(idx * dev_db_para[idy * pd_n + 1] + dev_db_para[idy * pd_n + 0]) / 1023+ dev_parameters[idy * p_n + 4] ;//?õ?ms??
	int ibit = temp / 20+dev_parameters[idy*p_n+0]+ dev_parameters[idy * p_n + 3]*30;//ǰ300bitΪ??һ??֡??????һ????֡????301??ʼ?ǵ?ǰ??֡?ĵ?һ??bit
	__shared__ float memoryi[MAX_CHAN][threadPerBlock];//????GPUÿ?߳̿??ڵĹ????ڴ?Ϊ48KB

	if (idx < dev_sum[0])
	{
		memoryi[idy][threadIdx.x] = dev_parameters[p_n * idy + 1] * tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 1023))\
			* tex1D(t_cosTable, (float)cph)* tex1Dfetch(t_navbit, (ibit + prn * 1800))/250;
	}

	__syncthreads();

	int i = dev_sum[1] / 2;
	while (i !=0)//ֻ????һ?ι?Լ
	{
		if (idy < i)
			memoryi[idy][threadIdx.x] += memoryi[idy + i][threadIdx.x];
		__syncthreads();
		i /= 2;
	}
	//if(idy==1)
	//int num1 = idx * 2;
	if (idy == 0)
	{
		int j = dev_noise[idx];
		dev_i_buff[idx] = memoryi[0][threadIdx.x];  //short((memoryi[0][threadIdx.x] + 64) >> 7)   dev_noise[idx]
		//dev_i_buff[idx] = dev_noise[idx];
	}

	//??????һ·????ֵ
	if (idx < dev_sum[0])
	{
		memoryi[idy][threadIdx.x] = dev_parameters[p_n * idy + 1] * tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 1023))\
			* tex1D(t_sinTable, (float)cph) * tex1Dfetch(t_navbit, (ibit + prn * 1800))/250;
	}
	__syncthreads();
	i = dev_sum[1] / 2;
	while (i !=0)
	{
		if (idy < i)
			memoryi[idy][threadIdx.x] += memoryi[idy + i][threadIdx.x];
		__syncthreads();
		i /= 2;
	}
	//num1 -= 1;
	if (idy == 0)
	{
		int j = dev_noise[idx + dev_sum[0]];
		dev_i_buff[idx + dev_sum[0]] = memoryi[0][threadIdx.x]; //+dev_noise[idx+dev_sum[0]]
		//dev_i_buff[idx + dev_sum[0]] = dev_noise[idx + dev_sum[0]];
	}
}



void produce_samples_withCuda(Table* Tb, channel_t* channel, int fs, float* parameters, float* dev_parameters, int* sum, int* dev_sum, float* dev_i_buff, int satnum, float* dev_noise, double* db_para, double* dev_db_para)
{
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);
	int sat_all = satnum;
	int samples = fs / Rev_fre;
	float* dev_buff;
	for (int j = 0; j < sat_all; j++, channel++)
	{
		if (channel->prn != 0)
		{
			parameters[j * p_n + 0] = channel->ibit;
			parameters[j * p_n + 1] = channel->amp;
			parameters[j * p_n + 2] = channel->prn;
			parameters[j * p_n + 3] = channel->iword;
			parameters[j * p_n + 4] = channel->icode;
			db_para[j * pd_n + 0] = channel->code_phase;
			db_para[j * pd_n + 1] = channel->code_phasestep;
			db_para[j * pd_n + 2] = channel->carr_phase;
			db_para[j * pd_n + 3] = channel->carr_phasestep;
		}
	}

	
	//
	checkCuda(cudaMalloc((void**)&dev_buff, samples * sizeof(float) * 2));
	checkCuda(cudaMemcpy(dev_parameters, parameters, sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_db_para, db_para, sizeof(double) * sat_all * pd_n, cudaMemcpyHostToDevice));

	sum[0] = samples; sum[1] = MAX_CHAN;
  	checkCuda(cudaMemcpy(dev_sum, &sum[0], 2 * sizeof(int), cudaMemcpyHostToDevice));
	float blockPerGridx = fs / Rev_fre / threadPerBlock;
	(blockPerGridx - (int)blockPerGridx == 0) ? blockPerGridx : blockPerGridx = (int)blockPerGridx + 1;
	dim3 block(blockPerGridx, 1);
	dim3 thread(threadPerBlock, satnum);
	cudaBPSK << <block, thread >> > (dev_parameters, dev_buff, dev_sum,dev_noise,dev_db_para);

	checkCuda(cudaMemcpy(Tb->i_buff, dev_buff, sizeof(float) * samples * 2, cudaMemcpyDeviceToHost));

	cudaFree(dev_buff);
}

void Storedata(channel_t* channel,int count_call,int LoopNumber, int* sum, int satnum, int fs, float* parameters, double* db_para)
{
	int sat_all = satnum;
	int samples = fs / Rev_fre;
	sum[0] = samples; 
	sum[1] = MAX_CHAN;
	int step = satnum * p_n;//0.1s?????£?parameters???ݵĳ???

	if (count_call == 0)//?????ǵ?һ??1s?????ݣ?ѭ?????????Ǵ?1??ʼ?ģ?????LoopNumber??Ҫ??һ
	{
		for (int j = 0; j < sat_all; j++, channel++)
		{
			if (channel->prn != 0)
			{
				parameters[j * p_n + 0 + (LoopNumber - 1) * step] = channel->ibit;
				parameters[j * p_n + 1 + (LoopNumber - 1) * step] = channel->amp;
				parameters[j * p_n + 2 + (LoopNumber - 1) * step] = channel->prn;
				parameters[j * p_n + 3 + (LoopNumber - 1) * step] = channel->iword;
				parameters[j * p_n + 4 + (LoopNumber - 1) * step] = channel->icode;
				db_para[j * pd_n + 0 + (LoopNumber - 1) * step] = channel->code_phase;
				db_para[j * pd_n + 1 + (LoopNumber - 1) * step] = channel->code_phasestep;
				db_para[j * pd_n + 2 + (LoopNumber - 1) * step] = channel->carr_phase;
				db_para[j * pd_n + 3 + (LoopNumber - 1) * step] = channel->carr_phasestep;
			}
		}
	}
	else if (count_call > 0)
	{
		for (int j = 0; j < sat_all; j++, channel++)
		{
			if (channel->prn != 0)
			{
				parameters[j * p_n + 0 + LoopNumber * step] = channel->ibit;
				parameters[j * p_n + 1 + LoopNumber * step] = channel->amp;
				parameters[j * p_n + 2 + LoopNumber * step] = channel->prn;
				parameters[j * p_n + 3 + LoopNumber * step] = channel->iword;
				parameters[j * p_n + 4 + LoopNumber * step] = channel->icode;
				db_para[j * pd_n + 0 + LoopNumber * step] = channel->code_phase;
				db_para[j * pd_n + 1 + LoopNumber * step] = channel->code_phasestep;
				db_para[j * pd_n + 2 + LoopNumber * step] = channel->carr_phase;
				db_para[j * pd_n + 3 + LoopNumber * step] = channel->carr_phasestep;
			}
		}
	}
}



//?Գ??˵?һ??1s?????????ݽ???cuda???ļ???
void CudaStream(Table* Tb, float* parameters, double* db_para, float* dev_noise, int* sum, int fs, int satnum)
{
	//?????????Ƿ?֧???豸?ص?????
	cudaDeviceProp prop;
	int whichDevice;
	checkCuda(cudaGetDevice(&whichDevice));
	checkCuda(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap)
	{
		printf("ERROR??Device will not handle overlaps,so no speed up from stream!");
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	//??ʼ??cuda??
	cudaStream_t stream0, stream1, stream2, stream3, stream4;
	checkCuda(cudaStreamCreate(&stream0));
	checkCuda(cudaStreamCreate(&stream1));
	checkCuda(cudaStreamCreate(&stream2));
	checkCuda(cudaStreamCreate(&stream3));
	checkCuda(cudaStreamCreate(&stream4));

	//?????????õ??ı???
	int sat_all = satnum;
	int samples = fs / Rev_fre;
	float* dev_buff0, * dev_buff1, * dev_buff2, * dev_buff3, * dev_buff4;
	int* dev_sum;
	float* dev_parameters0, * dev_parameters1, * dev_parameters2, * dev_parameters3, * dev_parameters4;
	double* dev_db_para0, * dev_db_para1, * dev_db_para2, * dev_db_para3, * dev_db_para4;
	//??GPU?Ϸ????ڴ?
	checkCuda(cudaMalloc((void**)&dev_parameters0, sizeof(float) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_parameters1, sizeof(float) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_parameters2, sizeof(float) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_parameters3, sizeof(float) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_parameters4, sizeof(float) * sat_all * p_n));


	checkCuda(cudaMalloc((void**)&dev_db_para0, sizeof(double) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_db_para1, sizeof(double) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_db_para2, sizeof(double) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_db_para3, sizeof(double) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_db_para4, sizeof(double) * sat_all * p_n));


	checkCuda(cudaMalloc((void**)&dev_sum, sizeof(int) * 2));
	checkCuda(cudaMalloc((void**)&dev_buff0, samples * sizeof(float) * 2));
	checkCuda(cudaMalloc((void**)&dev_buff1, samples * sizeof(float) * 2));
	checkCuda(cudaMalloc((void**)&dev_buff2, samples * sizeof(float) * 2));
	checkCuda(cudaMalloc((void**)&dev_buff3, samples * sizeof(float) * 2));
	checkCuda(cudaMalloc((void**)&dev_buff4, samples * sizeof(float) * 2));

	//ҳ?????ڴ???ǰ???Ѿ??????˷????븳ֵ??parameters??db_para??sum????????????
	//????????????????ʱ???仯?????Բ???Ҫ????ҳ?????ڴ?

	int step = sat_all * p_n;//0.1s???ݣ?dev_parameters?????ݸ???
	for (int i = 0; i < 10; i = i + 5)
	{
		//?????ݴ????????䵽?豸??
		checkCuda(cudaMemcpyAsync(dev_parameters0, parameters + step * i, sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice, stream0));
		checkCuda(cudaMemcpyAsync(dev_parameters1, parameters + step * (i + 1), sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice, stream1));
		checkCuda(cudaMemcpyAsync(dev_parameters2, parameters + step * (i + 2), sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice, stream2));
		checkCuda(cudaMemcpyAsync(dev_parameters3, parameters + step * (i + 3), sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice, stream3));
		checkCuda(cudaMemcpyAsync(dev_parameters4, parameters + step * (i + 4), sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice, stream4));

		checkCuda(cudaMemcpyAsync(dev_db_para0, db_para + step * i, sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice, stream0));
		checkCuda(cudaMemcpyAsync(dev_db_para1, db_para + step * (i + 1), sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice, stream1));
		checkCuda(cudaMemcpyAsync(dev_db_para2, db_para + step * (i + 2), sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice, stream2));
		checkCuda(cudaMemcpyAsync(dev_db_para3, db_para + step * (i + 3), sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice, stream3));
		checkCuda(cudaMemcpyAsync(dev_db_para4, db_para + step * (i + 4), sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice, stream4));

		checkCuda(cudaMemcpyAsync(dev_sum, sum, sizeof(int) * 2, cudaMemcpyHostToDevice, stream0));

		float blockPerGridx = fs / Rev_fre / threadPerBlock;
		(blockPerGridx - (int)blockPerGridx == 0) ? blockPerGridx : blockPerGridx = (int)blockPerGridx + 1;
		dim3 block(blockPerGridx, 1);
		dim3 thread(threadPerBlock, satnum);
		cudaBPSK << <block, thread, 0, stream0 >> > (dev_parameters0, dev_buff0, dev_sum, dev_noise, dev_db_para0);
		cudaBPSK << <block, thread, 0, stream1 >> > (dev_parameters1, dev_buff1, dev_sum, dev_noise, dev_db_para1);
		cudaBPSK << <block, thread, 0, stream2 >> > (dev_parameters2, dev_buff2, dev_sum, dev_noise, dev_db_para2);
		cudaBPSK << <block, thread, 0, stream3 >> > (dev_parameters3, dev_buff3, dev_sum, dev_noise, dev_db_para3);
		cudaBPSK << <block, thread, 0, stream4 >> > (dev_parameters4, dev_buff4, dev_sum, dev_noise, dev_db_para4);

		checkCuda(cudaMemcpyAsync(Tb->i_buff + i * samples * 2, dev_buff0, samples * sizeof(float) * 2, cudaMemcpyDeviceToHost, stream0));
		checkCuda(cudaMemcpyAsync(Tb->i_buff + (i + 1) * samples * 2, dev_buff1, samples * sizeof(float) * 2, cudaMemcpyDeviceToHost, stream1));
		checkCuda(cudaMemcpyAsync(Tb->i_buff + (i + 2) * samples * 2, dev_buff2, samples * sizeof(float) * 2, cudaMemcpyDeviceToHost, stream2));
		checkCuda(cudaMemcpyAsync(Tb->i_buff + (i + 3) * samples * 2, dev_buff3, samples * sizeof(float) * 2, cudaMemcpyDeviceToHost, stream3));
		checkCuda(cudaMemcpyAsync(Tb->i_buff + (i + 4) * samples * 2, dev_buff4, samples * sizeof(float) * 2, cudaMemcpyDeviceToHost, stream4));
	}
	//??????????????CPU??GPU??ͬ??????
	checkCuda(cudaStreamSynchronize(stream0));
	checkCuda(cudaStreamSynchronize(stream1));
	checkCuda(cudaStreamSynchronize(stream2));
	checkCuda(cudaStreamSynchronize(stream3));
	checkCuda(cudaStreamSynchronize(stream4));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elasptime;
	cudaEventElapsedTime(&elasptime, start, stop);//????cuda????ʱ??
	printf("      \ntime=%.5f s\n", elasptime / 1000);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//?ͷ??????ڴ?(?????ڴ????ͷŵ?????)
	checkCuda(cudaFree(dev_parameters0));
	checkCuda(cudaFree(dev_parameters1));
	checkCuda(cudaFree(dev_parameters2));
	checkCuda(cudaFree(dev_parameters3));
	checkCuda(cudaFree(dev_parameters4));
	checkCuda(cudaFree(dev_db_para0));
	checkCuda(cudaFree(dev_db_para1));
	checkCuda(cudaFree(dev_db_para2));
	checkCuda(cudaFree(dev_db_para3));
	checkCuda(cudaFree(dev_db_para4));
	checkCuda(cudaFree(dev_sum));
	checkCuda(cudaFree(dev_buff0));
	checkCuda(cudaFree(dev_buff1));
	checkCuda(cudaFree(dev_buff2));
	checkCuda(cudaFree(dev_buff3));
	checkCuda(cudaFree(dev_buff4));
	checkCuda(cudaStreamDestroy(stream0));
	checkCuda(cudaStreamDestroy(stream1));
	checkCuda(cudaStreamDestroy(stream2));
	checkCuda(cudaStreamDestroy(stream3));
	checkCuda(cudaStreamDestroy(stream4));


}


//??Ϊ??һ??1s??????ֻ????0.2s-1s??9??0.1s?????ݣ???????һ??1s????????10??0.1s???ݲ?ͬ???ʵ???дһ??????????????
void CudaStream_firstSec(Table* Tb, float* parameters, double* db_para, float* dev_noise, int* sum, int fs, int satnum)
{
	//?????????Ƿ?֧???豸?ص?????
	cudaDeviceProp prop;
	int whichDevice;
	checkCuda(cudaGetDevice(&whichDevice));
	checkCuda(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap)
	{
		printf("ERROR??Device will not handle overlaps,so no speed up from stream!");
	}
	
	//??ʼ??cuda??????Ϊ??9???????????Բ???3????????
	cudaStream_t stream0, stream1,stream2;
	checkCuda(cudaStreamCreate(&stream0));
	checkCuda(cudaStreamCreate(&stream1));
	checkCuda(cudaStreamCreate(&stream2));

	//?????????õ??ı???
	int sat_all = satnum;
	int samples = fs / Rev_fre;
	float* dev_buff0, * dev_buff1, * dev_buff2;
	int* dev_sum;
	float* dev_parameters0, * dev_parameters1, * dev_parameters2;
	double* dev_db_para0, * dev_db_para1, * dev_db_para2;

	//??GPU?Ϸ????ڴ?
	checkCuda(cudaMalloc((void**)&dev_parameters0, sizeof(float) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_parameters1, sizeof(float) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_parameters2, sizeof(float) * sat_all * p_n));

	checkCuda(cudaMalloc((void**)&dev_db_para0, sizeof(double) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_db_para1, sizeof(double) * sat_all * p_n));
	checkCuda(cudaMalloc((void**)&dev_db_para2, sizeof(double) * sat_all * p_n));


	checkCuda(cudaMalloc((void**)&dev_sum, sizeof(int) * 2));
	checkCuda(cudaMalloc((void**)&dev_buff0, samples * sizeof(float) * 2));
	checkCuda(cudaMalloc((void**)&dev_buff1, samples * sizeof(float) * 2));
	checkCuda(cudaMalloc((void**)&dev_buff2, samples * sizeof(float) * 2));

	//ҳ?????ڴ???ǰ???Ѿ??????˷????븳ֵ??parameters??db_para??sum????????????
	//????????????????ʱ???仯?????Բ???Ҫ????ҳ?????ڴ?

	int step = sat_all * p_n;//0.1s???ݣ?dev_parameters?????ݸ???
	for (int i = 0; i < 9; i = i + 3)
	{
		//?????ݴ????????䵽?豸??
		checkCuda(cudaMemcpyAsync(dev_parameters0, parameters + step * i, sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice, stream0));
		checkCuda(cudaMemcpyAsync(dev_parameters1, parameters + step * (i + 1), sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice, stream1));
		checkCuda(cudaMemcpyAsync(dev_parameters2, parameters + step * (i + 2), sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice, stream2));

		checkCuda(cudaMemcpyAsync(dev_db_para0, db_para + step * i, sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice, stream0));
		checkCuda(cudaMemcpyAsync(dev_db_para1, db_para + step * (i + 1), sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice, stream1));
		checkCuda(cudaMemcpyAsync(dev_db_para2, db_para + step * (i + 2), sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice, stream2));

		checkCuda(cudaMemcpyAsync(dev_sum, sum, sizeof(int) * 2, cudaMemcpyHostToDevice, stream0));

		float blockPerGridx = fs / Rev_fre / threadPerBlock;
		(blockPerGridx - (int)blockPerGridx == 0) ? blockPerGridx : blockPerGridx = (int)blockPerGridx + 1;
		dim3 block(blockPerGridx, 1);
		dim3 thread(threadPerBlock, satnum);
		cudaBPSK << <block, thread, 0, stream0 >> > (dev_parameters0, dev_buff0, dev_sum, dev_noise, dev_db_para0);
		cudaBPSK << <block, thread, 0, stream1 >> > (dev_parameters1, dev_buff1, dev_sum, dev_noise, dev_db_para1);
		cudaBPSK << <block, thread, 0, stream2 >> > (dev_parameters2, dev_buff2, dev_sum, dev_noise, dev_db_para2);

		checkCuda(cudaMemcpyAsync(Tb->i_buff + i * samples * 2, dev_buff0, samples * sizeof(float) * 2, cudaMemcpyDeviceToHost, stream0));
		checkCuda(cudaMemcpyAsync(Tb->i_buff + (i + 1) * samples * 2, dev_buff1, samples * sizeof(float) * 2, cudaMemcpyDeviceToHost, stream1));
		checkCuda(cudaMemcpyAsync(Tb->i_buff + (i + 2) * samples * 2, dev_buff2, samples * sizeof(float) * 2, cudaMemcpyDeviceToHost, stream2));
	}
	//??????????????CPU??GPU??ͬ??????
	checkCuda(cudaStreamSynchronize(stream0));
	checkCuda(cudaStreamSynchronize(stream1));
	checkCuda(cudaStreamSynchronize(stream2));


	//?ͷ??????ڴ?(?????ڴ????ͷŵ?????)
	checkCuda(cudaFree(dev_parameters0));
	checkCuda(cudaFree(dev_parameters1));
	checkCuda(cudaFree(dev_parameters2));
	checkCuda(cudaFree(dev_db_para0));
	checkCuda(cudaFree(dev_db_para1));
	checkCuda(cudaFree(dev_db_para2));
	checkCuda(cudaFree(dev_sum));
	checkCuda(cudaFree(dev_buff0));
	checkCuda(cudaFree(dev_buff1));
	checkCuda(cudaFree(dev_buff2));
	checkCuda(cudaStreamDestroy(stream0));
	checkCuda(cudaStreamDestroy(stream1));
	checkCuda(cudaStreamDestroy(stream2));
}



















void GPUMemroy_delete(Table* Tb)
{
	cudaUnbindTexture(t_sinTable);
	cudaUnbindTexture(t_cosTable);
	cudaUnbindTexture(t_CAcode);
	cudaFree(Tb->dev_CAcode);
	cudaFreeArray(Tb->cu_cosTable);
	cudaFreeArray(Tb->cu_sinTable);
	//free(Tb->i_buff);
}
