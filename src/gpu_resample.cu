#include <stdlib.h>
#include <stdio.h>

#include "stdefs.h"
#include "resample.h"

#define CUDA_CHECK_RETURN(value) {          \
    cudaError_t _m_cudaStat = value;        \
    if (_m_cudaStat != cudaSuccess) {       \
        fprintf(stderr, 					\
		"Error %s at line %d in file %s\n",	\
        cudaGetErrorString(_m_cudaStat), 	\
		__LINE__, __FILE__);       			\
        exit(1);                            \
    } }

WORD 	*d_Vs;
HWORD 	*d_Imp;
HWORD	*d_ImpD;
HWORD	*d_X;
HWORD	*d_TandP, *d_TxorP;
size_t	*d_indices;

HWORD	*TandP, *TxorP;

WORD	*Vs;
size_t	*indices;

const int BLOCKS = 256, THREADS = 256;

__global__ void kernel_FilterUp(
	HWORD d_Imp[], HWORD d_ImpD[], UHWORD Nwing,  
	HWORD *d_X, HWORD *d_TandP, HWORD *d_TxorP, size_t *d_indices, 
	WORD *d_Vs, int g_count, UHWORD LpScl) {
	/* 1-Dimentional thread blocks */
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int t_num = blockDim.x * gridDim.x;
	HWORD *Hp, *Hdp = NULL, *End;
	HWORD a = 0;
	WORD t, v = 0;
	HWORD Ph;
	HWORD Inc;

	for (int i = idx; i < g_count; i += t_num) {
		HWORD const *Xp = &d_X[d_indices[i]];
		Inc = -1;
		Ph = d_TandP[i];
		Hp = &d_Imp[Ph>>Na];
		End = &d_Imp[Nwing];
		// Branch-condition: Interp
		Hdp = &d_ImpD[Ph>>Na];
		a = Ph & Amask;
		// Branch-condition: Inc(Right-wing)
		// End--;
		while (Hp < End) {
			t = *Hp;
			t += (((WORD)*Hdp)*a)>>Na;
			Hdp += Npc;
			t *= *Xp;
			if (t & (1<<(Nhxn-1)))
				t += (1<<(Nhxn-1));
			t >>= Nhxn;
			v += t;
			Hp += Npc;
			Xp += Inc;
		}
		d_Vs[i] = v;
	
		v = 0;
		Inc = 1;
		Ph = d_TxorP[i];
		Hp = &d_Imp[Ph>>Na];
		End = &d_Imp[Nwing];
		// Branch-cond: Interp
		Hdp = &d_ImpD[Ph>>Na];
		a = Ph & Amask;
		Xp = &d_X[d_indices[i]];
		Xp = Xp+1;
		// Branch-condition: Inc(Right-wing)
		End--;
		if (Ph == 0) {
			Hp += Npc;
			Hdp += Npc;
		}
		while (Hp < End) {
			t = *Hp;
			t += (((WORD)*Hdp)*a)>>Na;
			Hdp += Npc;
			t *= *Xp;
			if (t & (1<<(Nhxn-1)))
				t += (1<<(Nhxn-1));
			t >>= Nhxn;
			v += t;
			Hp += Npc;
			Xp += Inc;
		}
		d_Vs[i] += v;
	
		d_Vs[i] >>= Nhg;
		d_Vs[i] *= LpScl;
	}
}

/* Inlined function defined in resampleisubs.c */
extern "C" HWORD WordToHword(WORD v, int scl);

extern "C" {

void GPU_Init(HWORD Imp[], HWORD ImpD[], UHWORD Nwing, 
			  int IBUFFSIZE, int OBUFFSIZE, double factor) {
	/* Imp[] and ImpD[] */
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Imp, sizeof(HWORD)*Nwing));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_ImpD, sizeof(HWORD)*Nwing));
	CUDA_CHECK_RETURN(cudaMemcpy(d_Imp, Imp, 
		sizeof(HWORD)*Nwing, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_ImpD, ImpD, 
		sizeof(HWORD)*Nwing, cudaMemcpyHostToDevice));

	/* X, Y */
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_X, sizeof(HWORD)*IBUFFSIZE));

	int u_fac = (int)(factor + 1);	
	/* v */
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Vs, sizeof(WORD)*IBUFFSIZE*u_fac));
	Vs = (WORD*)malloc(sizeof(WORD)*IBUFFSIZE*u_fac);

	/* TandP TxorP */
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_TandP,	sizeof(HWORD)*IBUFFSIZE*u_fac));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_TxorP,	sizeof(HWORD)*IBUFFSIZE*u_fac));
	TandP = (HWORD*)malloc(sizeof(HWORD)*IBUFFSIZE*u_fac);
	TxorP = (HWORD*)malloc(sizeof(HWORD)*IBUFFSIZE*u_fac);

	/* *Time>>Np */
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_indices, sizeof(size_t)*IBUFFSIZE*u_fac));
	indices = (size_t*)malloc(sizeof(size_t)*IBUFFSIZE*u_fac);
}

void GPU_Destruct() {
	/* Imp[] and ImpD[] */
	CUDA_CHECK_RETURN(cudaFree(d_Imp));	
	CUDA_CHECK_RETURN(cudaFree(d_ImpD));

	/* X, Y */
	CUDA_CHECK_RETURN(cudaFree(d_X));

	/* v */
	CUDA_CHECK_RETURN(cudaFree(d_Vs));

	/* TandP TxorP */
	CUDA_CHECK_RETURN(cudaFree(d_TandP));
	CUDA_CHECK_RETURN(cudaFree(d_TxorP));
	free(TandP);
	free(TxorP);

	/* *Time>>Np */
	free(indices);
}

int GPU_SrcUP(HWORD X[], HWORD Y[], double factor, UWORD *Time,
			  UHWORD Nx, UHWORD Nwing, UHWORD LpScl,
			  HWORD Imp[], HWORD ImpD[], BOOL Interp, 
			  int IBUFFSIZE, int OBUFFSIZE) {
	double dt;
	UWORD dtb;

	dt = 1.0/factor;
	dtb = dt*(1<<Np) + 0.5;

#ifdef GPU_DEBUG
	FILE *fp1 = fopen("./verify/gpu_TandP.txt", "a");
	FILE *fp2 = fopen("./verify/gpu_TxorP.txt", "a");
	FILE *fp3 = fopen("./verify/gpu_X.txt", "a");
	FILE *fp4 = fopen("./verify/gpu_indices.txt", "a");
	FILE *fp5 = fopen("./verify/gpu_Y.txt", "a");
	FILE *fp6 = fopen("./verify/gpu_v.txt", "a");
#endif
	
	/* GPU need */
	int g_count = ((1<<Np)*(WORD)Nx + dtb - 1) / dtb;
	int realTime;
	for (int i = 0; i < g_count; ++i) {
		realTime = *Time + i * dtb;
		indices[i] = realTime>>Np;
		TandP[i] = (HWORD)(realTime&Pmask);
		TxorP[i] = (HWORD)(((realTime^Pmask)+1)&Pmask);
#ifdef GPU_DEBUG
		fprintf(fp1, "%hd\n", TandP[i]);
		fprintf(fp2, "%hd\n", TxorP[i]);
		fprintf(fp4, "%zu\n", indices[i]);
#endif
	}
#ifdef GPU_DEBUG
	for (int i = 0; i < IBUFFSIZE; ++i)
		fprintf(fp3, "%hd\n", X[i]);
	fprintf(fp1, "\n");
	fprintf(fp2, "\n");
	fprintf(fp3, "\n");
	fprintf(fp4, "\n");
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);
	fclose(fp4);
#endif

	*Time += g_count * dtb;

	CUDA_CHECK_RETURN(cudaMemcpy(d_indices, indices, sizeof(size_t)*g_count, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_X, X, sizeof(HWORD)*IBUFFSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_TandP, TandP, sizeof(HWORD)*g_count, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_TxorP, TxorP, sizeof(HWORD)*g_count, cudaMemcpyHostToDevice));
	kernel_FilterUp<<<BLOCKS, THREADS>>>(d_Imp, d_ImpD, Nwing, d_X, d_TandP, d_TxorP, d_indices, d_Vs, g_count, LpScl);
	CUDA_CHECK_RETURN(cudaMemcpy(Vs, d_Vs, sizeof(WORD)*g_count, cudaMemcpyDeviceToHost));
#ifdef GPU_DEBUG
	for (int i = 0; i < g_count; ++i) {
		fprintf(fp6, "%d\n", Vs[i]);
		Y[i] = WordToHword(Vs[i], NLpScl);
		fprintf(fp5, "%hd\n", Y[i]);
	}
	fprintf(fp5, "\n");
	fprintf(fp6, "\n");
	fclose(fp5);
	fclose(fp6);
#endif

	return g_count;
}

}
