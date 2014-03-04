#include <stdlib.h>
#include <stdio.h>

#include "stdefs.h"
#include "resample.h"

WORD *d_Vs;

__global__ void kernel( ) {
}

extern "C" {

void GPU_Init() {

}

int GPU_SrcUP(HWORD X[], HWORD Y[], double factor, UWORD *Time,
			  UHWORD Nx, UHWORD Nwing, UHWORD LpScl,
			  HWORD Imp[], HWORD ImpD[], BOOL Interp,
			  HWORD Xps[], HWORD TandP[], HWORD TxorP[],
			  WORD Vs[]) {
	HWORD *Xp, *Ystart;
	WORD v;
	
	double dt;
	UWORD dtb;
	UWORD endTime;

	dt = 1.0/factor;
	dtb = dt*(1<<Np) + 0.5;
	
	Ystart = Y;
	endTime = *Time + (1<<Np)*(WORD)Nx;

	/* GPU need */
	int g_count = ((1<<Np)*(WORD)Nx + dtb - 1) / dtb;
	int realTime;
	for (int i = 0; i < g_count; ++i) {
		realTime = *Time + i * dtb;
		Xps[i] = X[realTime>>Np];
		TandP[i] = (HWORD)(realTime&Pmask);
		TxorP[i] = (HWORD)(((realTime^Pmask)+1)&Pmask);
	}
	*Time += g_count * dtb;
}

}
