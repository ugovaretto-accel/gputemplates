//#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
// use this for double precision
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

//crap: cannot separate C++ from CUDA
#include <cuda.h>

typedef unsigned uint;


//simple kernel for 2D stencils
/*extern "C"*/ __global__ void Stencil2D(
                         const real_t* dom,    // input
                         real_t* dom2,        // output
                         uint nrows, uint ncolumns,     // size
                         const real_t* stencil // filter      
                        )
{

//infinite loop produces the following behavior:
// -tesla:   nothing, computation goes on forever, program can
//           stopped with ^C
// -geforce: computation stops after a number of seconds; reading
//           data back results in an error
// -cpu:     compiler reports unreachable code and program
//           segfaults during kernel execution
//while(1);

    const int FILTER_SIZE = 3;
    const int IDX_BOUND = FILTER_SIZE >> 1;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    real_t sum = 0.0;
    for( int i = -IDX_BOUND; i <= IDX_BOUND; ++i )
    {
        int idxiS  = ( i + IDX_BOUND ) * FILTER_SIZE + IDX_BOUND;
        for ( int j = -IDX_BOUND; j <= IDX_BOUND; ++j )
        {
            int idxS = idxiS + j;
            int idxI = r + i;
            int idxJ = c + j;
            if( idxI < 0 ) idxI = nrows + i; //i < 0
            if( idxJ < 0 ) idxJ = ncolumns + j; //j < 0
            if( idxI >= nrows ) idxI -= nrows; //i > 0
            if( idxJ >= ncolumns ) idxJ -= ncolumns; // j > 0   
            sum += stencil[ idxS  ] * dom[ idxI * ncolumns + idxJ ];
        }
    }
    dom2[ r * ncolumns + c ] = sum;	
}

/*extern "C"*/ void RunStencil2D( dim3 b, dim3 tpb,
								  const real_t* dom, // input
								  real_t* dom2, // output
								  unsigned nrows, unsigned ncolumns,// size
								  const real_t* stencil // filter      
								)
{
	Stencil2D<<<b,tpb>>>(dom, dom2, nrows, ncolumns, stencil );
}