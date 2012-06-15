//#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
// use this for double precision
typedef double real_t;
#else
typedef float real_t;
#endif

typedef unsigned uint;

// compile-time 2D loop

template < int N, int M, int I = N - 1, int J = M - 1 >
struct Loop2
{
	template < typename OpT >
	__device__ static void Execute( const OpT& op )
    {
		op( N - I - 1, M - J - 1 );
		Loop2< N, M, I, J - 1 >::Execute( op );
    }
};

template < int N, int M, int I >
struct Loop2< N, M, I, 0 >
{
	template < typename OpT >
	__device__ static void Execute( const OpT& op )
    {
 		op( N - I - 1, M - 1 );
		// end of inner loop: repeat inner loop from start (0)
		// and increment outer loop (+1)
		Loop2< N, M, I - 1, M - 1 >::Execute( op );
    }
};


template < int N, int  M>
struct Loop2< N, M, 0, 0 >
{
	template < typename OpT >
	__device__ static void Execute( const OpT& op )
    {
		op( N - 1, M - 1 );
    }
};

// Stencil operator: accumulate results into member variable; perform assignment outside operator
// note 1: boundary checks will go away; halo regions will be transferred to GPU and kernel applied
//         to a subdomain
// note 2: IDX_BOUND offsets are currently applied to i,j index because the reference implementation of
//         compile-time 2D loops only works with 0..1 indices (and the code was copied verbatim from another kernel); 
//         easy fix: add i and j offsets which are used to compute the index to be passed to operator()
// note 3: the update of the output cell is done with one assignement at the end of the loop reading data
//         from a temporary variable because it's too time consuming to access global memory at each
//         iteration. Solution: Add a parameter to the () operator which is invoked with a different type
//         at the last iteration OR make an additional ::Finish() method call in the <.., .., 0, 0> loop specialization
// note 4: using CUDA instead of OpenCL makes it impossible to hide kernel invocation since somewhere the
//         programmer has to define a kernel function and write the DSEL based kernel inside it; the kernel
//         will then have to be wrapped into some class whose instance will be passed to the run-time     
// note 5: is it really a good idea to use C++ in kernels ? Lot of overhead e.g. constructors/destructors
class Stencil
{
public:

   __device__ Stencil( const real_t* dom, 
                       const real_t* stencil,
                       int nrows,
                       int ncolumns,
                       int row,
                       int column ) : 
        dom_( dom ), stencil_( stencil ), nrows_( nrows ), ncolumns_( ncolumns ), r_( row ), c_( column ), sum_( 0 )
        {}

   __device__ void operator()( int i, int j ) const
    {
        //Error - static is not allowed within a __device__ or __gloabl__ function
        const int FILTER_SIZE = 3;
        const int IDX_BOUND = 1;
        i -= IDX_BOUND;
        j -= IDX_BOUND;
        const int idxiS  = ( i + IDX_BOUND ) * FILTER_SIZE + IDX_BOUND;
        int idxS = idxiS + j;
        int idxI = r_ + i;
        int idxJ = c_ + j;
        if( idxI < 0 ) idxI = nrows_ + i; //i < 0
        if( idxJ < 0 ) idxJ = ncolumns_ + j; //j < 0
        if( idxI >= nrows_ ) idxI -= nrows_; //i > 0
        if( idxJ >= ncolumns_ ) idxJ -= ncolumns_; // j > 0  
        sum_ += stencil_[ idxS  ] * dom_[ idxI * ncolumns_ + idxJ ]; 	
    }
    __device__ real_t Sum() const
    {
        return sum_;
    }
private:
    const real_t* dom_;
    const real_t* stencil_;
    const int nrows_;
    const int ncolumns_;
    const int r_;
    const int c_;
    mutable real_t sum_;
private:
    Stencil();
    Stencil( const Stencil& );
    Stencil& operator=( const Stencil& );
};



//simple kernel for 2D stencils
extern "C" __global__ void Stencil2D(
                         const real_t* dom,    // input
                         real_t* dom2,        // output
                         uint nrows, uint ncolumns,     // size
                         const real_t* stencil // filter      
                        )
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    Stencil s( dom, stencil, nrows, ncolumns, r, c );
    // throw( 1 ); Error - device code does not support exception handling
    Loop2< 3, 3 >::Execute( s );
    dom2[ r * ncolumns + c ] = s.Sum();
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



