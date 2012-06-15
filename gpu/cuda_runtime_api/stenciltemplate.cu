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
		op.Finish(); //signal finish of loop execution
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
//         NOT TRUE ANYMORE we can call a templated kernel!!!    
// note 5: is it really a good idea to use C++ in kernels ? Lot of overhead e.g. constructors/destructors
template< typename StencilOperator > class Stencil
{
public:

   __device__ Stencil( const real_t* dom, 
                       const real_t* stencil,
                       const int& nrows,
                       const int& ncolumns,
                       const int& row,
                       const int& column,
                       real_t* dom2 ) : 
        stencilOperator_( dom , stencil, nrows, ncolumns, row, column ), sum_( 0 ), outputGrid_( dom2 )
        {}

   __device__ void operator()( int i, int j ) const
    {
    
        sum_ += stencilOperator_( i, j );
    }
    __device__ real_t Sum() const
    {
        return sum_;
    }
    __device__ void Finish() const
    {
        outputGrid_[ stencilOperator_.Row() * stencilOperator_.NumColumns() + stencilOperator_.Column() ] = sum_;
    }
private:
    const StencilOperator stencilOperator_;
    mutable real_t sum_;
    mutable real_t* outputGrid_;
private:
    Stencil();
    Stencil( const Stencil& );
    Stencil& operator=( const Stencil& );
};


//simple kernel for 2D stencils
template < class T, int STENCIL_ROWS, int STENCIL_COLUMNS > 
__global__  void Stencil2D(
                         const real_t* dom, // input
                         real_t* dom2, // output
                         uint nrows, uint ncolumns, // size
                         const real_t* stencil // filter      
                        )
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    Stencil< T > s( dom, stencil, nrows, ncolumns, r, c, dom2 );
    Loop2< STENCIL_ROWS, STENCIL_COLUMNS >::Execute( s );    
}


class StencilOperator
{
public:
    __device__ StencilOperator( const real_t* dom, 
                     const real_t* stencil,
                     const int& nrows,
                     const int& ncolumns,
                     const int& r,
                     const int& c ) 
                    : dom_( dom ), stencil_( stencil ), nrows_( nrows ), ncolumns_( ncolumns ), r_( r ), c_( c ), sum_( 0 ) {}
   __device__ real_t operator()( int i, int j ) const
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
        return stencil_[ idxS  ] * dom_[ idxI * ncolumns_ + idxJ ]; 	
    }
    __device__ const int& Row() const { return r_; }
    __device__ const int& Column() const { return c_; }
    __device__ const int& NumRows() const { return nrows_; }
    __device__ const int& NumColumns() const { return ncolumns_; }
private:
    const real_t* dom_;
    const real_t* stencil_;
    const int& nrows_;
    const int& ncolumns_;
    const int& r_;
    const int& c_;
    mutable real_t sum_;      
};

/*extern "C"*/ void RunStencil2D( dim3 b, dim3 tpb,
								  const real_t* dom, // input
								  real_t* dom2, // output
								  unsigned nrows, unsigned ncolumns,// size
								  const real_t* stencil // filter      
								)
{

    
	Stencil2D< StencilOperator, 3, 3 ><<<b,tpb>>>( dom, dom2, nrows, ncolumns, stencil );
}



