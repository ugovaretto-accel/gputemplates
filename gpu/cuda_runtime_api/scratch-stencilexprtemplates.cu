//#define DOUBLE_PRECISION

//always defined to avoid dangling references
#define STORE_EXPRESSIONS


//#include <boost/mpl/for_each.hpp>
//#include <boost/mpl/range_c.hpp>

#include <functional>
#include "exprtemplates.h"

#ifdef DOUBLE_PRECISION
// use this for double precision
typedef double real_t;
#else
typedef float real_t;
#endif

typedef unsigned uint;

struct StencilContext
{
public:
    struct VariableTag {};
    struct ConstTag {};
    template < int i, int j > struct PlaceHolder {};
    
public:
    __device__ StencilContext( const real_t* dom, 
                               const int& nrows,
                               const int& ncolumns,
                               const int& r,
                               const int& c ) 
                    : dom_( dom ), nrows_( nrows ), ncolumns_( ncolumns ), r_( r ), c_( c ) {}
   template < int i, int j >                 
   __device__ real_t operator()( VariableTag, PlaceHolder< i, j > ) const
    {
        int idxI = r_ + i;
        int idxJ = c_ + j;
        // checks will go away by passing a larger region + start offset
        if( idxI < 0 ) idxI = nrows_ + i; //i < 0
        if( idxJ < 0 ) idxJ = ncolumns_ + j; //j < 0
        if( idxI >= nrows_ ) idxI -= nrows_; //i > 0
        if( idxJ >= ncolumns_ ) idxJ -= ncolumns_; // j > 0
        return dom_[ idxI * ncolumns_ + idxJ ]; 	
    }
    template < int i, int j > 
    __device__ real_t operator()( ConstTag, PlaceHolder< i, j > ) const
    {
         // throw "NOT IMPLEMENTED"; cannot use exceptions in device code 
         return 0.f;
    }
    
private:
    const real_t* dom_;
    const int& nrows_;
    const int& ncolumns_;
    const int& r_;
    const int& c_;
};


struct ValueSet
{
    real_t* p_;
    int& pos_;
    ValueSet( real_t* p, int& pos ) : p_( p_ ), pos_( pos ) {}
    template< typename U >
    __device__ void operator()( U x )
    {
        *(p_ + pos_) = real_t( x );
        ++pos_; 
    }
    int CallMe() { return 1; }
};


//using namespace boost::mpl;
//ValueSet vs( dom2, pos );
//for_each< range_c<int,0,10> >( vs );  


//simple kernel for 2D stencils
template < typename ExprT > 
__global__  void Stencil2D(
                         const real_t* dom, // input
                         real_t* dom2, // output
                         uint nrows, uint ncolumns, // size
                         ExprT expr      
                        )
{ 
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    StencilContext ctx( dom, nrows, ncolumns, r, c );
    dom2[ r * ncolumns + c ] = expr.Eval( ctx );
    int pos = std::mem_fun( &ValueSet::CallMe )();
    
}


///Macro to simplify variable declaration; VarExpression is something that has an assignment operator;
///\todo use ConstExpression instead
#define DEC_GRID_CELL( x_, y_, NAME_ ) VarExpression< x_, y_ > const NAME_;
 

//Define operators; Add, Sub, Mul are pre-defined in the expr. template include file
DEF_TWO_OPERAND_OPERATOR( +, Add )
DEF_TWO_OPERAND_OPERATOR( -, Sub )
DEF_TWO_OPERAND_OPERATOR( *, Mul )

/// Proxy function which calls the actual kernel.
/// This function should accept an expression as a parameter in place of the stencil weights
/// instead of defining it internally
void RunStencil2D( dim3 b, dim3 tpb,
				   const real_t* dom, // input
				   real_t* dom2, // output
				   unsigned nrows, unsigned ncolumns,// size
				   const real_t* stencil // filter  <- NOT NEEDED ANYMORE    
				 )
{

    // Data access: index is relative to central cell
    DEC_GRID_CELL( -1,  1, NW );
    DEC_GRID_CELL(  0,  1, N );
    DEC_GRID_CELL(  1,  1, NE );
    DEC_GRID_CELL( -1,  0, W );
    DEC_GRID_CELL(  0,  0, CENTER );
    DEC_GRID_CELL(  1,  0, E );
    DEC_GRID_CELL( -1, -1, SW );
    DEC_GRID_CELL(  0, -1, S );
    DEC_GRID_CELL(  1, -1, SE );

    // Stencil kernel; will be passed from the outside in the future
    //--------------------------------
    //     ______________________
    //    | -1/12 | -1/6 | -1/12 |
    //    | -1/6  |   1  | -1/6  |
    //    | -1/12 | -1/6 | -1/12 |
    //     ----------------------
    //-------------------------------- 
    
#define STENCIL_EXPRESSION ( (-1.f/12.f) * NW - (1.f/6.f) * N - (1.f/12.f) * NE \
                             -(1.f/6.f)  * W  + CENTER - (1.f/6.f) * E          \
                             -(1.f/12.f) * SW - (1.f/6.f) * S - (1.f/12.f) * SE )

#define STENCIL_EXPRESSION_SIMPLIFIED ( (-1.f/12.f) * ( NW + NE + SW + SE ) - (1.f/6.f) * ( N + E + W + S ) + CENTER )    

    // invoke actual kernel performing the computation
    Stencil2D<<<b,tpb>>>( dom, dom2, nrows, ncolumns, STENCIL_EXPRESSION );
    // note assignment operator works on the CPU, no idea what happens on the gpu;
    // CENTER = 2.0f results into an error when copying back the results
}



