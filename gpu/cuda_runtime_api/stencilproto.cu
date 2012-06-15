// Include all of Proto
#include <boost/proto/proto.hpp>

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
// use this for double precision
typedef double real_t;
#else
typedef float real_t;
#endif

typedef unsigned uint;

namespace proto = boost::proto;
using proto::_;

template < int i, int j > struct GridCell {};


struct StencilContext : proto::callable_context< StencilContext const >
{
public:
    __host__ StencilContext( const real_t* dom, 
                               const int& nrows,
                               const int& ncolumns,
                               const int& r,
                               const int& c ) 
                    : dom_( dom ), nrows_( nrows ), ncolumns_( ncolumns ), r_( r ), c_( c ) {}
   template < int i, int j >                 
   __host__ real_t operator()( proto::tag::terminal, GridCell< i, j > ) const
    {
        //Error - static is not allowed within a __host__ or __gloabl__ function
        const int FILTER_SIZE = 3;
        const int IDX_BOUND = 1;
        const int idxiS  = ( i + IDX_BOUND ) * FILTER_SIZE + IDX_BOUND;
        int idxS = idxiS + j;
        int idxI = r_ + i;
        int idxJ = c_ + j;
        if( idxI < 0 ) idxI = nrows_ + i; //i < 0
        if( idxJ < 0 ) idxJ = ncolumns_ + j; //j < 0
        if( idxI >= nrows_ ) idxI -= nrows_; //i > 0
        if( idxJ >= ncolumns_ ) idxJ -= ncolumns_; // j > 0
        return dom_[ idxI * ncolumns_ + idxJ ]; 	
    }
    typedef real_t result_type;
private:
    const real_t* dom_;
    const int& nrows_;
    const int& ncolumns_;
    const int& r_;
    const int& c_;
};


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
    dom2[ r * ncolumns + c ] = proto::eval( expr, ctx ); 
}



#define DEC_GRID_CELL( x, y, NAME_ ) proto::terminal< GridCell< x, y > >::type const NAME_ = {{}}; 


void RunStencil2D( dim3 b, dim3 tpb,
				   const real_t* dom, // input
				   real_t* dom2, // output
				   unsigned nrows, unsigned ncolumns,// size
				   const real_t* stencil // filter      
				 )
{

    DEC_GRID_CELL( -1,  1, NW );
    DEC_GRID_CELL(  0,  1, N );
    DEC_GRID_CELL(  1,  1, NE );
    /*DEC_GRID_CELL(  0, -1, W );
    DEC_GRID_CELL(  0,  0, CENTER );
    DEC_GRID_CELL(  0,  1, E );
    DEC_GRID_CELL( -1, -1, SW );
    DEC_GRID_CELL(  0, -1, S );
    DEC_GRID_CELL(  1, -1, SE );*/
     
	Stencil2D<<<b,tpb>>>( dom, dom2, nrows, ncolumns, NW + N - NE );
}



