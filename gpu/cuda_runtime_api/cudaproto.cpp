/// OPENCL STENCIL CODE PORTED TO CUDA
/// NOTES :-( : 
/// - uses state machine model i.e. functions do not receive a context/device handler
///   but their behavior depends on the current state, it means that it is impossible
///   to understand on which device/context the function operates by looking at the function call site;
///   does not make any sense, prone to errors, since the programmer has to track state changes;
///   state machines make sense only in high performance code, certainly not in initialization
/// - passing parameters to kernels in the driver API is an example of non-portable design: requires the knowledge of
///   data alignment information on the device upon which the code will run! The programmer's guide
///   states that the alignment to use is the one of the host (?), however it seems it changes the
///   alignment to 8 when generating the code (?). Note that the run-time API makes use of the driver API
///   so the same problems are simply hidden. Also the sample code uses the non-portable __alignof
///   keyword which btw does not work with members of structs/classes
/// - requires multiple compilations and the use of an excel(!) sheet to compute block/thread layout;
///   in OpenCL we can simply call clGetKernelWorkGroupInfo, and this is by the way done
///   at run-time so tuning can be done dynamically depending on the data layout without
///   any recompilation
/// - coding conventions: the run-time API uses the same coding conventions for error (status) codes
///   and function names (?!)


#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>
#include "Timer.h"

#ifdef DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

///The one an only global variable: enable/disable CSV output useful for collecting data
///@todo handle inside global Print() function 
bool CSV = false;

//-----------------------------------------------------------------------------
struct CUDAExecutionContext
{
    //CUDA
    CUcontext context;
    CUdevice device;
    CUmodule program;
    CUfunction kernel;
    std::string kernelSource;
    //size_t wgroupSize; //"optimal" workgroup size as returned by run-time N/A for CUDA
    //application
    /*CUdeviceptr*/ real_t* domCL;
    /*CUdeviceptr*/ real_t* dom2CL;
    /*CUdeviceptr*/ real_t* stencilCL;
    unsigned rows;
    unsigned columns;
    // only initialize application objects
    CUDAExecutionContext() : domCL( 0 ), dom2CL( 0 ), stencilCL( 0 ), rows( 0 ), columns( 0 ) {}
    CUDAExecutionContext( CUcontext ctx,
                          CUdevice d,
                          CUmodule p,
                          CUfunction k,
                          const std::string& ks )
                          : context( ctx ), device( d ), program( p ),
                            kernel( k ), kernelSource( ks ),
                            domCL( 0 ), dom2CL( 0 ), stencilCL( 0 ),
                            rows( 0 ), columns( 0 ) {}
};


//-----------------------------------------------------------------------------
CUDAExecutionContext InitCUDAGPU()
{

    return CUDAExecutionContext();

}

//-----------------------------------------------------------------------------
/// Returns index of first non-compliant element or rows x columns if all
/// elements compliant
size_t ValidateResults( const real_t* ref, 
                        const real_t* result,
                        unsigned rows,
                        unsigned columns,
                        real_t eps )
{
    // can be done in 1D; done in 2D to make it easier to identify the cells
    for( unsigned r = 0; r != rows ; ++r )
    {
        for( unsigned c = 0; c != columns; ++c )
        {
            const size_t idx = r * columns + c;
            real_t sum = 0.f;
            if( std::abs( ref[ idx ] - result[ idx ] ) > eps ) return idx;
        }
    }
    return rows * columns;
}


//-----------------------------------------------------------------------------
// OpenCL has to be properly initialed before calling this function
double halo_filter_trivial( const real_t* dom, 
                            real_t* dom2,
                            const real_t* stencil, 
                            unsigned nr,
                            unsigned nc
                           )
{
    Timer timer;
    timer.Start();
    assert( dom );
    assert( dom2 );
    assert( stencil );
    assert( nr > 0 );
    assert( nc > 0 );
    const int FILTER_WIDTH = 3;
    const int FILTER_IDX_BOUND = FILTER_WIDTH / 2;
    for( int r = 0; r != nr; ++r )
    {
        for( int c = 0; c != nc; ++c )
        {
            real_t sum = 0.f;
            for( int i = -FILTER_IDX_BOUND; i <= FILTER_IDX_BOUND; ++i )
            {
                for( int j = -FILTER_IDX_BOUND; j <= FILTER_IDX_BOUND; ++j )
                {
                    int y = r + i;
                    int x = c + j;
                    if( y < 0 ) y = nr + i;
                    if( x < 0 ) x = nc + j;
                    if( y >= int( nc ) ) y = i - 1;
                    if( x >= int( nr ) ) x = j - 1;
                    assert( x >= 0 );
                    assert( x < int( nc ) );
                    assert( y >= 0 );
                    assert( y < int( nr ) );
                    sum += stencil[ (i + FILTER_IDX_BOUND) * FILTER_WIDTH + (j + FILTER_IDX_BOUND) ] * dom[ y * nc + x ];
                }
            }
            dom2[ r * nc  + c ] = sum;
        }
    }
    return timer.Stop();
}

//----------------------------------------------------------------------------
// Type information totally lost!
/*extern "C"*/ void RunStencil2D( dim3 blocks,
                                  dim3 threadsPerBlock,
                                  const real_t* dom, // input
                                  real_t* dom2, // output
                                  unsigned nrows, unsigned ncolumns,// size
                                  const real_t* stencil // filter      
                                );

//-----------------------------------------------------------------------------
// CUDA has to be properly initialized before calling this function
double halo_filter_cuda( const real_t* dom, 
                         real_t* dom2,
                         const real_t* stencil, 
                         unsigned nr,
                         unsigned nc,
                         size_t nsteps, //not used
                         CUDAExecutionContext& ec,
                         int workGroupSize[ 2 ] )
{
    assert( dom );
    assert( dom2 );
    assert( stencil );
    assert( nr > 0 );
    assert( nc > 0 );
    assert( nsteps > 0 ); // not used

    if( workGroupSize[ 0 ] < 1 ) throw std::logic_error( "Workgroup size dim[0] must be > 0" );

    //why is it called error ? is cudaSuccess an error ?
    //yet another design flaw: the cudaError_t type is an enum, so it cannot be initialized to
    //anything (e.g. success -1) and there is no 'unspecified' or 'uninitialized' status code that could be used for initializatio
    cudaError_t status; //cannot initialize to any meaningful value 
    
    // allocate new memory objects 
    const size_t DOMAIN_SIZE = nr * nc * sizeof( real_t );
    const size_t FILTER_SIZE = 9 * sizeof( real_t ); //3x3 - fixed

    // release previously allocated objects if needed
    // the execution context structure is initialized with rows and colums set to zero
    // this is therefore a signal that the buffers have not been yet allocated or that
    // the size has changed
    if(  ec.rows != nr || ec.columns != nc )
    {
        if( ec.domCL != 0 ) // not first allocation ?
        {
            status = cudaFree( ec.domCL ); 
            if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaFree()" );
            status = cudaFree( ec.dom2CL );
            if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaFree()" );
            status = cudaFree( ec.stencilCL );
            if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaFree()" );
        }
        
        //in
        status = cudaMalloc( /*ec.context,*/ reinterpret_cast<void**>(&ec.domCL), DOMAIN_SIZE ); //NO CONTEXT, STATE MACHINE MODEL :-((
        if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaMalloc()" );
        //out
        status = cudaMalloc( /*ec.context,*/ reinterpret_cast<void**>(&ec.dom2CL), DOMAIN_SIZE );
        if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaMalloc()" );
        //filter
        status = cudaMalloc( /*ec.context,*/ reinterpret_cast<void**>(&ec.stencilCL), FILTER_SIZE );
        if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaMalloc()" );
    }
    
    // transfer data
    status = cudaMemcpy( ec.domCL, dom, DOMAIN_SIZE, cudaMemcpyHostToDevice );
    if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaMemcpy()" );
    status = cudaMemcpy( ec.dom2CL, dom2, DOMAIN_SIZE, cudaMemcpyHostToDevice );
    if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaMemcpy()" );
    status = cudaMemcpy( ec.stencilCL, stencil, FILTER_SIZE, cudaMemcpyHostToDevice );
    if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaMemcpy()" );
    
    //update size info
    ec.rows = nr;
    ec.columns = nc;
    
    dim3 blocks( ec.columns / workGroupSize[ 0 ], ec.rows / workGroupSize[ 1 ] );
    dim3 threadsPerBlock( workGroupSize[ 0 ], workGroupSize[ 1 ] );

    Timer timer;
    timer.Start();
    // note that when calling a kernel type information is completely lost! CUdeviceptrs are converted
    // automatically to the parameters accepted by the kernel without any check!!!!
    RunStencil2D(blocks, threadsPerBlock, ec.domCL, ec.dom2CL, ec.rows, ec.columns, ec.stencilCL );
    cudaThreadSynchronize();
    const double elapsed = timer.Stop();
   
    //READ BACK RESULTS
    status = cudaMemcpy( reinterpret_cast< void* >( dom2 ), reinterpret_cast< const void* >( ec.dom2CL ), DOMAIN_SIZE, cudaMemcpyDeviceToHost );
    if( status != cudaSuccess ) throw std::runtime_error( "ERROR - cudaMemcpy()" );
    return elapsed;    
}

//------------------------------------------------------------------------------
/// Relase CUDA resources
void ReleaseResources( CUDAExecutionContext& ec )
{
    cudaThreadSynchronize(); //assuming the context is valid, very hard to manage with state machine behavior
    //since erro codes are now enums cannot simpy OR results and check at the end
    if( cudaFree( ec.domCL ) != cudaSuccess )
    {
        throw std::runtime_error( "Error - releasing resources" );
    }
    if( cudaFree( ec.dom2CL ) != cudaSuccess )
    {
        throw std::runtime_error( "Error - releasing resources" );
    }
    if( cudaFree( ec.stencilCL ) != cudaSuccess )
    {
        throw std::runtime_error( "Error - releasing resources" );
    }
    // who knows what happens to the context and module, it will probably stay around until the process
    // is terminated
    //status |= cuModuleUnload( ec.program );
    //status |= cuCtxDestroy( ec.context );    
}


//------------------------------------------------------------------------------
/// Set domain values - copied from original OpenMP reference code
void set_dom_val( real_t* dom, real_t* dom2, unsigned nr, unsigned nc )
{
    for( unsigned i = 0; i != nr; ++i )
    {
        for( unsigned j = 0; j != nc; ++j )
        {
            dom[ i * nc + j ] = real_t( i * nc + j );
            dom2[ i * nc + j ] =  dom[ i * nc + j ];
        }
    }
}

//------------------------------------------------------------------------------
///entry point
int main( int argc, char** argv )
{


    const std::string DEF_PLATFORM_TYPE( "cpu" );
    const unsigned DEF_PROBLEM_SIZE = 16;
    std::string platformType( DEF_PLATFORM_TYPE );
    int workGroupSize[] = { 4, 1 }; 
    unsigned problemSize = DEF_PROBLEM_SIZE;
    // not currently used for CUDA; can we simply implement formula
    // in NVIDIA's excel macro ?
    //bool computeWGroupSize = false;
    CSV = false;
    int i = 1;
    while( argc > 1 && i < argc )
    {
        //if( std::string( "-kernelcode" ) == argv[ i ] && ( i < (argc - 1 ) ) ) kernelSource = argv[ ++i ];
        if( std::string( "-platformType" ) == argv[ i ] && ( i < (argc - 1 ) ) ) platformType = argv[ ++i ];
        //else if( std::string( "-kernelname" ) == argv[ i ] && ( i < (argc - 1 ) ) ) kernelName = argv[ ++i ];
        else if( std::string( "-wgroupsize" ) == argv[ i ] && ( i < (argc - 2 ) ) )
        {
            workGroupSize[ 0 ] = ::atoi( argv[ ++i ] );
            workGroupSize[ 1 ] = ::atoi( argv[ ++i ] );
        }
        else if( std::string( "-domainscale" ) == argv[ i ] && ( i < (argc - 1 ) ) ) problemSize = ::atoi( argv[ ++i ] );
        else if( std::string( "-h" ) == argv[ i ] ) 
        {
            std::cout << "usage: " << argv[ 0 ]
                << "\n -platformType [gpu|cpu]"
                << "\n -wgroupsize <# of work items 2D> e.g. 2 2\n"
                << "\n -domainscale <scaling factor for domain size>"
                << "\n -csv output data in csv format for gathering statistics"
                << std::endl;
            ++i;
            return 0;
        }
        else if( std::string( "-csv" ) == argv[ i ] ) { CSV = true; ++i; }
        else ++i;
        //else std::cout << "Error - run " << argv[ 0 ] << " -h for help" << std::endl;
    }
    if( !CSV )
    {
        std::cout << "Platform      : " << platformType   << std::endl;
        std::cout << "Workgroup size: " << '('
                  << workGroupSize[ 0 ] << ',' 
                  << workGroupSize[ 1 ] << ')'
                  << std::endl;
#ifdef DOUBLE_PRECISION
                std::cout << "Double precision" << std::endl;
#else
                std::cout << "Single precision" << std::endl;
#endif           
    }

    try
    {
        const unsigned STENCIL_SIZE = 3;
        real_t stencil[ STENCIL_SIZE * STENCIL_SIZE ];
        const unsigned ntests = 2;
        const unsigned nsteps = 2;
        unsigned array_size[ 2 ];
        
        // initialize OpenCL environment
        CUDAExecutionContext ec;
        if( platformType != "cpu" ) ec = InitCUDAGPU();

        // initialize test array element: 8, 16, 24, 32...
        for( unsigned itest = 0; itest < ntests; itest++ )
        {
            array_size[ itest ] = 8 * ( 1 + itest );
        }
        
        // filter initialization
        const real_t pi = 3.141592654f;
        const real_t sixth =  1.f / 6.f;
        const real_t twelfth = 1.f / 12.f;
        const real_t one = 1.f;
        const real_t two = 2.f; // ?
        // 3x3 row major
        //--------------------------------
        //  ______________________
        // | -1/12 | -1/6 | -1/12 |
        // | -1/6  |   1  | -1/6  |
        // | -1/12 | -1/6 | -1/12 |
        //  ----------------------
        //--------------------------------
        stencil[ 0 ] = -twelfth;     //0,0
        stencil[ 2 ] = stencil[ 0 ]; //0,2
        stencil[ 6 ] = stencil[ 0 ]; //2,0
        stencil[ 8 ] = stencil[ 0 ]; //2,2
        stencil[ 1 ] = -sixth;       //0,1
        stencil[ 3 ] = stencil[ 1 ]; //1,0
        stencil[ 5 ] = stencil[ 1 ]; //1,2
        stencil[ 7 ] = stencil[ 1 ]; //2,1
        stencil[ 4 ] = one;          //1,1 center

        real_t* dom  = 0;
        real_t* dom2 = 0;
        
        if( !CSV )
        {
            std::cout << "\n===========================================" << std::endl;
            std::cout << "Running tests:" << std::endl;
        }
        // run tests
        double totalTime = 0.;
        for( unsigned itest = 0; itest < ntests; itest++ )
        {
            unsigned m = array_size[ itest ] * problemSize; //*nthreads ?? // scale with workgroup size ?
            unsigned n = m;
            delete [] dom; // note: deleting a null pointer has no effect
            delete [] dom2;
            dom  = new real_t[ m * n ];
            dom2 = new real_t[ m * n ];
            const real_t* result = nsteps % 2 == 0 ? dom : dom2;
            // note that for big arrays it makes sense to inizialize everything inside a kernel on the GPU
            set_dom_val( dom, dom2, m, n ); //myid ?
            double testTime = 0.;
            for( size_t istep = 0; istep < nsteps; istep++ )
            {
                if( platformType != "cpu" )
                    testTime += halo_filter_cuda( dom, dom2, stencil, m, n, nsteps, ec, workGroupSize );
                else
                    testTime += halo_filter_trivial( dom, dom2, stencil, m, n );
                std::swap( dom, dom2 );
            }
            if( !CSV ) 
            {
                std::cout << "Domain size: " << m << " x " << n << " time: " << testTime << " (ms)" << std::endl;
            }
            else
            {
                //only print result from largest domain:
                //<platform>,<rows>,<columns>,<workgroupsize[x]>,<workgroupsize[y]>,<time in ms>,
                // <value at 0>, <value at 10% domain size>, <value at 50% domain size>, <value at 90% domain size>,
                // <value at last element of domain array>, <precision>
                if( itest == ntests - 1 )
                {
                    std::cout << platformType << ',' << m << ',' << n << ',' 
                              << workGroupSize[ 0 ] << ',' << workGroupSize[ 1 ] << ','
                              << testTime << ','
                              // print sample output from five cells
                              << result[ 0                      ] << ','
                              << result[ size_t( 0.1  * m * n ) ] << ','
                              << result[ size_t( 0.5  * m * n ) ] << ','
                              << result[ size_t( 0.9  * m * n ) ] << ','
                              << result[ size_t( m * n - 1 )    ] << ','
                              // print precision information
#ifdef DOUBLE_PRECISION
                                                          << "double"
#else
                                                          << "single"
#endif
                                                          << std::endl;
                }
            }
            totalTime += testTime; 
        }
        delete [] dom;
        delete [] dom2;
        if( !CSV )
        {
            std::cout << "Total execution time: " << totalTime << " (ms)" << std::endl;
        }

        //cleanup
        if( platformType != "cpu" )
        {
            ReleaseResources( ec );
        }
    }
    catch( const std::exception& e )
    {
        std::cerr << e.what() << std::endl;
    }

#ifdef _MSC_VER
#ifdef _DEBUG
    std::cout << "\npress ENTER to exit" << std::endl;
    getchar();
#endif
#endif
    return 0;
}

