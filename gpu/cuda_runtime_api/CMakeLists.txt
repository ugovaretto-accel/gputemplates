cmake_minimum_required( VERSION 2.8 )

project(cudaproto)

include(FindCUDA)
find_package( Boost )
if( Boost_FOUND )
  include_directories( ${Boost_INCLUDE_DIR} )
endif()

set( SRCS cudaproto.cpp stencilexprtemplates.cu exprtemplates.h )

include_directories( ${CUDA_INCLUDE_DIRS} )

cuda_add_executable( cudaproto ${SRCS} )

