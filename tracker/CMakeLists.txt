
set(BIN trackerlib)
include_directories ("${PROJECT_SOURCE_DIR}/tracker")
include_directories ("${PROJECT_SOURCE_DIR}/tracker/kernels")

set(SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/capture/VideoPreProcessorCPU.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/capture/fileSource.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/calib/calib.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/calib/ResultTxt.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/calib/GroundTruth.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/calib/TrackingParams.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/reconstruct/zconv.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/reconstruct/VoxelGrid.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/multicore/multicore.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/rendering/VertexBuffer2.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/rendering/shader.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/rendering/BaseBuffer2.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/rendering/TriangleBuffer2.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/reconstruct/TrackFrame.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/timer/performanceCounter.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/image2/Image2.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/image2/ImagePyramid2.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/tracker/phototracker.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/tracker/shapetracker.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/tracker/basic_math.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/tracker/eigenMath.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/tracker/expm.c)



# Since TinyXML is only object files, we need to do some tricks:
#SET(TINYXML_SOURCES
#      tinyxml.h tinyxml.cpp
#      tinystr.h tinystr.cpp
#      tinyxmlerror.cpp
#      tinyxmlparser.cpp)
#ADD_LIBRARY(tinyxml STATIC ${TINYXML_SOURCES})

# cudakernels
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/VoxelGrid_cuda.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/devices.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/square_array.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/cudaWarp.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/d2z.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/downSample2Cuda.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/hostUtils.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/rgb2gray.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/undistort.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/convert2FloatCuda.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/cuda_histogram.h)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/expmCuda.h)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/f2cCuda.h)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/kernelUtils.h)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/histogram64.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/filterIndex256.cu)

#list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/histogram256.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/warp.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/matrixMul_kernel.cu)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/tracker/kernels/cuda_hist.cu)

CUDA_ADD_LIBRARY(${BIN} STATIC ${SRC_FILES})
if (${CMAKE_BUILD_TYPE} MATCHES "Release")
        target_link_libraries(${BIN} tinyxml )
else()
        target_link_libraries(${BIN} tinyxml )
endif()

