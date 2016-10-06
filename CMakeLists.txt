
cmake_minimum_required (VERSION 3.5)
set(PROJECT_NAME recon)

project(${PROJECT_NAME})

# The version number.
set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()
#message(${CMAKE_BUILD_TYPE})
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
SET(CMAKE_LIBRARY_POSTFIX "")
SET(CMAKE_FIND_LIBRARY_PREFIXES "lib" "")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

 # should we use our own math functions?
option (VERBOSE_MODE "Verbose mode" OFF) 
  
# add the binary tree to the search path for include files
# so that we will find SDLconfig.h
include_directories ("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")

set(CMAKE_CXX_FLAGS_RELEASE " ")
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_DEBUG " ")
set(CMAKE_C_FLAGS_DEBUG "")
set(CMAKE_DEBUG_POSTFIX "d")

if(CMAKE_CONFIGURATION_TYPES)
	if (${CMAKE_BUILD_TYPE} MATCHES "Release")
		set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "allowed MSVC configuration types" FORCE)		
	else()
		set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "allowed MSVC configuration types" FORCE)			
	endif()
endif()

if(MSVC)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /MTd")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /MTd")
else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -fopenmp")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE}  -fopenmp")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE}  -fopenmp")
endif()

# Default places for archives (static), libs (static and shared) and binaries
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/libs)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/libs)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if (VERBOSE_MODE)
	set(CMAKE_VERBOSE_MAKEFILE ON)
endif (VERBOSE_MODE)

# note this must be first definition since cuda 5.5 also defines internal glew
if(GLEW_ROOT_DIR)
else()
  set(GLEW_ROOT_DIR "${PROJECT_SOURCE_DIR}/extlibs/x64/usr/local" CACHE PATH "" FORCE)
endif()
find_package(GLEW)
include_directories(${GLEW_INCLUDE_DIR})

if(CUDA_ROOT_DIR)
else()
  set(CUDA_ROOT_DIR "/usr/local/cuda" CACHE PATH "" FORCE)
  set(CUDA_TOOLKIT_ROOT "/usr/local/cuda" CACHE PATH "" FORCE)
  set(CUDA_SDK_ROOT_DIR "$ENV{HOME}/NVIDIA_CUDA-8.0_Samples/common" CACHE PATH "" FORCE)
endif()
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIR} ${CUDA_SDK_ROOT_DIR})

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
include_directories("${CUDA_SDK_ROOT_DIR}/inc")
# turn on CUDA opt
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -ccbin gcc-4.9")
# find packages
if(OPENCV_ROOT_DIR)
else()
  set(OPENCV_ROOT_DIR "${PROJECT_SOURCE_DIR}/extlibs/x64/usr/local" CACHE PATH "" FORCE)
endif()
find_package(OpenCV REQUIRED core imgproc highgui video videoio ts superres shape photo ml features2d flann imgcodecs)
include_directories("${OPENCV_INCLUDE_DIR}")

if(SDL2_ROOT_DIR)
else()
  set(SDL2_ROOT_DIR "${PROJECT_SOURCE_DIR}/extlibs/x64/usr/local" CACHE PATH "" FORCE)
endif()
find_package(SDL2)
include_directories(${SDL2_INCLUDE_DIR})

if(FREETYPE_ROOT_DIR)
else()
  set(FREETYPE_ROOT_DIR "${PROJECT_SOURCE_DIR}/extlibs/x64/usr/local" CACHE PATH "" FORCE)
endif()
find_package(FreeType)
include_directories(${FREETYPE_INCLUDE_DIR})
include_directories(${FREETYPE_INCLUDE_DIR}/freetype)
include_directories(${FREETYPE_INCLUDE_DIR}/freetype/config)

find_package(OpenGL)
include_directories( ${OPENGL_INCLUDE_DIRS})

add_definitions(-DEIGEN_DONT_ALIGN)
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/extlibs/x64/usr/local/include")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${SDL2_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIR_ft2build})
INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIR_freetype2})
INCLUDE_DIRECTORIES(${LIBUSB_1_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${FREENECT_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/utils/include")
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/tinyxml")

# Enable this to compile the library
add_subdirectory(tinyxml)
add_subdirectory(tracker)
add_subdirectory(rgbdscan)
