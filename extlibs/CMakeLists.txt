project(download_deps)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE on)
include(ExternalProject)
include(ProcessorCount)

if (NOT INSTALL_ROOT)
message("\
**************************
INSTALL_ROOT must be defined
INSTALL_ROOT = path to where install the compiles software, e.g, /home/myname/root
**************************")
  message(FATAL_ERROR)
endif()

ProcessorCount(JJ)

set (CC "${TOOLCHAIN}gcc")
set (CXX "${TOOLCHAIN}g++")
set (LINK "${TOOLCHAIN}g++")
set (BUILD x86-linux)

set (INSTALL_ROOT_X64 ${INSTALL_ROOT}/x64)
set (INSTALL_REL usr/local)
set (INSTALL_X64 ${INSTALL_ROOT_X64}/${INSTALL_REL})

#***********************************
# Downloads
#***********************************
ExternalProject_Add(opencv-download
  PREFIX downloads/opencv-download
	URL http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/3.1.0/opencv-3.1.0.zip
#  URL https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/3.1.0/opencv-3.1.0.zip/download
  DOWNLOAD_NAME opencv-3.1.0.zip
  URL_HASH SHA1=8c932b68fe2e1575e88dde759ab1ed1d53d6f41b
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

ExternalProject_Add(glew-download
  PREFIX downloads/glew-download
  URL http://downloads.sourceforge.net/project/glew/glew/2.0.0/glew-2.0.0.tgz
  DOWNLOAD_NAME glew-2.0.0.tgz
  URL_HASH SHA1=6e15c84f7e1fad34cd3679f784a233744ddf048f
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)


ExternalProject_Add(freetype-download
  PREFIX downloads/freetype-download
  URL http://download.savannah.gnu.org/releases/freetype/freetype-2.6.5.tar.gz
  DOWNLOAD_NAME freetype-2.6.5.tar.gz
  URL_HASH SHA1=c5a453613a736d8c4a5ef96586a2db5bc6cdbd96
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

ExternalProject_Add(eigen-download
  PREFIX downloads/eigen-download
  URL http://bitbucket.org/eigen/eigen/get/3.2.9.tar.gz
  DOWNLOAD_NAME 3.2.9.tar.gz
  URL_HASH SHA1=e8463be62ca64a036cbf0b8f86a3e55f8e3b03d1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)


ExternalProject_Add(sdl2-download
  PREFIX downloads/sdl2-download
  URL https://www.libsdl.org/release/SDL2-2.0.4.tar.gz
  DOWNLOAD_NAME SDL2-2.0.4.tar.gz
  URL_HASH SHA1=470a23dfc4b6220c687925ebbe5fe96287f8fb08
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

add_custom_target(download_all 
  DEPENDS opencv-download glew-download freetype-download eigen-download sdl2-download)

add_custom_target(all-x64 DEPENDS opencv-x64 glew-x64 freetype-x64 eigen-x64 sdl2-x64)

#***********************************
# builds, Linux
#***********************************

# OpenCV
ExternalProject_Get_Property(opencv-download download_dir download_name)
  ExternalProject_Add(opencv-x64
    PREFIX builds/opencv-x64
    DEPENDS opencv-download
    URL ${download_dir}/${download_name}
    URL_HASH SHA1=8c932b68fe2e1575e88dde759ab1ed1d53d6f41b
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=${INSTALL_X64} -DBUILD_CUDA_STUBS=OFF -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PACKAGE=ON -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TBB=ON -DBUILD_TESTS=OFF -DBUILD_TIFF=OFF -DBUILD_WITH_DEBUG_INFO=OFF -DBUILD_WITH_DYNAMIC_IPP=OFF -DBUILD_ZLIB=OFF -DBUILD_opencv_apps=ON -DBUILD_opencv_calib3d=ON -DBUILD_opencv_core=ON -DBUILD_opencv_features2d=ON -DBUILD_opencv_flann=ON -DBUILD_opencv_highgui=ON -DBUILD_opencv_imgcodecs=ON -DBUILD_opencv_imgproc=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_ml=ON -DBUILD_opencv_objdetect=ON -DBUILD_opencv_photo=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_shape=ON -DBUILD_opencv_stitching=ON -DBUILD_opencv_superres=ON -DBUILD_opencv_ts=OFF -DBUILD_opencv_video=ON -DBUILD_opencv_videoio=ON -DBUILD_opencv_videostab=ON -DBUILD_opencv_world=OFF  -DCMAKE_BUILD_TYPE=Release -DCUDA_BUILD_CUBIN=OFF -DCUDA_BUILD_EMULATION=OFF -DCUDA_SEPARABLE_COMPILATION=OFF -DCUDA_VERBOSE_BUILD=OFF -DDOWNLOAD_EXTERNAL_TEST_DATA=OFF -DENABLE_AVX=OFF -DENABLE_AVX2=OFF -DENABLE_COVERAGE=OFF -DENABLE_FAST_MATH=OFF -DENABLE_FMA3=OFF -DENABLE_IMPL_COLLECTION=OFF -DENABLE_NOISY_WARNINGS=OFF -DENABLE_OMIT_FRAME_POINTER=ON -DENABLE_POPCNT=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DENABLE_PROFILING=OFF -DENABLE_SOLUTION_FOLDERS=OFF -DENABLE_SSE=ON -DENABLE_SSE2=ON -DENABLE_SSE3=ON -DENABLE_SSE41=OFF -DENABLE_SSE42=OFF -DENABLE_SSE3=OFF -DGENERATE_ABI_DESCRIPTOR=OFF -DINSTALL_CREATE_DISTRIB=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_TESTS=OFF -DINSTALL_TO_MANGLED_PATHS=OFF -DOPENCV_WARNINGS_ARE_ERRORS=OFF -DWITH_1394=OFF -DWITH_CLP=OFF -DWITH_CUBLAS=OFF -DWITH_CUDA=OFF -DWITH_CUFFT=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_GDAL=OFF -DWITH_GIGEAPI=OFF -DWITH_GPHOTO2=ON -DWITH_GSTREAMER=ON -DWITH_GSTREAMER_0_10=OFF -DWITH_GTK=OFF -DWITH_GTK_2_X=OFF -DWITH_IPP=OFF -DWITH_IPP_A=OFF -DWITH_JASPER=ON -DWITH_JPEG=ON -DWITH_LIBV4L=OFF -DWITH_MATLAB=OFF -DWITH_NVCUVID=OFF -DWITH_OPENCL=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCL_SVM=OFF -DWITH_OPENEXR=OFF -DWITH_OPENGL=OFF -DWITH_OPENMP=OFF -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF -DWITH_PNG=ON -DWITH_PTHREADS_PF=ON -DWITH_PVAPI=ON -DWITH_QT=OFF -DWITH_TBB=OFF -DWITH_TIFF=ON -DWITH_UNICAP=OFF -DWITH_V4L=OFF -DWITH_VA=OFF -DWITH_VA_INTEL=OFF -DWITH_VTK=OFF -DWITH_WEBP=ON -DWITH_XIMEA=OFF -DWITH_XINE=OFF
    BUILD_COMMAND make -j${JJ}    
    INSTALL_COMMAND make install
  )


# glew
ExternalProject_Get_Property(glew-download download_dir download_name)
  ExternalProject_Add(glew-x64
    PREFIX builds/glew-x64
    DEPENDS glew-download
    URL ${download_dir}/${download_name}
    URL_HASH SHA1=6e15c84f7e1fad34cd3679f784a233744ddf048f
	  CONFIGURE_COMMAND cd ./build/cmake && cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=${INSTALL_X64}
   	BUILD_IN_SOURCE 1
    BUILD_COMMAND cd ./build/cmake && make -j${JJ}
    INSTALL_COMMAND cd ./build/cmake && make install
  )

# freetype
ExternalProject_Get_Property(freetype-download download_dir download_name)
  ExternalProject_Add(freetype-x64
    PREFIX builds/freetype-x64
    DEPENDS freetype-download
    URL ${download_dir}/${download_name}
    URL_HASH SHA1=c5a453613a736d8c4a5ef96586a2db5bc6cdbd96
	  CONFIGURE_COMMAND ./configure --prefix=${INSTALL_X64}
	  BUILD_IN_SOURCE 1
    BUILD_COMMAND make -j${JJ}
    INSTALL_COMMAND make install
  )

# eigen
ExternalProject_Get_Property(eigen-download download_dir download_name)
  ExternalProject_Add(eigen-x64
    PREFIX builds/eigen-x64
	  DEPENDS eigen-download opencv-x64
    URL ${download_dir}/${download_name}
    URL_HASH SHA1=e8463be62ca64a036cbf0b8f86a3e55f8e3b03d1
   	PATCH_COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/lib-eigen-patch/Functors.h" ./Eigen/src/Core/Functors.h
	  CONFIGURE_COMMAND echo "no need to configure eigen :)"
	  BUILD_IN_SOURCE 1
    BUILD_COMMAND echo "no need to build eigen :)"
    INSTALL_COMMAND cp -R Eigen ${INSTALL_X64}/include
  )

# sdl
ExternalProject_Get_Property(sdl2-download download_dir download_name)
  ExternalProject_Add(sdl2-x64
    PREFIX builds/sdl2-x64
    DEPENDS sdl2-download
    URL ${download_dir}/${download_name}
    URL_HASH SHA1=470a23dfc4b6220c687925ebbe5fe96287f8fb08
	CONFIGURE_COMMAND ./configure --prefix=${INSTALL_X64}
   	BUILD_IN_SOURCE 1
    BUILD_COMMAND make -j${JJ}
    INSTALL_COMMAND make install
  )


