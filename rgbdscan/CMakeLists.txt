include_directories ("${PROJECT_SOURCE_DIR}/rgbdscan/include")
include_directories ("${PROJECT_SOURCE_DIR}/utils/include")
include_directories ("${PROJECT_SOURCE_DIR}/tracker")
include_directories ("${PROJECT_SOURCE_DIR}/tracker/kernels")

set(BIN rgbdscan)
add_definitions(-DENABLE_CUDA)

set(SRC_FILES ${PROJECT_SOURCE_DIR}/rgbdscan/src/main.cpp)
#list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/framebufferObject.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/GLSLProgram.cpp)
#list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/renderbuffer.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/Renderer.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/Shaders.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/GLWindow.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/fbo.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/SleepMs.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/teapot.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/screenshot.cpp)
list(APPEND SRC_FILES ${PROJECT_SOURCE_DIR}/utils/src/TrueTypeText.cpp)


#define MT / MTd build type for MSVC 2010:
set(PLATFORM_LIBS "")
if(MSVC)
	message("msvc detected!")
	SET(IGNORE_DEFAULT_LIBRARY_FLAGS_RELEASE "/NODEFAULTLIB:msvcrt.lib")
	SET(IGNORE_DEFAULT_LIBRARY_FLAGS_DEBUG "/NODEFAULTLIB:msvcrtd.lib /NODEFAULTLIB:msvcrt.lib")
endif()

cuda_add_executable(${BIN} ${SRC_FILES})
if(MSVC)
	if (${CMAKE_BUILD_TYPE} MATCHES "Release")
		SET_TARGET_PROPERTIES(${BIN} PROPERTIES LINK_FLAGS_RELEASE ${IGNORE_DEFAULT_LIBRARY_FLAGS_RELEASE})
	else() 
		SET_TARGET_PROPERTIES(${BIN} PROPERTIES LINK_FLAGS_DEBUG ${IGNORE_DEFAULT_LIBRARY_FLAGS_DEBUG})
	endif()
endif()

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
        target_link_libraries(${BIN} trackerlib ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${SDL2_LIBRARIES} ${PNG_LIBRARY} ${GLEW_LIBRARY} ${FREETYPE_LIBRARY} ${CUDA_LIBRARIES} ${OPENCV_LIBRARIES}  ${PLATFORM_LIBS})
else()
        target_link_libraries(${BIN} trackerlib ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${SDL2_LIBRARIES} ${PNG_LIBRARY} ${GLEW_LIBRARY}  ${FREETYPE_LIBRARY} ${CUDA_LIBRARIES} ${OPENCV_DEBUG_LIBRARIES} ${PLATFORM_LIBS})
endif()

