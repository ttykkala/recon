#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_DIR
# GLEW_LIBRARY
# GLEW_DLL

message("trying to find local GLEW..")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")

find_path(GLEW_INCLUDE_DIR GL/glew.h
    PATHS "${GLEW_ROOT_DIR}/include"
    DOC "The GLEW include path"
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_DEFAULT_PATH
)

FIND_LIBRARY( GLEW_LIBRARY
NAMES libGLEW.a
PATHS
"${GLEW_ROOT_DIR}/lib"
NO_SYSTEM_ENVIRONMENT_PATH
NO_DEFAULT_PATH
DOC "The GLEW library")

set(GLEW_DLL "")

message(${GLEW_INCLUDE_DIR})
message(${GLEW_LIBRARY})

else()

#find_path(GLEW_INCLUDE_DIR GL/glew.h 
#	PATHS "${GLEW_ROOT_DIR}/include"
#	DOC "The GLEW include path"
# 	NO_SYSTEM_ENVIRONMENT_PATH
#	NO_DEFAULT_PATH
#)

#FIND_LIBRARY( GLEW_LIBRARY
#NAMES glew32.lib
#PATHS "${GLEW_ROOT_DIR}/lib/x64/win64"
#DOC "The GLEW library"
#NO_SYSTEM_ENVIRONMENT_PATH
#NO_DEFAULT_PATH)

#FIND_FILE( GLEW_DLL
#NAMES glew32.dll
#PATHS "${GLEW_ROOT_DIR}/lib/x64/win64"
#DOC "The GLEW DLL"
#NO_SYSTEM_ENVIRONMENT_PATH
#NO_DEFAULT_PATH)

endif()


IF (GLEW_INCLUDE_DIR)
SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
message("local GLEW found!")
ELSE (GLEW_INCLUDE_DIR)
SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_DIR)

MARK_AS_ADVANCED( GLEW_FOUND )
