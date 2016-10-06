# - Try to find OCULUS
# Once done this will define
#  OCULUS_FOUND
#  OCULUS_INCLUDE_DIRS
#  OCULUS_LIBRARIES
#  OCULUS_DLL

message("trying to find local OCULUS..")

FIND_PATH( OCULUS_INCLUDE_DIR
		NAMES OVR_CAPI_0_5_0.h 
		PATHS "${OCULUS_ROOT_DIR}/include"
		DOC "The directory where OVR_CAPI_0_5_0.h resides")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	FIND_LIBRARY(OCULUS_LIBRARY
	NAMES libOVR.a
	PATHS 
	${OCULUS_ROOT_DIR}/lib/x64/linux
	/usr/local/lib
	NO_DEFAULT_PATH)

	find_library(OCULUS_KERNEL_LIBRARY
	NAMES libOVRKernel.a
  	PATHS "${OCULUS_ROOT_DIR}/lib/x64/linux"
	/usr/local/lib
	NO_DEFAULT_PATH)
	
	# on linux dl and pthread are also required
	set(OCULUS_LIBRARIES ${OCULUS_LIBRARY} ${OCULUS_KERNEL_LIBRARY} dl rt pthread)
else()
	find_library(OCULUS_LIBRARY
	NAMES libOVR.lib
 	PATHS "${OCULUS_ROOT_DIR}/lib/x86/win32"
	NO_DEFAULT_PATH)

	find_library(OCULUS_KERNEL_LIBRARY
	NAMES libOVRKernel.lib
  	PATHS "${OCULUS_ROOT_DIR}/lib/x86/win32")

	set(OCULUS_LIBRARIES ${OCULUS_LIBRARY} ${OCULUS_KERNEL_LIBRARY})
endif()
message("${OCULUS_LIBRARY}")  

set(OCULUS_INCLUDE_DIRS ${OCULUS_INCLUDE_DIR} )

IF (OCULUS_INCLUDE_DIR)
SET( OCULUS_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
message("local OCULUS found!")
ELSE (OCULUS_INCLUDE_DIR)
SET( OCULUS_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (OCULUS_INCLUDE_DIR)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set <module>_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OCULUS
  DEFAULT_MSG
  OCULUS_LIBRARY
  OCULUS_INCLUDE_DIR)

mark_as_advanced(OCULUS_INCLUDE_DIR OCULUS_LIBRARIES)
