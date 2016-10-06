# - Try to find LAS
# Once done this will define
#  LAS_FOUND
#  LAS_INCLUDE_DIR
#  LAS_LIBRARY

message("trying to find local LAS..")

FIND_PATH( LAS_INCLUDE_DIR
		NAMES liblas.hpp 
		PATHS "${LAS_ROOT_DIR}/include"
		DOC "The directory where liblas.hpp resides")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	message("trying to find linux lib2..")
	FIND_LIBRARY(LAS_LIBRARY
	NAMES las
	PATHS 
	"${LAS_ROOT_DIR}/lib/x64/linux"
	NO_DEFAULT_PATH)
	message("${LAS_LIBRARY}")  
else()
	find_library(LAS_LIBRARY
	NAMES las
 	PATHS "${LAS_ROOT_DIR}/lib/x86/win32"
	NO_DEFAULT_PATH)
endif()


IF (LAS_INCLUDE_DIR)
SET( LAS_FOUND 1 CACHE STRING "Set to 1 if LAS is found, 0 otherwise")
message("local LAS found!")
ELSE (LAS_INCLUDE_DIR)
SET( LAS_FOUND 0 CACHE STRING "Set to 1 if LAS is found, 0 otherwise")
ENDIF (LAS_INCLUDE_DIR)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set <module>_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LAS
  DEFAULT_MSG
  LAS_LIBRARY
  LAS_INCLUDE_DIR)

mark_as_advanced(LAS_INCLUDE_DIR LAS_LIBRARY)
