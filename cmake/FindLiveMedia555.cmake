# - Try to find SDL2
# Once done this will define
#  LIVEMEDIA555_FOUND
#  LIVEMEDIA555_INCLUDE_DIR
#  LIVEMEDIA555_LIBRARY
include(FindPackageHandleStandardArgs)

message("trying to find local livemedia555..")

FIND_PATH( LIVEMEDIA555_INCLUDE_DIR
		NAMES liveMedia.hh
		PATHS "${LIVEMEDIA555_ROOT_DIR}/include"
		DOC "The directory where livemedia.h resides")

find_library(LIVEMEDIA555_LIBRARY
  NAMES static-live555.lib
  PATHS "${LIVEMEDIA555_ROOT_DIR}/lib/x86/win32")

find_library(LIVEMEDIA555_DEBUG_LIBRARY
  NAMES static-live555d.lib
  PATHS "${LIVEMEDIA555_ROOT_DIR}/lib/x86/win32")

IF (LIVEMEDIA555_INCLUDE_DIR)
SET( LIVEMEDIA555_FOUND 1 CACHE STRING "Set to 1 if Livemedia555 is found, 0 otherwise")
message("local livemedia555 found!")
ELSE (LIVEMEDIA555_INCLUDE_DIR)
SET( LIVEMEDIA555_FOUND 0 CACHE STRING "Set to 1 if Livemedia555 is found, 0 otherwise")
ENDIF (LIVEMEDIA555_INCLUDE_DIR)

# handle the QUIETLY and REQUIRED arguments and set <module>_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LIVEMEDIA555
  DEFAULT_MSG
  LIVEMEDIA555_LIBRARY
  LIVEMEDIA555_INCLUDE_DIR)

mark_as_advanced(LIVEMEDIA555_INCLUDE_DIR LIVEMEDIA555_LIBRARY)
