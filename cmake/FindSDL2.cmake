# - Try to find SDL2
# Once done this will define
#  SDL2_FOUND
#  SDL2_INCLUDE_DIRS
#  SDL2_LIBRARIES

message("trying to find local SDL2..")

FIND_PATH( SDL2_INCLUDE_DIR
		NAMES SDL.h 
		PATHS "${SDL2_ROOT_DIR}/include/SDL2"
		DOC "The directory where SDL.h resides")

#if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")

	FIND_LIBRARY(SDL2_LIBRARY
	NAMES libSDL2.a
	PATHS "${SDL2_ROOT_DIR}/lib"
	NO_DEFAULT_PATH)

	find_library(SDL2main_LIBRARY
	NAMES libSDL2main.a
  	PATHS "${SDL2_ROOT_DIR}/lib"
	NO_DEFAULT_PATH)

#	find_file(SDL2_DLL
#	NAMES libSDL2.so
# 	PATHS "{SDL2_ROOT_DIR}/lib"
#	NO_DEFAULT_PATH)
	# on linux dl and pthread are also required
	set(SDL2_LIBRARIES ${SDL2_LIBRARY} ${SDL2main_LIBRARY} dl pthread)
#else()
#	find_library(SDL2_LIBRARY
#	NAMES SDL2.lib
# 	PATHS "${SDL2_ROOT_DIR}/lib/x64/win64"
#	NO_DEFAULT_PATH)

#	find_library(SDL2main_LIBRARY
#	NAMES SDL2main.lib
#  	PATHS "${SDL2_ROOT_DIR}/lib/x64/win64")

#	find_file(SDL2_DLL
#	NAMES SDL2.dll
#	PATHS "${SDL2_ROOT_DIR}/lib/x64/win64")

#	set(SDL2_LIBRARIES ${SDL2_LIBRARY} ${SDL2main_LIBRARY})
#endif()
message("${SDL2_LIBRARY}")  

set(SDL_INCLUDE_DIRS ${SDL2_INCLUDE_DIR} )

IF (SDL2_INCLUDE_DIR)
SET( SDL2_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
message("local SDL2 found!")
ELSE (SDL2_INCLUDE_DIR)
SET( SDL2_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (SDL2_INCLUDE_DIR)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set <module>_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(SDL2
  DEFAULT_MSG
  SDL2_LIBRARY
  SDL2_INCLUDE_DIR)

mark_as_advanced(SDL2_INCLUDE_DIR SDL2_LIBRARIES)
