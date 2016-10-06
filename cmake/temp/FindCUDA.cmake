# - Try to find CUDA
# Once done this will define
#  CUDA_FOUND
#  CUDA_INCLUDE_DIRS
#  CUDA_LIBRARIES
#  CUDA_DLL
#  CUDA_ADD_EXECUTABLE( cuda_target file0 file1 ...
#                        [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#   -- Creates an executable "cuda_target" which is made up of the files
#      specified.  All of the non CUDA C files are compiled using the standard
#      build rules specified by CMAKE and the cuda files are compiled to object
#      files using nvcc and the host compiler.  In addition CUDA_INCLUDE_DIRS is
#      added automatically to include_directories().  Some standard CMake target
#      calls can be used on the target after calling this macro
#      (e.g. set_target_properties and target_link_libraries), but setting
#      properties that adjust compilation flags will not affect code compiled by
#      nvcc.  Such flags should be modified before calling CUDA_ADD_EXECUTABLE,
#      CUDA_ADD_LIBRARY or CUDA_WRAP_SRCS.
#   CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME( output_file_var
#                                                        cuda_target
#                                                        object_files )
#   -- Compute the name of the intermediate link file used for separable
#      compilation.  This file name is typically passed into
#      CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS.  output_file_var is produced
#      based on cuda_target the list of objects files that need separable
#      compilation as specified by object_files.  If the object_files list is
#      empty, then output_file_var will be empty.  This function is called
#      automatically for CUDA_ADD_LIBRARY and CUDA_ADD_EXECUTABLE.  Note that
#      this is a function and not a macro.
##
#   CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS( output_file_var cuda_target
#                                            nvcc_flags object_files)
#
#   -- Generates the link object required by separable compilation from the given
#      object files.  This is called automatically for CUDA_ADD_EXECUTABLE and
#      CUDA_ADD_LIBRARY, but can be called manually when using CUDA_WRAP_SRCS
#      directly.  When called from CUDA_ADD_LIBRARY or CUDA_ADD_EXECUTABLE the
#      nvcc_flags passed in are the same as the flags passed in via the OPTIONS
#      argument.  The only nvcc flag added automatically is the bitness flag as
#      specified by CUDA_64_BIT_DEVICE_CODE.  Note that this is a function
#      instead of a macro.
#   CUDA_WRAP_SRCS ( cuda_target format generated_files file0 file1 ...
#                    [STATIC | SHARED | MODULE] [OPTIONS ...] )
#   -- This is where all the magic happens.  CUDA_ADD_EXECUTABLE,
#      CUDA_ADD_LIBRARY, CUDA_COMPILE, and CUDA_COMPILE_PTX all call this
#      function under the hood.
#
#      Given the list of files (file0 file1 ... fileN) this macro generates
#      custom commands that generate either PTX or linkable objects (use "PTX" or
#      "OBJ" for the format argument to switch).  Files that don't end with .cu
#      or have the HEADER_FILE_ONLY property are ignored.
#
#      The arguments passed in after OPTIONS are extra command line options to
#      give to nvcc.  You can also specify per configuration options by
#      specifying the name of the configuration followed by the options.  General
#      options must preceed configuration specific options.  Not all
#      configurations need to be specified, only the ones provided will be used.
#
#         OPTIONS -DFLAG=2 "-DFLAG_OTHER=space in flag"
#         DEBUG -g
#         RELEASE --use_fast_math
#         RELWITHDEBINFO --use_fast_math;-g
#         MINSIZEREL --use_fast_math
#
#      For certain configurations (namely VS generating object files with
#      CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE set to ON), no generated file will
#      be produced for the given cuda file.  This is because when you add the
#      cuda file to Visual Studio it knows that this file produces an object file
#      and will link in the resulting object file automatically.
#
#      This script will also generate a separate cmake script that is used at
#      build time to invoke nvcc.  This is for several reasons.
#
#        1. nvcc can return negative numbers as return values which confuses
#        Visual Studio into thinking that the command succeeded.  The script now
#        checks the error codes and produces errors when there was a problem.
#
#        2. nvcc has been known to not delete incomplete results when it
#        encounters problems.  This confuses build systems into thinking the
#        target was generated when in fact an unusable file exists.  The script
#        now deletes the output files if there was an error.
#
#        3. By putting all the options that affect the build into a file and then
#        make the build rule dependent on the file, the output files will be
#        regenerated when the options change.
#
#      This script also looks at optional arguments STATIC, SHARED, or MODULE to
#      determine when to target the object compilation for a shared library.
#      BUILD_SHARED_LIBS is ignored in CUDA_WRAP_SRCS, but it is respected in
#      CUDA_ADD_LIBRARY.  On some systems special flags are added for building
#      objects intended for shared libraries.  A preprocessor macro,
#      <target_name>_EXPORTS is defined when a shared library compilation is
#      detected.
#
#      Flags passed into add_definitions with -D or /D are passed along to nvcc.
#
#
message("trying to find local CUDA..")

FIND_PATH( CUDA_INCLUDE_DIR
		NAMES cuda_runtime.h 
		PATHS "${CUDA_ROOT_DIR}/include"
		DOC "The directory where cuda_runtime.h resides")
		
FIND_PATH( CUDA_SDK_ROOT_DIR
		NAMES helper_cuda.h 
		PATHS "${CUDA_ROOT_DIR}/include/cuda_sdk_common/inc"
		DOC "The directory where helper_cuda.h resides")		

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	FIND_LIBRARY(CUDA_LIBRARY
	NAMES cudart_static
	PATHS 
	${CUDA_ROOT_DIR}/lib/x64/linux
	/usr/local/lib
	NO_DEFAULT_PATH)

	find_library(NPPC_LIBRARY
	NAMES nppc_static
  	PATHS "${CUDA_ROOT_DIR}/lib/x64/linux"
	/usr/local/lib
	NO_DEFAULT_PATH)

	find_library(NPPI_LIBRARY
	NAMES nppi_static
  	PATHS "${CUDA_ROOT_DIR}/lib/x64/linux"
	/usr/local/lib
	NO_DEFAULT_PATH)
		

	find_library(CULIBOS_LIBRARY
	NAMES culibos
	PATHS "${CUDA_ROOT_DIR}/lib/x64/linux"
	/usr/local/lib
	NO_DEFAULT_PATH)
#	find_file(SDL2_DLL
#	NAMES libSDL2.so
# 	PATHS "${SDL2_ROOT_DIR}/lib/x64/linux"
#	/usr/local/lib
#	NO_DEFAULT_PATH)
	# on linux dl and pthread are also required
#	set(SDL2_LIBRARIES ${SDL2_LIBRARY} ${SDL2main_LIBRARY} dl pthread)
	set(CUDA_LIBS ${CUDA_LIBRARY} ${NPPC_LIBRARY} ${NPPI_LIBRARY} ${CULIBOS_LIBRARY} dl rt)
else()
	find_library(CUDA_LIBRARY
	NAMES cudart_static
 	PATHS "${CUDA_ROOT_DIR}/lib/x64/win64"
	NO_DEFAULT_PATH)

	find_library(NPPC_LIBRARY
	NAMES nppc
  	PATHS "${CUDA_ROOT_DIR}/lib/x64/win64")

	find_library(NPPI_LIBRARY
	NAMES nppi
  	PATHS "${CUDA_ROOT_DIR}/lib/x64/win64")
	
#	find_file(SDL2_DLL
#	NAMES SDL2.dll
# 	PATHS "${SDL2_ROOT_DIR}/lib/x86/win32")
#	set(SDL2_LIBRARIES ${SDL2_LIBRARY} ${SDL2main_LIBRARY})
	set(CUDA_LIBS ${CUDA_LIBRARY} ${NPPC_LIBRARY} ${NPPI_LIBRARY})
endif()
message("${CUDA_LIBRARY}")  

set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIR} )

IF (CUDA_INCLUDE_DIR)
SET( CUDA_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
message("local CUDA found!")
ELSE (CUDA_INCLUDE_DIR)
SET( CUDA_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (CUDA_INCLUDE_DIR)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set <module>_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(CUDA
  DEFAULT_MSG
  CUDA_LIBRARY
  CUDA_INCLUDE_DIR
  CUDA_SDK_ROOT_DIR)

mark_as_advanced(CUDA_INCLUDE_DIR CUDA_LIBRARIES CUDA_SDK_ROOT_DIR)

##############################################################################
# Helper to add the include directory for CUDA only once
function(CUDA_ADD_CUDA_INCLUDE_ONCE)
  get_directory_property(_include_directories INCLUDE_DIRECTORIES)
  set(_add TRUE)
  if(_include_directories)
    foreach(dir ${_include_directories})
      if("${dir}" STREQUAL "${CUDA_INCLUDE_DIRS}")
        set(_add FALSE)
      endif()
    endforeach()
  endif()
  if(_add)
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()
endfunction()

##############################################################################
# Parse the OPTIONS from ARGN and set the variables prefixed by _option_prefix
#
macro(CUDA_PARSE_NVCC_OPTIONS _option_prefix)
  set( _found_config )
  foreach(arg ${ARGN})
    # Determine if we are dealing with a perconfiguration flag
    foreach(config ${CUDA_configuration_types})
      string(TOUPPER ${config} config_upper)
      if (arg STREQUAL "${config_upper}")
        set( _found_config _${arg})
        # Set arg to nothing to keep it from being processed further
        set( arg )
      endif()
    endforeach()

    if ( arg )
      list(APPEND ${_option_prefix}${_found_config} "${arg}")
    endif()
  endforeach()
endmacro()

##############################################################################
# Separate the OPTIONS out from the sources
#
macro(CUDA_GET_SOURCES_AND_OPTIONS _sources _cmake_options _options)
  set( ${_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    if("x${arg}" STREQUAL "xOPTIONS")
      set( _found_options TRUE )
    elseif(
        "x${arg}" STREQUAL "xWIN32" OR
        "x${arg}" STREQUAL "xMACOSX_BUNDLE" OR
        "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
        "x${arg}" STREQUAL "xSTATIC" OR
        "x${arg}" STREQUAL "xSHARED" OR
        "x${arg}" STREQUAL "xMODULE"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if ( _found_options )
        list(APPEND ${_options} ${arg})
      else()
        # Assume this is a file
        list(APPEND ${_sources} ${arg})
      endif()
    endif()
  endforeach()
endmacro()


###############################################################################
###############################################################################
# Separable Compilation Link
###############################################################################
###############################################################################

# Compute the filename to be used by CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS
function(CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME output_file_var cuda_target object_files)
  if (object_files)
    set(generated_extension ${CMAKE_${CUDA_C_OR_CXX}_OUTPUT_EXTENSION})
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${cuda_target}.dir/${CMAKE_CFG_INTDIR}/${cuda_target}_intermediate_link${generated_extension}")
  else()
    set(output_file)
  endif()

  set(${output_file_var} "${output_file}" PARENT_SCOPE)
endfunction()

# Setup the build rule for the separable compilation intermediate link file.
function(CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS output_file cuda_target options object_files)
  if (object_files)

    set_source_files_properties("${output_file}"
      PROPERTIES
      EXTERNAL_OBJECT TRUE # This is an object file not to be compiled, but only
                           # be linked.
      GENERATED TRUE       # This file is generated during the build
      )

    # For now we are ignoring all the configuration specific flags.
    set(nvcc_flags)
    CUDA_PARSE_NVCC_OPTIONS(nvcc_flags ${options})
    if(CUDA_64_BIT_DEVICE_CODE)
      list(APPEND nvcc_flags -m64)
    else()
      list(APPEND nvcc_flags -m32)
    endif()
    # If -ccbin, --compiler-bindir has been specified, don't do anything.  Otherwise add it here.
    list( FIND nvcc_flags "-ccbin" ccbin_found0 )
    list( FIND nvcc_flags "--compiler-bindir" ccbin_found1 )
    if( ccbin_found0 LESS 0 AND ccbin_found1 LESS 0 AND CUDA_HOST_COMPILER )
      list(APPEND nvcc_flags -ccbin "\"${CUDA_HOST_COMPILER}\"")
    endif()

    # Create a list of flags specified by CUDA_NVCC_FLAGS_${CONFIG} and CMAKE_${CUDA_C_OR_CXX}_FLAGS*
    set(config_specific_flags)
    set(flags)
    foreach(config ${CUDA_configuration_types})
      string(TOUPPER ${config} config_upper)
      # Add config specific flags
      foreach(f ${CUDA_NVCC_FLAGS_${config_upper}})
        list(APPEND config_specific_flags $<$<CONFIG:${config}>:${f}>)
      endforeach()
      set(important_host_flags)
      _cuda_get_important_host_flags(important_host_flags ${CMAKE_${CUDA_C_OR_CXX}_FLAGS_${config_upper}})
      foreach(f ${important_host_flags})
        list(APPEND flags $<$<CONFIG:${config}>:-Xcompiler> $<$<CONFIG:${config}>:${f}>)
      endforeach()
    endforeach()
    # Add CMAKE_${CUDA_C_OR_CXX}_FLAGS
    set(important_host_flags)
    _cuda_get_important_host_flags(important_host_flags ${CMAKE_${CUDA_C_OR_CXX}_FLAGS})
    foreach(f ${important_host_flags})
      list(APPEND flags -Xcompiler ${f})
    endforeach()

    # Add our general CUDA_NVCC_FLAGS with the configuration specifig flags
    set(nvcc_flags ${CUDA_NVCC_FLAGS} ${config_specific_flags} ${nvcc_flags})

    file(RELATIVE_PATH output_file_relative_path "${CMAKE_BINARY_DIR}" "${output_file}")

    # Some generators don't handle the multiple levels of custom command
    # dependencies correctly (obj1 depends on file1, obj2 depends on obj1), so
    # we work around that issue by compiling the intermediate link object as a
    # pre-link custom command in that situation.
    set(do_obj_build_rule TRUE)
    if (MSVC_VERSION GREATER 1599)
      # VS 2010 and 2012 have this problem.  If future versions fix this issue,
      # it should still work, it just won't be as nice as the other method.
      set(do_obj_build_rule FALSE)
    endif()

    if (do_obj_build_rule)
      add_custom_command(
        OUTPUT ${output_file}
        DEPENDS ${object_files}
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${nvcc_flags} -dlink ${object_files} -o ${output_file}
        ${flags}
        COMMENT "Building NVCC intermediate link file ${output_file_relative_path}"
        )
    else()
      add_custom_command(
        TARGET ${cuda_target}
        PRE_LINK
        COMMAND ${CMAKE_COMMAND} -E echo "Building NVCC intermediate link file ${output_file_relative_path}"
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${nvcc_flags} ${flags} -dlink ${object_files} -o "${output_file}"
        )
    endif()
 endif()
endfunction()

##############################################################################
# Helper to avoid clashes of files with the same basename but different paths.
# This doesn't attempt to do exactly what CMake internals do, which is to only
# add this path when there is a conflict, since by the time a second collision
# in names is detected it's already too late to fix the first one.  For
# consistency sake the relative path will be added to all files.
function(CUDA_COMPUTE_BUILD_PATH path build_path)
  #message("CUDA_COMPUTE_BUILD_PATH([${path}] ${build_path})")
  # Only deal with CMake style paths from here on out
  file(TO_CMAKE_PATH "${path}" bpath)
  if (IS_ABSOLUTE "${bpath}")
    # Absolute paths are generally unnessary, especially if something like
    # file(GLOB_RECURSE) is used to pick up the files.

    string(FIND "${bpath}" "${CMAKE_CURRENT_BINARY_DIR}" _binary_dir_pos)
    if (_binary_dir_pos EQUAL 0)
      file(RELATIVE_PATH bpath "${CMAKE_CURRENT_BINARY_DIR}" "${bpath}")
    else()
      file(RELATIVE_PATH bpath "${CMAKE_CURRENT_SOURCE_DIR}" "${bpath}")
    endif()
  endif()

  # This recipe is from cmLocalGenerator::CreateSafeUniqueObjectFileName in the
  # CMake source.

  # Remove leading /
  string(REGEX REPLACE "^[/]+" "" bpath "${bpath}")
  # Avoid absolute paths by removing ':'
  string(REPLACE ":" "_" bpath "${bpath}")
  # Avoid relative paths that go up the tree
  string(REPLACE "../" "__/" bpath "${bpath}")
  # Avoid spaces
  string(REPLACE " " "_" bpath "${bpath}")

  # Strip off the filename.  I wait until here to do it, since removin the
  # basename can make a path that looked like path/../basename turn into
  # path/.. (notice the trailing slash).
  get_filename_component(bpath "${bpath}" PATH)

  set(${build_path} "${bpath}" PARENT_SCOPE)
  #message("${build_path} = ${bpath}")
endfunction()



#####################################################################
## CUDA_INCLUDE_NVCC_DEPENDENCIES
##

# So we want to try and include the dependency file if it exists.  If
# it doesn't exist then we need to create an empty one, so we can
# include it.

# If it does exist, then we need to check to see if all the files it
# depends on exist.  If they don't then we should clear the dependency
# file and regenerate it later.  This covers the case where a header
# file has disappeared or moved.

macro(CUDA_INCLUDE_NVCC_DEPENDENCIES dependency_file)
  set(CUDA_NVCC_DEPEND)
  set(CUDA_NVCC_DEPEND_REGENERATE FALSE)


  # Include the dependency file.  Create it first if it doesn't exist .  The
  # INCLUDE puts a dependency that will force CMake to rerun and bring in the
  # new info when it changes.  DO NOT REMOVE THIS (as I did and spent a few
  # hours figuring out why it didn't work.
  if(NOT EXISTS ${dependency_file})
    file(WRITE ${dependency_file} "#FindCUDA.cmake generated file.  Do not edit.\n")
  endif()
  # Always include this file to force CMake to run again next
  # invocation and rebuild the dependencies.
  #message("including dependency_file = ${dependency_file}")
  include(${dependency_file})

  # Now we need to verify the existence of all the included files
  # here.  If they aren't there we need to just blank this variable and
  # make the file regenerate again.
#   if(DEFINED CUDA_NVCC_DEPEND)
#     message("CUDA_NVCC_DEPEND set")
#   else()
#     message("CUDA_NVCC_DEPEND NOT set")
#   endif()
  if(CUDA_NVCC_DEPEND)
    #message("CUDA_NVCC_DEPEND found")
    foreach(f ${CUDA_NVCC_DEPEND})
      # message("searching for ${f}")
      if(NOT EXISTS ${f})
        #message("file ${f} not found")
        set(CUDA_NVCC_DEPEND_REGENERATE TRUE)
      endif()
    endforeach()
  else()
    #message("CUDA_NVCC_DEPEND false")
    # No dependencies, so regenerate the file.
    set(CUDA_NVCC_DEPEND_REGENERATE TRUE)
  endif()

  #message("CUDA_NVCC_DEPEND_REGENERATE = ${CUDA_NVCC_DEPEND_REGENERATE}")
  # No incoming dependencies, so we need to generate them.  Make the
  # output depend on the dependency file itself, which should cause the
  # rule to re-run.
  if(CUDA_NVCC_DEPEND_REGENERATE)
    set(CUDA_NVCC_DEPEND ${dependency_file})
    #message("Generating an empty dependency_file: ${dependency_file}")
    file(WRITE ${dependency_file} "#FindCUDA.cmake generated file.  Do not edit.\n")
  endif()

endmacro()



##############################################################################
# This helper macro populates the following variables and setups up custom
# commands and targets to invoke the nvcc compiler to generate C or PTX source
# dependent upon the format parameter.  The compiler is invoked once with -M
# to generate a dependency file and a second time with -cuda or -ptx to generate
# a .cpp or .ptx file.
# INPUT:
#   cuda_target         - Target name
#   format              - PTX, CUBIN, FATBIN or OBJ
#   FILE1 .. FILEN      - The remaining arguments are the sources to be wrapped.
#   OPTIONS             - Extra options to NVCC
# OUTPUT:
#   generated_files     - List of generated files
##############################################################################
##############################################################################

##############################################################################
# This helper macro populates the following variables and setups up custom
# commands and targets to invoke the nvcc compiler to generate C or PTX source
# dependent upon the format parameter.  The compiler is invoked once with -M
# to generate a dependency file and a second time with -cuda or -ptx to generate
# a .cpp or .ptx file.
# INPUT:
#   cuda_target         - Target name
#   format              - PTX, CUBIN, FATBIN or OBJ
#   FILE1 .. FILEN      - The remaining arguments are the sources to be wrapped.
#   OPTIONS             - Extra options to NVCC
# OUTPUT:
#   generated_files     - List of generated files
##############################################################################
##############################################################################

macro(CUDA_WRAP_SRCS cuda_target format generated_files)

  # If CMake doesn't support separable compilation, complain
  if(CUDA_SEPARABLE_COMPILATION AND CMAKE_VERSION VERSION_LESS "2.8.10.1")
    message(SEND_ERROR "CUDA_SEPARABLE_COMPILATION isn't supported for CMake versions less than 2.8.10.1")
  endif()

  # Set up all the command line flags here, so that they can be overridden on a per target basis.

  set(nvcc_flags "")

  # Emulation if the card isn't present.
  if (CUDA_BUILD_EMULATION)
    # Emulation.
    set(nvcc_flags ${nvcc_flags} --device-emulation -D_DEVICEEMU -g)
  else()
    # Device mode.  No flags necessary.
  endif()

  if(CUDA_HOST_COMPILATION_CPP)
    set(CUDA_C_OR_CXX CXX)
  else()
    if(CUDA_VERSION VERSION_LESS "3.0")
      set(nvcc_flags ${nvcc_flags} --host-compilation C)
    else()
      message(WARNING "--host-compilation flag is deprecated in CUDA version >= 3.0.  Removing --host-compilation C flag" )
    endif()
    set(CUDA_C_OR_CXX C)
  endif()

  set(generated_extension ${CMAKE_${CUDA_C_OR_CXX}_OUTPUT_EXTENSION})

  if(CUDA_64_BIT_DEVICE_CODE)
    set(nvcc_flags ${nvcc_flags} -m64)
  else()
    set(nvcc_flags ${nvcc_flags} -m32)
  endif()

  if(CUDA_TARGET_CPU_ARCH)
    set(nvcc_flags ${nvcc_flags} "--target-cpu-architecture=${CUDA_TARGET_CPU_ARCH}")
  endif()

  # This needs to be passed in at this stage, because VS needs to fill out the
  # value of VCInstallDir from within VS.  Note that CCBIN is only used if
  # -ccbin or --compiler-bindir isn't used and CUDA_HOST_COMPILER matches
  # $(VCInstallDir)/bin.
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(ccbin_flags -D "\"CCBIN:PATH=$(VCInstallDir)bin\"" )
  else()
    set(ccbin_flags)
  endif()

  # Figure out which configure we will use and pass that in as an argument to
  # the script.  We need to defer the decision until compilation time, because
  # for VS projects we won't know if we are making a debug or release build
  # until build time.
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set( CUDA_build_configuration "$(ConfigurationName)" )
  else()
    set( CUDA_build_configuration "${CMAKE_BUILD_TYPE}")
  endif()

  # Initialize our list of includes with the user ones followed by the CUDA system ones.
  set(CUDA_NVCC_INCLUDE_ARGS ${CUDA_NVCC_INCLUDE_ARGS_USER} "-I${CUDA_INCLUDE_DIRS}")
  # Get the include directories for this directory and use them for our nvcc command.
  # Remove duplicate entries which may be present since include_directories
  # in CMake >= 2.8.8 does not remove them.
  get_directory_property(CUDA_NVCC_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES CUDA_NVCC_INCLUDE_DIRECTORIES)
  if(CUDA_NVCC_INCLUDE_DIRECTORIES)
    foreach(dir ${CUDA_NVCC_INCLUDE_DIRECTORIES})
      list(APPEND CUDA_NVCC_INCLUDE_ARGS -I${dir})
    endforeach()
  endif()

  # Reset these variables
  set(CUDA_WRAP_OPTION_NVCC_FLAGS)
  foreach(config ${CUDA_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(CUDA_WRAP_OPTION_NVCC_FLAGS_${config_upper})
  endforeach()

  CUDA_GET_SOURCES_AND_OPTIONS(_cuda_wrap_sources _cuda_wrap_cmake_options _cuda_wrap_options ${ARGN})
  CUDA_PARSE_NVCC_OPTIONS(CUDA_WRAP_OPTION_NVCC_FLAGS ${_cuda_wrap_options})

  # Figure out if we are building a shared library.  BUILD_SHARED_LIBS is
  # respected in CUDA_ADD_LIBRARY.
  set(_cuda_build_shared_libs FALSE)
  # SHARED, MODULE
  list(FIND _cuda_wrap_cmake_options SHARED _cuda_found_SHARED)
  list(FIND _cuda_wrap_cmake_options MODULE _cuda_found_MODULE)
  if(_cuda_found_SHARED GREATER -1 OR _cuda_found_MODULE GREATER -1)
    set(_cuda_build_shared_libs TRUE)
  endif()
  # STATIC
  list(FIND _cuda_wrap_cmake_options STATIC _cuda_found_STATIC)
  if(_cuda_found_STATIC GREATER -1)
    set(_cuda_build_shared_libs FALSE)
  endif()

  # CUDA_HOST_FLAGS
  if(_cuda_build_shared_libs)
    # If we are setting up code for a shared library, then we need to add extra flags for
    # compiling objects for shared libraries.
    set(CUDA_HOST_SHARED_FLAGS ${CMAKE_SHARED_LIBRARY_${CUDA_C_OR_CXX}_FLAGS})
  else()
    set(CUDA_HOST_SHARED_FLAGS)
  endif()
  # Only add the CMAKE_{C,CXX}_FLAGS if we are propagating host flags.  We
  # always need to set the SHARED_FLAGS, though.
  if(CUDA_PROPAGATE_HOST_FLAGS)
    set(_cuda_host_flags "set(CMAKE_HOST_FLAGS ${CMAKE_${CUDA_C_OR_CXX}_FLAGS} ${CUDA_HOST_SHARED_FLAGS})")
  else()
    set(_cuda_host_flags "set(CMAKE_HOST_FLAGS ${CUDA_HOST_SHARED_FLAGS})")
  endif()

  set(_cuda_nvcc_flags_config "# Build specific configuration flags")
  # Loop over all the configuration types to generate appropriate flags for run_nvcc.cmake
  foreach(config ${CUDA_configuration_types})
    string(TOUPPER ${config} config_upper)
    # CMAKE_FLAGS are strings and not lists.  By not putting quotes around CMAKE_FLAGS
    # we convert the strings to lists (like we want).

    if(CUDA_PROPAGATE_HOST_FLAGS)
      # nvcc chokes on -g3 in versions previous to 3.0, so replace it with -g
      set(_cuda_fix_g3 FALSE)

      if(CMAKE_COMPILER_IS_GNUCC)
        if (CUDA_VERSION VERSION_LESS  "3.0" OR
            CUDA_VERSION VERSION_EQUAL "4.1" OR
            CUDA_VERSION VERSION_EQUAL "4.2"
            )
          set(_cuda_fix_g3 TRUE)
        endif()
      endif()
      if(_cuda_fix_g3)
        string(REPLACE "-g3" "-g" _cuda_C_FLAGS "${CMAKE_${CUDA_C_OR_CXX}_FLAGS_${config_upper}}")
      else()
        set(_cuda_C_FLAGS "${CMAKE_${CUDA_C_OR_CXX}_FLAGS_${config_upper}}")
      endif()

      set(_cuda_host_flags "${_cuda_host_flags}\nset(CMAKE_HOST_FLAGS_${config_upper} ${_cuda_C_FLAGS})")
    endif()

    # Note that if we ever want CUDA_NVCC_FLAGS_<CONFIG> to be string (instead of a list
    # like it is currently), we can remove the quotes around the
    # ${CUDA_NVCC_FLAGS_${config_upper}} variable like the CMAKE_HOST_FLAGS_<CONFIG> variable.
    set(_cuda_nvcc_flags_config "${_cuda_nvcc_flags_config}\nset(CUDA_NVCC_FLAGS_${config_upper} ${CUDA_NVCC_FLAGS_${config_upper}} ;; ${CUDA_WRAP_OPTION_NVCC_FLAGS_${config_upper}})")
  endforeach()

  # Get the list of definitions from the directory property
  get_directory_property(CUDA_NVCC_DEFINITIONS COMPILE_DEFINITIONS)
  if(CUDA_NVCC_DEFINITIONS)
    foreach(_definition ${CUDA_NVCC_DEFINITIONS})
      list(APPEND nvcc_flags "-D${_definition}")
    endforeach()
  endif()

  if(_cuda_build_shared_libs)
    list(APPEND nvcc_flags "-D${cuda_target}_EXPORTS")
  endif()

  # Reset the output variable
  set(_cuda_wrap_generated_files "")

  # Iterate over the macro arguments and create custom
  # commands for all the .cu files.
  foreach(file ${ARGN})
    # Ignore any file marked as a HEADER_FILE_ONLY
    get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
    if(${file} MATCHES "\\.cu$" AND NOT _is_header)

      # Allow per source file overrides of the format.
      get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
      if(NOT _cuda_source_format)
        set(_cuda_source_format ${format})
      endif()

      if( ${_cuda_source_format} MATCHES "OBJ")
        set( cuda_compile_to_external_module OFF )
      else()
        set( cuda_compile_to_external_module ON )
        if( ${_cuda_source_format} MATCHES "PTX" )
          set( cuda_compile_to_external_module_type "ptx" )
        elseif( ${_cuda_source_format} MATCHES "CUBIN")
          set( cuda_compile_to_external_module_type "cubin" )
        elseif( ${_cuda_source_format} MATCHES "FATBIN")
          set( cuda_compile_to_external_module_type "fatbin" )
        else()
          message( FATAL_ERROR "Invalid format flag passed to CUDA_WRAP_SRCS for file '${file}': '${_cuda_source_format}'.  Use OBJ, PTX, CUBIN or FATBIN.")
        endif()
      endif()

      if(cuda_compile_to_external_module)
        # Don't use any of the host compilation flags for PTX targets.
        set(CUDA_HOST_FLAGS)
        set(CUDA_NVCC_FLAGS_CONFIG)
      else()
        set(CUDA_HOST_FLAGS ${_cuda_host_flags})
        set(CUDA_NVCC_FLAGS_CONFIG ${_cuda_nvcc_flags_config})
      endif()

      # Determine output directory
      cuda_compute_build_path("${file}" cuda_build_path)
      set(cuda_compile_intermediate_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${cuda_target}.dir/${cuda_build_path}")
      if(CUDA_GENERATED_OUTPUT_DIR)
        set(cuda_compile_output_dir "${CUDA_GENERATED_OUTPUT_DIR}")
      else()
        if ( cuda_compile_to_external_module )
          set(cuda_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
        else()
          set(cuda_compile_output_dir "${cuda_compile_intermediate_directory}")
        endif()
      endif()

      # Add a custom target to generate a c or ptx file. ######################

      get_filename_component( basename ${file} NAME )
      if( cuda_compile_to_external_module )
        set(generated_file_path "${cuda_compile_output_dir}")
        set(generated_file_basename "${cuda_target}_generated_${basename}.${cuda_compile_to_external_module_type}")
        set(format_flag "-${cuda_compile_to_external_module_type}")
        file(MAKE_DIRECTORY "${cuda_compile_output_dir}")
      else()
        set(generated_file_path "${cuda_compile_output_dir}/${CMAKE_CFG_INTDIR}")
        set(generated_file_basename "${cuda_target}_generated_${basename}${generated_extension}")
        if(CUDA_SEPARABLE_COMPILATION)
          set(format_flag "-dc")
        else()
          set(format_flag "-c")
        endif()
      endif()

      # Set all of our file names.  Make sure that whatever filenames that have
      # generated_file_path in them get passed in through as a command line
      # argument, so that the ${CMAKE_CFG_INTDIR} gets expanded at run time
      # instead of configure time.
      set(generated_file "${generated_file_path}/${generated_file_basename}")
      set(cmake_dependency_file "${cuda_compile_intermediate_directory}/${generated_file_basename}.depend")
      set(NVCC_generated_dependency_file "${cuda_compile_intermediate_directory}/${generated_file_basename}.NVCC-depend")
      set(generated_cubin_file "${generated_file_path}/${generated_file_basename}.cubin.txt")
      set(custom_target_script "${cuda_compile_intermediate_directory}/${generated_file_basename}.cmake")

      # Setup properties for obj files:
      if( NOT cuda_compile_to_external_module )
        set_source_files_properties("${generated_file}"
          PROPERTIES
          EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked.
          )
      endif()

      # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path.
      get_filename_component(file_path "${file}" PATH)
      if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
      else()
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      endif()

      if( NOT cuda_compile_to_external_module AND CUDA_SEPARABLE_COMPILATION)
        list(APPEND ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS "${generated_file}")
      endif()

      # Bring in the dependencies.  Creates a variable CUDA_NVCC_DEPEND #######
      cuda_include_nvcc_dependencies(${cmake_dependency_file})

      # Convience string for output ###########################################
      if(CUDA_BUILD_EMULATION)
        set(cuda_build_type "Emulation")
      else()
        set(cuda_build_type "Device")
      endif()

      # Build the NVCC made dependency file ###################################
      set(build_cubin OFF)
      if ( NOT CUDA_BUILD_EMULATION AND CUDA_BUILD_CUBIN )
         if ( NOT cuda_compile_to_external_module )
           set ( build_cubin ON )
         endif()
      endif()

      # Configure the build script
	  message("${CUDA_run_nvcc}")
	  message("${custom_target_script}")     
	  configure_file("${CUDA_run_nvcc}" "${custom_target_script}" @ONLY)

      # So if a user specifies the same cuda file as input more than once, you
      # can have bad things happen with dependencies.  Here we check an option
      # to see if this is the behavior they want.
      if(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE)
        set(main_dep MAIN_DEPENDENCY ${source_file})
      else()
        set(main_dep DEPENDS ${source_file})
      endif()

      if(CUDA_VERBOSE_BUILD)
        set(verbose_output ON)
      elseif(CMAKE_GENERATOR MATCHES "Makefiles")
        set(verbose_output "$(VERBOSE)")
      else()
        set(verbose_output OFF)
      endif()

      # Create up the comment string
      file(RELATIVE_PATH generated_file_relative_path "${CMAKE_BINARY_DIR}" "${generated_file}")
      if(cuda_compile_to_external_module)
        set(cuda_build_comment_string "Building NVCC ${cuda_compile_to_external_module_type} file ${generated_file_relative_path}")
      else()
        set(cuda_build_comment_string "Building NVCC (${cuda_build_type}) object ${generated_file_relative_path}")
      endif()

      # Build the generated file and dependency file ##########################
      add_custom_command(
        OUTPUT ${generated_file}
        # These output files depend on the source_file and the contents of cmake_dependency_file
        ${main_dep}
        DEPENDS ${CUDA_NVCC_DEPEND}
        DEPENDS ${custom_target_script}
        # Make sure the output directory exists before trying to write to it.
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
        COMMAND ${CMAKE_COMMAND} ARGS
          -D verbose:BOOL=${verbose_output}
          ${ccbin_flags}
          -D build_configuration:STRING=${CUDA_build_configuration}
          -D "generated_file:STRING=${generated_file}"
          -D "generated_cubin_file:STRING=${generated_cubin_file}"
          -P "${custom_target_script}"
        WORKING_DIRECTORY "${cuda_compile_intermediate_directory}"
        COMMENT "${cuda_build_comment_string}"
        )

      # Make sure the build system knows the file is generated.
      set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)

      list(APPEND _cuda_wrap_generated_files ${generated_file})

      # Add the other files that we want cmake to clean on a cleanup ##########
      list(APPEND CUDA_ADDITIONAL_CLEAN_FILES "${cmake_dependency_file}")
      list(REMOVE_DUPLICATES CUDA_ADDITIONAL_CLEAN_FILES)
      set(CUDA_ADDITIONAL_CLEAN_FILES ${CUDA_ADDITIONAL_CLEAN_FILES} CACHE INTERNAL "List of intermediate files that are part of the cuda dependency scanning.")

    endif()
  endforeach()

  # Set the return parameter
  set(${generated_files} ${_cuda_wrap_generated_files})
endmacro()



# Helper to add the include directory for CUDA only once
function(CUDA_ADD_CUDA_INCLUDE_ONCE)
  get_directory_property(_include_directories INCLUDE_DIRECTORIES)
  set(_add TRUE)
  if(_include_directories)
    foreach(dir ${_include_directories})
      if("${dir}" STREQUAL "${CUDA_INCLUDE_DIRS}")
        set(_add FALSE)
      endif()
    endforeach()
  endif()
  if(_add)
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()
endfunction()

function(CUDA_BUILD_SHARED_LIBRARY shared_flag)
  set(cmake_args ${ARGN})
  # If SHARED, MODULE, or STATIC aren't already in the list of arguments, then
  # add SHARED or STATIC based on the value of BUILD_SHARED_LIBS.
  list(FIND cmake_args SHARED _cuda_found_SHARED)
  list(FIND cmake_args MODULE _cuda_found_MODULE)
  list(FIND cmake_args STATIC _cuda_found_STATIC)
  if( _cuda_found_SHARED GREATER -1 OR
      _cuda_found_MODULE GREATER -1 OR
      _cuda_found_STATIC GREATER -1)
    set(_cuda_build_shared_libs)
  else()
    if (BUILD_SHARED_LIBS)
      set(_cuda_build_shared_libs SHARED)
    else()
      set(_cuda_build_shared_libs STATIC)
    endif()
  endif()
  set(${shared_flag} ${_cuda_build_shared_libs} PARENT_SCOPE)
endfunction()

###############################################################################
###############################################################################
# ADD LIBRARY
###############################################################################
###############################################################################
macro(CUDA_ADD_LIBRARY cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  CUDA_BUILD_SHARED_LIBRARY(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${cuda_target}
    ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()


###############################################################################
###############################################################################
# ADD EXECUTABLE
###############################################################################
###############################################################################
macro(CUDA_ADD_EXECUTABLE cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_executable(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${cuda_target}
    ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )
endmacro()


