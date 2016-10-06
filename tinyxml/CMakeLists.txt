#set(TINYXML_PATH tinyxml )
INCLUDE_DIRECTORIES("./")

# Since TinyXML is only object files, we need to do some tricks:
SET(TINYXML_SOURCES
      tinyxml.h tinyxml.cpp
      tinystr.h tinystr.cpp
      tinyxmlerror.cpp
      tinyxmlparser.cpp)
ADD_LIBRARY(tinyxml STATIC ${TINYXML_SOURCES})

