rit_add_library(minkowski_distance
  SOURCES
  MinkowskiDistance.cpp
  HEADERS
  MinkowskiDistance.h
)

target_link_libraries(minkowski_distance
  PUBLIC 
    opencv_core
  PRIVATE
)
