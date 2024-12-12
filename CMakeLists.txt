add_subdirectory(classifiers)
add_subdirectory(data_readers)
add_subdirectory(evaluators)
add_subdirectory(labeling)

rit_add_executable(knn_livedemo
  SOURCES
    knn_show.cpp
    knn_functions.cpp
)

target_link_libraries(knn_show 
  ${OpenCV_LIBS}     # All required opencv libraries
  ${Boost_LIBRARIES} # All required boost libraries
)
