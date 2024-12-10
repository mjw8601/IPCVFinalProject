rit_add_executable(classify_final_with_knn 
  SOURCES
    classify_final_with_knn.cpp
)

target_link_libraries(classify_final_with_knn
  rit::statistics_classifiers
  rit::statistics_data_readers
  rit::statistics_evaluators
  opencv_core
  opencv_highgui
)
