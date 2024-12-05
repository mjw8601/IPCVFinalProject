rit_add_executable(label_plates
  SOURCES
    label_plates.cpp
    ../knn_functions.cpp
)

target_link_libraries(label_plates 
  ${OpenCV_LIBS}     # All required opencv libraries
  ${Boost_LIBRARIES} # All required boost libraries
)
