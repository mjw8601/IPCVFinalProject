rit_add_executable(knnconcept
  SOURCES
    knnconcept.cpp
)

target_link_libraries(knnconcept 
  Boost::filesystem
  Boost::program_options
  opencv_core
  opencv_highgui
  opencv_imgcodecs
  opencv_imgproc
)
