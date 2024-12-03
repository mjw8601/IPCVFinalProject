rit_add_executable(knnconcept
  SOURCES
    knnconcept.cpp
)

target_link_libraries(knnconcept 
  ${OpenCV_LIBS}     # All required opencv libraries
  ${Boost_LIBRARIES} # All required boost libraries
)
