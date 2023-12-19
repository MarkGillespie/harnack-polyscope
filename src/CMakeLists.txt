cmake_minimum_required(VERSION 2.8.9)

# Maybe stop from CMAKEing in the wrong place
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Go use the /build directory.")
endif()

add_executable(
  run
  main.cpp
  utils.cpp
  math_elliptic_integral.cpp
  generalized_barycentric_coordinates.cpp
  geometry_utils.cpp
)
target_include_directories(run PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
# borrow args.hxx directly from polyscope
target_include_directories(run PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../deps/polyscope/deps/args/")
target_link_libraries(run polyscope geometry-central)

target_include_directories(run PUBLIC ${FCPW_EIGEN_INCLUDES})
target_include_directories(run PUBLIC ${FCPW_ENOKI_INCLUDES})
target_link_libraries(run fcpw)

find_package(FFTW REQUIRED)

add_executable(
  dipoles
  dipoles.cpp
  utils.cpp
  math_elliptic_integral.cpp
  generalized_barycentric_coordinates.cpp
  geometry_utils.cpp
  poisson_problem.cpp
)
target_include_directories(dipoles PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
# borrow args.hxx directly from polyscope
target_include_directories(dipoles PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../deps/polyscope/deps/args/")
target_link_libraries(dipoles polyscope geometry-central)

target_include_directories(dipoles PUBLIC ${FCPW_EIGEN_INCLUDES})
target_include_directories(dipoles PUBLIC ${FCPW_ENOKI_INCLUDES} ${FFTW_INCLUDES})
target_link_libraries(dipoles fcpw ${FFTW_LIBRARIES})
