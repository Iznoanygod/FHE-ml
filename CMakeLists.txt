cmake_minimum_required(VERSION 3.16)
project(fheml CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Point CMake at an OpenFHE install with -DOpenFHE_DIR=<prefix>/lib/OpenFHE
# or -DCMAKE_PREFIX_PATH=<prefix>.
find_package(OpenFHE 1.2 REQUIRED CONFIG)
find_package(OpenMP)
set(CMAKE_BUILD_RPATH ${OpenFHE_LIBDIR})
set(CMAKE_INSTALL_RPATH ${OpenFHE_LIBDIR})
# OpenFHE libraries depend on each other; DT_RPATH (unlike RUNPATH) covers that.
add_link_options(-Wl,--disable-new-dtags)

# OpenFHE exports -Werror; keep its other flags (MATHBACKEND must match).
string(REPLACE "-Werror" "" FHEML_OPENFHE_FLAGS "${OpenFHE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FHEML_OPENFHE_FLAGS}")

add_library(fheml
  src/context.cpp
  src/linear.cpp
  src/layers.cpp
  src/network.cpp)
target_include_directories(fheml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(fheml SYSTEM PUBLIC
  ${OpenFHE_INCLUDE}
  ${OpenFHE_INCLUDE}/third-party/include
  ${OpenFHE_INCLUDE}/core
  ${OpenFHE_INCLUDE}/pke
  ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(fheml PUBLIC ${OpenFHE_LIBDIR})
target_link_libraries(fheml PUBLIC ${OpenFHE_SHARED_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(fheml PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(fheml PRIVATE -Wall -Wextra)

add_executable(mnist_cnn examples/mnist_cnn.cpp)
target_link_libraries(mnist_cnn fheml)

add_executable(conv_demo examples/conv_demo.cpp)
target_link_libraries(conv_demo fheml)

enable_testing()
add_executable(test_fheml tests/test_fheml.cpp)
target_link_libraries(test_fheml fheml)
add_test(NAME test_fheml COMMAND test_fheml)
