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
  src/network.cpp
  src/train.cpp
  src/transformer.cpp)
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
add_executable(test_train tests/test_train.cpp)
target_link_libraries(test_train fheml)
add_test(NAME test_train COMMAND test_train)

# Python bindings: build/python/fheml (add build/python to PYTHONPATH).
option(FHEML_PYTHON "Build the Python module (needs pybind11)" ON)
if(FHEML_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module)
  if(Python3_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                    OUTPUT_VARIABLE FHEML_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  find_package(pybind11 CONFIG HINTS ${FHEML_PYBIND11_DIR})
  if(pybind11_FOUND)
    set_target_properties(fheml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    pybind11_add_module(_fheml python/bindings.cpp)
    target_link_libraries(_fheml PRIVATE fheml)
    set_target_properties(_fheml PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/fheml)
    foreach(f __init__.py nn.py export.py)
      configure_file(python/fheml/${f} ${CMAKE_BINARY_DIR}/python/fheml/${f} COPYONLY)
    endforeach()
  else()
    message(STATUS "pybind11 not found: skipping the Python module (pip install pybind11)")
  endif()
endif()
