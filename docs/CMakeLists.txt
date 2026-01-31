cmake_minimum_required(VERSION 3.14)
project(lloyal_docs NONE)

include(FetchContent)

FetchContent_Declare(
  doxygen-awesome-css
  GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
  GIT_TAG        v2.4.1
)
FetchContent_MakeAvailable(doxygen-awesome-css)

find_package(Doxygen REQUIRED)

set(AWESOME_CSS_DIR "${doxygen-awesome-css_SOURCE_DIR}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  @ONLY
)

add_custom_target(docs
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  COMMENT "Generating API documentation with Doxygen + doxygen-awesome"
)
