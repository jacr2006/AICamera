cmake_minimum_required(VERSION 3.4.1)

add_library(
             native-lib
             SHARED
             src/main/cpp/native-lib.cpp
             )
find_library(
          jnigraphics
          android-lib
          android
          )

include(AndroidNdkModules)
android_ndk_import_module_cpufeatures()

add_library(
    c10
    STATIC
    IMPORTED
    )
set_target_properties(
        c10
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libc10.a
)
add_library(
    thread_pool
    STATIC
    IMPORTED
    )
set_target_properties(
    thread_pool
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libpthreadpool.a
    )
add_library(
    clog
    STATIC
    IMPORTED
    )
set_target_properties(
    clog
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libclog.a
    )

add_library(
    protobuf
    STATIC
    IMPORTED
    )
set_target_properties(
    protobuf
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libprotobuf.a
    )


add_library(
        torch
        SHARED
        IMPORTED
)
set_target_properties(
        torch
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libtorch.so
)

include_directories( src/main/cpp )

find_library(
     log-lib
     log
     )

target_link_libraries(
        native-lib
        torch
        jnigraphics
        ${log-lib}
        ${android-lib})