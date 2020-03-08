######## A simple cmakelists.txt file for OpenCV() #############
cmake_minimum_required(VERSION 3.2)
PROJECT(NPDFace)
#########################################INCLUDE #################################
FIND_PACKAGE(OpenMP QUIET REQUIRED)  
FIND_PACKAGE(CUDA QUIET REQUIRED)
FIND_PACKAGE(OpenCV QUIET REQUIRED ) 
if(OPENMP_FOUND)  
message("OPENMP FOUND")  
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  
#########################################################################################
INCLUDE_DIRECTORIES("C:/OpenCV/include/" C:/OpenCV/include/opencv C:/OpenCV/include/opencv2)


CUDA_ADD_EXECUTABLE(NPDFace 
Bootstrap.cpp
Bootstrap.h
Configuration.cpp
Configuration.h
DQTree.cpp
DQTree.h
GentleAdaBoost.cpp
GentleAdaBoost.h
main.cpp
Message.h
Model.cpp
Model.h
NPDFeature.cpp
NPDFeature.h
ReadData.cpp
ReadData.h
minMSE.h
minMSE.cu
minMSE.cpp
)
TARGET_LINK_LIBRARIES (NPDFace ${OpenCV_LIBS} )
# ########## end ####################################
