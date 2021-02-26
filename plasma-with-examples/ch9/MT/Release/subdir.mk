################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Main.cpp \
../Output.cpp \
../PotentialSolver.cpp \
../Source.cpp \
../Species.cpp \
../World.cpp 

OBJS += \
./Main.o \
./Output.o \
./PotentialSolver.o \
./Source.o \
./Species.o \
./World.o 

CPP_DEPS += \
./Main.d \
./Output.d \
./PotentialSolver.d \
./Source.d \
./Species.d \
./World.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -O2 -g -pg -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


