mkdir build/
cd build/
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DIREE_BUILD_COMPILER=ON -DIREE_TARGET_BACKEND_ROCM=ON -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_HIP_TEST_TARGET_CHIP=gfx1100
cmake --build . --config Debug --target iree_hal_drivers_amdgpu_cts_amdgpu_all_command_buffer_test
