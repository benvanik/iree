#IREE_HAL_AMDGPU_LIBHSA_PATH=/home/nod/src/ROCR-Runtime/build/rocr/lib/
TRY=0
while true; do
  echo "try ${TRY}:"
  ROCR_VISIBLE_DEVICES=0 HSA_ALLOCATE_QUEUE_DEV_MEM=1 ./build/runtime/src/iree/hal/drivers/amdgpu/cts/amdgpu_all_command_buffer_test
  if [ $? -ne 0 ]; then
    echo "FAILED"
    break
  fi
  let TRY=${TRY}+1
done

