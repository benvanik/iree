; ModuleID = 'kernels_cl.bc'
source_filename = "kernels.cl"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__oclc_unsafe_math_opt = hidden local_unnamed_addr addrspace(4) constant i8 0, align 1
@__oclc_daz_opt = hidden local_unnamed_addr addrspace(4) constant i8 0, align 1
@__oclc_correctly_rounded_sqrt32 = hidden local_unnamed_addr addrspace(4) constant i8 1, align 1
@__oclc_finite_only_opt = hidden local_unnamed_addr addrspace(4) constant i8 0, align 1
@__oclc_wavefrontsize64 = hidden local_unnamed_addr addrspace(4) constant i8 0, align 1
@__oclc_ISA_version = hidden local_unnamed_addr addrspace(4) constant i32 11000, align 4
@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define protected amdgpu_kernel void @add_one(i32 noundef %n, ptr addrspace(1) noalias nocapture noundef align 4 %buffer) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 {
entry:
  %call = tail call i64 @__ockl_get_global_id(i32 noundef 0) #2
  %conv = zext i32 %n to i64
  %cmp = icmp ult i64 %call, %conv
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds nuw i32, ptr addrspace(1) %buffer, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4, !tbaa !9
  %add = add i32 %0, 1
  store i32 %add, ptr addrspace(1) %arrayidx, align 4, !tbaa !9
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare i64 @__ockl_get_global_id(i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "amdgpu-flat-work-group-size"="1,256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" }
attributes #2 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!opencl.ocl.version = !{!3}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 1}
!5 = !{!"none", !"none"}
!6 = !{!"uint32_t", !"uint32_t*"}
!7 = !{!"uint", !"uint*"}
!8 = !{!"", !"restrict"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
