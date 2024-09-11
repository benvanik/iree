; ModuleID = 'kernels_hip.bc'
source_filename = "kernels.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__const.__assert_fail.fmt = private unnamed_addr addrspace(4) constant [47 x i8] c"%s:%u: %s: Device-side assertion `%s' failed.\0A\00", align 16
@__hip_cuid_ = addrspace(1) global i8 0
@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_ to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress noreturn nounwind
define weak void @__cxa_pure_virtual() local_unnamed_addr #0 {
entry:
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: mustprogress noreturn nounwind
define weak void @__cxa_deleted_virtual() local_unnamed_addr #0 {
entry:
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: convergent mustprogress noinline nounwind
define weak hidden void @__assert_fail(ptr noundef %assertion, ptr noundef %file, i32 noundef %line, ptr noundef %function) local_unnamed_addr #2 {
entry:
  %fmt = alloca [47 x i8], align 16, addrspace(5)
  %fmt.ascast = addrspacecast ptr addrspace(5) %fmt to ptr
  call void @llvm.lifetime.start.p5(i64 47, ptr addrspace(5) %fmt) #9
  call void @llvm.memcpy.p0.p4.i64(ptr noundef nonnull align 16 dereferenceable(47) %fmt.ascast, ptr addrspace(4) noundef align 16 dereferenceable(47) @__const.__assert_fail.fmt, i64 47, i1 false)
  %call = tail call i64 @__ockl_fprintf_stderr_begin() #10
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %tmp.0 = phi ptr [ %fmt.ascast, %entry ], [ %incdec.ptr, %while.cond ]
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %tmp.0, i64 1
  %0 = load i8, ptr %tmp.0, align 1, !tbaa !5
  %tobool.not = icmp eq i8 %0, 0
  br i1 %tobool.not, label %while.end, label %while.cond, !llvm.loop !8

while.end:                                        ; preds = %while.cond
  %sub.ptr.lhs.cast = ptrtoint ptr %incdec.ptr to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %fmt.ascast to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sext = shl i64 %sub.ptr.sub, 32
  %conv3 = ashr exact i64 %sext, 32
  %call4 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %call, ptr noundef %fmt.ascast, i64 noundef %conv3, i32 noundef 0) #10
  br label %while.cond7

while.cond7:                                      ; preds = %while.cond7, %while.end
  %tmp6.0 = phi ptr [ %file, %while.end ], [ %incdec.ptr8, %while.cond7 ]
  %incdec.ptr8 = getelementptr inbounds nuw i8, ptr %tmp6.0, i64 1
  %1 = load i8, ptr %tmp6.0, align 1, !tbaa !5
  %tobool9.not = icmp eq i8 %1, 0
  br i1 %tobool9.not, label %while.end11, label %while.cond7, !llvm.loop !10

while.end11:                                      ; preds = %while.cond7
  %sub.ptr.lhs.cast12 = ptrtoint ptr %incdec.ptr8 to i64
  %sub.ptr.rhs.cast13 = ptrtoint ptr %file to i64
  %sub.ptr.sub14 = sub i64 %sub.ptr.lhs.cast12, %sub.ptr.rhs.cast13
  %sext69 = shl i64 %sub.ptr.sub14, 32
  %conv18 = ashr exact i64 %sext69, 32
  %call19 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %call4, ptr noundef %file, i64 noundef %conv18, i32 noundef 0) #10
  %conv20 = zext i32 %line to i64
  %call21 = call i64 @__ockl_fprintf_append_args(i64 noundef %call19, i32 noundef 1, i64 noundef %conv20, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i32 noundef 0) #10
  br label %while.cond24

while.cond24:                                     ; preds = %while.cond24, %while.end11
  %tmp23.0 = phi ptr [ %function, %while.end11 ], [ %incdec.ptr25, %while.cond24 ]
  %incdec.ptr25 = getelementptr inbounds nuw i8, ptr %tmp23.0, i64 1
  %2 = load i8, ptr %tmp23.0, align 1, !tbaa !5
  %tobool26.not = icmp eq i8 %2, 0
  br i1 %tobool26.not, label %while.end28, label %while.cond24, !llvm.loop !11

while.end28:                                      ; preds = %while.cond24
  %sub.ptr.lhs.cast29 = ptrtoint ptr %incdec.ptr25 to i64
  %sub.ptr.rhs.cast30 = ptrtoint ptr %function to i64
  %sub.ptr.sub31 = sub i64 %sub.ptr.lhs.cast29, %sub.ptr.rhs.cast30
  %sext70 = shl i64 %sub.ptr.sub31, 32
  %conv35 = ashr exact i64 %sext70, 32
  %call36 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %call21, ptr noundef %function, i64 noundef %conv35, i32 noundef 0) #10
  br label %while.cond39

while.cond39:                                     ; preds = %while.cond39, %while.end28
  %tmp38.0 = phi ptr [ %assertion, %while.end28 ], [ %incdec.ptr40, %while.cond39 ]
  %incdec.ptr40 = getelementptr inbounds nuw i8, ptr %tmp38.0, i64 1
  %3 = load i8, ptr %tmp38.0, align 1, !tbaa !5
  %tobool41.not = icmp eq i8 %3, 0
  br i1 %tobool41.not, label %while.end43, label %while.cond39, !llvm.loop !12

while.end43:                                      ; preds = %while.cond39
  %sub.ptr.lhs.cast44 = ptrtoint ptr %incdec.ptr40 to i64
  %sub.ptr.rhs.cast45 = ptrtoint ptr %assertion to i64
  %sub.ptr.sub46 = sub i64 %sub.ptr.lhs.cast44, %sub.ptr.rhs.cast45
  %sext71 = shl i64 %sub.ptr.sub46, 32
  %conv50 = ashr exact i64 %sext71, 32
  %call51 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %call36, ptr noundef %assertion, i64 noundef %conv50, i32 noundef 1) #10
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p5(i64 immarg, ptr addrspace(5) nocapture) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p4.i64(ptr noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #4

; Function Attrs: convergent nounwind
declare i64 @__ockl_fprintf_stderr_begin() local_unnamed_addr #5

; Function Attrs: convergent nounwind
declare i64 @__ockl_fprintf_append_string_n(i64 noundef, ptr noundef, i64 noundef, i32 noundef) local_unnamed_addr #5

; Function Attrs: convergent nounwind
declare i64 @__ockl_fprintf_append_args(i64 noundef, i32 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef, i32 noundef) local_unnamed_addr #5

; Function Attrs: mustprogress noinline nounwind
define weak hidden void @__assertfail() local_unnamed_addr #6 {
entry:
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(readwrite, inaccessiblemem: none)
define protected amdgpu_kernel void @add_one(i32 noundef %n, ptr addrspace(1) noundef %buffer.coerce) local_unnamed_addr #7 {
entry:
  %call.i8 = tail call i64 @__ockl_get_group_id(i32 noundef 0) #11
  %conv.i = trunc i64 %call.i8 to i32
  %call.i = tail call i64 @__ockl_get_local_size(i32 noundef 0) #11
  %conv.i9 = trunc i64 %call.i to i32
  %mul = mul i32 %conv.i9, %conv.i
  %call.i10 = tail call i64 @__ockl_get_local_id(i32 noundef 0) #11
  %conv.i11 = trunc i64 %call.i10 to i32
  %add = add i32 %mul, %conv.i11
  %cmp = icmp ult i32 %add, %n
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = ptrtoint ptr addrspace(1) %buffer.coerce to i64
  %1 = inttoptr i64 %0 to ptr
  %idxprom = zext i32 %add to i64
  %arrayidx = getelementptr inbounds nuw i32, ptr %1, i64 %idxprom
  %2 = load i32, ptr %arrayidx, align 4, !tbaa !13
  %add4 = add i32 %2, 1
  store i32 %add4, ptr %arrayidx, align 4, !tbaa !13
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare i64 @__ockl_get_group_id(i32 noundef) local_unnamed_addr #8

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare i64 @__ockl_get_local_size(i32 noundef) local_unnamed_addr #8

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare i64 @__ockl_get_local_id(i32 noundef) local_unnamed_addr #8

attributes #0 = { mustprogress noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { convergent mustprogress noinline nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" }
attributes #6 = { mustprogress noinline nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" }
attributes #7 = { convergent mustprogress nofree norecurse nounwind willreturn memory(readwrite, inaccessiblemem: none) "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #8 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" }
attributes #9 = { nounwind }
attributes #10 = { convergent nounwind }
attributes #11 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{!"clang version 20.0.0git"}
!5 = !{!6, !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = distinct !{!10, !9}
!11 = distinct !{!11, !9}
!12 = distinct !{!12, !9}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !6, i64 0}
