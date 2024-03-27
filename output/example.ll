; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @print(i32)

define void @foo() {
  call void @print(i32 20)
  ret void
}

define i32 @choco_main() {
  call void @foo()
  ret i32 0
}
