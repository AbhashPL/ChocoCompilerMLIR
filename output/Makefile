choco_program.run:linked_program.s
	clang++ linked_program.s -o choco_program.run

linked_program.s:linked_program.ll
	llc linked_program.ll -o linked_program.s

linked_program.ll:runtime.ll
	llvm-link -S runtime.ll example.ll -o linked_program.ll

runtime.ll:runtime.cpp
	clang++ -S -emit-llvm runtime.cpp -o runtime.ll

clean:
	rm *.s *.run linked_program.ll runtime.ll