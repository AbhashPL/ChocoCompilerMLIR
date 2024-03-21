#ifndef CHOCO_OPS
#define CHOCO_OPS

include "ChocoDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// FunctionInterface provides support for interacting with operations that behave like functions. In particular, these operations ::-
    //   - must be symbols, i.e. have the `Symbol` trait.
    //   - must have a single region, that may be comprised with multiple blocks,
    //     that corresponds to the function body.
    //     * when this region is empty, the operation corresponds to an external
    //       function.
    //     * leading arguments of the first block of the region are treated as
    //       function arguments.

    // The function, aside from implementing the various interface methods,
    // should have the following ODS arguments:

    //   - `function_type` (required)
    //     * A TypeAttr that holds the signature type of the function.

    //   - `arg_attrs` (optional)
    //     * An ArrayAttr of DictionaryAttr that contains attribute dictionaries
    //       for each of the function arguments.

    //   - `res_attrs` (optional)
    //     * An ArrayAttr of DictionaryAttr that contains attribute dictionaries
    //       for each of the function results.



// Op is the base class from which this particular ConstOp inherits, it is provided with the name of the dialect and the
// mnemonic of this operation
def Choco_ConstOp : Op<Choco_Dialect, "const", []> {
    let summary = "const numeric op";
    let arguments = (ins I32Attr:$value);
    let results = (outs I32);

    // Add custom build methods for the constant operation. These method populates
    // the `state` that MLIR uses to create operations, i.e. these are used when
    // using `builder.create<ConstantOp>(...)`.
    let builders = [
        // Build a constant with a given int
         OpBuilder<(ins "int":$val)>,
    ];

}


def Choco_AddOp : Op<Choco_Dialect, "add", []> {
    let summary = "add numeric op";
    let arguments = (ins I32:$lhs, I32:$rhs);
    let results = (outs I32);
}


def Choco_MulOp : Op<Choco_Dialect, "mul", []> {
    let summary = "mul numeric op";
    let arguments = (ins I32:$lhs, I32:$rhs);
    let results = (outs I32);
}

def Choco_AssignOp : Op<Choco_Dialect, "assign", []> {
    let summary = "assign operation";
    let arguments = (ins I32:$var, I32:$value);
    let results = (outs); // No results
    let assemblyFormat = "$value `:` type($value) `to` $var `:` type($var) attr-dict";
}


// FuncOp

def FuncOp : Op<Choco_Dialect, "func", [FunctionOpInterface, IsolatedFromAbove]> {

    let summary = "user defined function operation";


    let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    // CArg means c++ argument, it is a vector of 'NamedArguments' and it's default value is empty
    let builders = [OpBuilder<(ins
    "std::string":$name, "FunctionType":$type,
    CArg<"std::vector<NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    std::vector<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    std::vector<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

    let skipDefaultBuilders = 1;
}


// CallOp

def CallOp : Op<Choco_Dialect, "call", []> {
      let summary = "generic call operation";
      // The callee name is attached as a symbol reference via an attribute.

      let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<I32>:$inputs);

      let results = (outs I32);

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$callee, "std::vector<Value>":$arguments)>
    ];
}

 

def ReturnOp : Op<Choco_Dialect, "return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let summary = "return operation";

    // The return operation takes an optional input operand to return. This
    // value must match the return type of the enclosing function.
    let arguments = (ins Variadic<I32>:$input);

    // The return operation only emits the input in the format if it is present.
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    // Allow building ReturnOp with no return operand
    let builders = [
        OpBuilder<(ins)>
    ];

    // Provide extra utility definitions on the c++ operation class definition.
    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

}

#endif // CHOCO_OPS