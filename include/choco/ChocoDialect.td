#ifndef CHOCO_DIALECT
#define CHOCO_DIALECT

include "mlir/IR/DialectBase.td"

def Choco_Dialect : Dialect {
    let name = "choco";
    let summary = "A dialect for choco language";
    let description = [{
        Choco dialect defines types and operations for the choco language
    }];

    let cppNamespace = "::mlir::choco";
}


#endif // DIALECT_H