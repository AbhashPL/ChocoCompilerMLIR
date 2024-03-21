// passes over the AST. They are performed before going into MLIR codegen

// Uniquify pass -> make every variable name unique, so we don't need to worry about scoping

// Gather Pass -> gather juvenile statements into a main function

#include "AST.h"

namespace choco {

    Program& gatherPass(Program& prg) {
        std::vector<std::shared_ptr<Statement>> stmts;
        
        auto &original_stmts = prg.getStatements();
        
        std::vector<std::shared_ptr<Statement>>::iterator it = original_stmts.begin();
        while (it!= original_stmts.end())
        {   
            auto st = *it;
            if(st->getKind() != FuncStmt) {
                stmts.push_back(st);

                // erase without iterator invalidation
                it = original_stmts.erase(it);
            } else {
                ++it;
            }
        }

        if(!stmts.empty()) {

            // creating identifier token "main"
            Token ident;
            ident.ttype = tok_identifier;
            ident.value = "main";

            // creating "def" token
            Token func_cur;
            func_cur.ttype = tok_def;
            func_cur.value = "def";

            std::vector<std::shared_ptr<Identifier>> param_list{};

            Token block_cur;
            block_cur.ttype = tok_open_brace;
            block_cur.value = "{";
            
            Token ret_token;
            ret_token.ttype = tok_return;
            ret_token.value = "return";

            Token int_token;
            int_token.ttype = tok_numeric;
            int_token.value = "0";


            std::shared_ptr<Expression> return_value(new IntegerLiteral(int_token, 0));
            std::shared_ptr<ReturnStatement> ret_stmt(new ReturnStatement(ret_token, return_value));
            stmts.push_back(ret_stmt);

            //TODO: add a return 0; statement at the end of stmts
            std::shared_ptr<BlockStatement> body(new BlockStatement(block_cur, stmts));

            std::shared_ptr<Function> func(new Function(func_cur, ident, param_list, body));

            // adds FuncStmt to the end
            prg.addFuncStatement(func);
        }
        
        return prg;
    }

}