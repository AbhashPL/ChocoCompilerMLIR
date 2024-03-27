// passes over the AST. They are performed before going into MLIR codegen

// Uniquify pass -> make every variable name unique, so we don't need to worry about scoping

// Gather Pass -> gather juvenile statements into a main function

#include "AST.h"
#include <random>

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
            ident.value = "choco_main";

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

            // TODO: only add return at the end when there is no return stmt beforehand

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




    uint64_t random_num = 0;
    std::string genSym(std::string sym) {
        random_num+=1;
        std::string res = sym+std::to_string(random_num);
        return res;
    }

    // void uniquifyPass(ExpressionStatement& expStmt)
    // {
    //     uniquifyPass(expStmt.getExp());
    // }


    // void uniquifyPass(std::shared_ptr<Expression> exp) {

    //         // This function is responsible for dispatching the correct mlirGen function based on what kind of expression it is
    //         switch (exp->getKind())
    //         {
    //         case IntegerLiteralExp :
    //             {
    //                 std::shared_ptr<IntegerLiteral> inLit = std::dynamic_pointer_cast<IntegerLiteral>(exp);
                    
    //             }

    //         case IdentifierExp :
    //             {
    //                 std::shared_ptr<Identifier> ident = std::dynamic_pointer_cast<Identifier>(exp);
    //                 std::string name = ident->getName();

    //             }

    //         case InfixExpressionExp :
    //             {
    //                 std::shared_ptr<InfixExpression> inExp = std::dynamic_pointer_cast<InfixExpression>(exp);
                    
    //             }

    //         case CallExp :
    //             {   
    //                 std::shared_ptr<CallExpression> cExp = std::dynamic_pointer_cast<CallExpression>(exp);
                    
    //             }

    //         }

    // }


    // void uniquifyPass(Expression& exp) {
        
    // }

    // void uniquifyPass(BlockStatement& bs) {
    //     std::vector<std::shared_ptr<Statement>> sts = bs.getStatements();

    //     for (auto st : sts)
    //     {

    //         switch (st->getKind())
    //         {
    //             case ExprStmt:
    //             {
    //                 std::shared_ptr<ExpressionStatement> es = std::dynamic_pointer_cast<ExpressionStatement>(st);

    //                 if (!es) {
    //                     std::cout<<"couldn't typecast"<<std::endl;
    //                     exit(0);
    //                 }

    //                 uniquifyPass(*es);
    //                 break;
    //             }

    //             case VarDeclStmt:
    //             {
    //                 std::shared_ptr<VarDeclStatement> vs = std::dynamic_pointer_cast<VarDeclStatement>(st);

    //                 if (!vs) {
    //                     std::cout<<"couldn't typecast"<<std::endl;
    //                     exit(0);
    //                 }

    //                 uniquifyPass(*vs);
    //                 break;
    //             }

    //             case AssignStmt:
    //             {
    //                 std::shared_ptr<AssignStatement> as = std::dynamic_pointer_cast<AssignStatement>(st);

    //                 if (!as) {
    //                     std::cout<<"couldn't typecast"<<std::endl;
    //                     exit(0);
    //                 }

    //                 uniquifyPass(*as);
    //                 break;
    //             }

    //             case RetStmt:
    //             {
    //                 std::shared_ptr<ReturnStatement> rs = std::dynamic_pointer_cast<ReturnStatement>(st);

    //                 if (!rs) {
    //                     std::cout<<"couldn't typecast"<<std::endl;
    //                     exit(0);
    //                 }

    //                 uniquifyPass(*rs);
    //                 break;
    //             }

    //             case IfStmt:
    //             {
    //                 std::shared_ptr<IfStatement> is = std::dynamic_pointer_cast<IfStatement>(st);
                    
    //                 if (!is) {
    //                     std::cout<<"couldn't typecast"<<std::endl;
    //                     exit(0);
    //                 }

    //                 uniquifyPass(*is);
    //                 break;
    //             }
    //         }
    //     }
    // }

    // void uniquifyPass(Function& func) {

    // }

    // void uniquifyPass(IfStatement& is) {
        
    // }

    // void uniquifyPass(ReturnStatement& rs) {
        
    // }

    // void uniquifyPass(VarDeclStatement& vds) {
        
    // }

    // void uniquifyPass(AssignStatement& as) {
        
    // }



    // Program& uniquifyPass(Program& prg) {
        
    //     for(auto &st : prg.getStatements()) {
    //         // this will get the function statements
            
    //         // rename each function except 'choco_main'


    //         // then call the uniquifyPass on Blk statement

    //     }

    // }



}