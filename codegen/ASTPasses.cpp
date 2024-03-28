// passes over the AST. They are performed before going into MLIR codegen

// Uniquify pass -> make every variable name unique, so we don't need to worry about scoping

// Gather Pass -> gather juvenile statements into a main function

#include "AST.h"
#include "lexer.cpp"
#include <random>
#include <unordered_map>


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

            // creating identifier token "choco_main"
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
            
            auto lastStmt = stmts.back();
            
            if(lastStmt->getKind() != RetStmt) {
                std::shared_ptr<Expression> return_value(new IntegerLiteral(int_token, 0));
                std::shared_ptr<ReturnStatement> ret_stmt(new ReturnStatement(ret_token, return_value));
                stmts.push_back(ret_stmt);
            }

            std::shared_ptr<BlockStatement> body(new BlockStatement(block_cur, stmts));

            std::shared_ptr<Function> func(new Function(func_cur, ident, param_list, body));

            // adds FuncStmt to the end
            prg.addFuncStatement(func);
        }
        
        return prg;
    }

    void uniquifyPass(BlockStatement& bs);

    std::unordered_map<std::string, bool> useDef;
    std::unordered_map<std::string, std::string> renameTbl;
    std::unordered_map<std::string, std::string> funcRenameTbl;

    uint64_t random_num = 0;
    
    std::string genSym(std::string sym) {
        random_num+=1;
        std::string res = sym+std::to_string(random_num);
        return res;
    }
    
    // check whether this var that is being defined is in scope or not
    bool checkUseDef(std::string& name) {

    }

    void uniquifyPass(std::shared_ptr<Expression> exp) {

        // This function is responsible for dispatching the correct mlirGen function based on what kind of expression it is
        switch (exp->getKind())
        {
            case IntegerLiteralExp :
                {
                    std::shared_ptr<IntegerLiteral> inLit = std::dynamic_pointer_cast<IntegerLiteral>(exp);
                    return;
                }

            case IdentifierExp :
                {
                    std::shared_ptr<Identifier> ident = std::dynamic_pointer_cast<Identifier>(exp);
                    std::string name = ident->getName();

                    if (renameTbl.count(name) == 0) {
                        std::cout<<"Error: Use before define"<<std::endl;
                        exit(0);
                    } else {
                        Token tok;
                        tok.ttype = tok_identifier;
                        tok.value = renameTbl[name];
                        ident->setToken(tok);
                        ident->setName(renameTbl[name]);
                    }
                }

            case InfixExpressionExp :
                {
                    std::shared_ptr<InfixExpression> inExp = std::dynamic_pointer_cast<InfixExpression>(exp);
                    
                    auto left = inExp->getLeft();
                    auto right = inExp->getRight();
                    uniquifyPass(left);
                    uniquifyPass(right);
                    
                }

            case CallExp :
                {   
                    std::shared_ptr<CallExpression> cExp = std::dynamic_pointer_cast<CallExpression>(exp);
                    std::string name = cExp->getCallee();
                    auto args = cExp->getArgs();

                    if(funcRenameTbl.count(name) == 0 && name != "print") {
                        std::cout<<"Error: Calling function "<<name<<" before creating it"<<std::endl;
                        exit(0);
                    } else {

                        Token curToken;
                        curToken.ttype = tok_identifier;
                        curToken.value = funcRenameTbl[name];

                        std::shared_ptr<Identifier> newIdent(new Identifier(curToken, curToken.value));
                        cExp->setIdent(newIdent);

                        for(auto a : args) {
                            uniquifyPass(a);
                        }
                    }
                }
            }
    }

        void uniquifyPass(IfStatement& is) {
        auto cond = is.getCond();
        auto thn = is.getThen();
        auto els = is.getEls();

        uniquifyPass(cond);
        uniquifyPass(*thn);
        uniquifyPass(*els);
    }

    void uniquifyPass(ReturnStatement& rs) {
        auto exp = rs.getReturnValue();

        if(exp) {
            uniquifyPass(exp);
        }
    }

    void uniquifyPass(VarDeclStatement& vds) {
        std::string name = vds.getIdent()->getName();
        renameTbl[name] = genSym(name);

        Token curToken;
        curToken.ttype = tok_identifier;
        curToken.value = renameTbl[name];

        std::shared_ptr<Identifier> newIdent(new Identifier(curToken, curToken.value));
        vds.setIdent(newIdent); // moves ownership to the new identifier

        auto exp = vds.getValueExpr();
        uniquifyPass(exp);
    }

    void uniquifyPass(AssignStatement& as) {
        // TODO:
    }



    void uniquifyPass(ExpressionStatement& expStmt)
    {
        uniquifyPass(expStmt.getExp());
    }

    void uniquifyPass(BlockStatement& bs) {
        std::vector<std::shared_ptr<Statement>> sts = bs.getStatements();

        for (auto st : sts)
        {

            switch (st->getKind())
            {
                case ExprStmt:
                {
                    std::shared_ptr<ExpressionStatement> es = std::dynamic_pointer_cast<ExpressionStatement>(st);

                    if (!es) {
                        std::cout<<"couldn't typecast"<<std::endl;
                        exit(0);
                    }

                    uniquifyPass(*es);
                    break;
                }

                case VarDeclStmt:
                {
                    std::shared_ptr<VarDeclStatement> vs = std::dynamic_pointer_cast<VarDeclStatement>(st);

                    if (!vs) {
                        std::cout<<"couldn't typecast"<<std::endl;
                        exit(0);
                    }

                    uniquifyPass(*vs);
                    break;
                }

                case AssignStmt:
                {
                    std::shared_ptr<AssignStatement> as = std::dynamic_pointer_cast<AssignStatement>(st);

                    if (!as) {
                        std::cout<<"couldn't typecast"<<std::endl;
                        exit(0);
                    }

                    uniquifyPass(*as);
                    break;
                }

                case RetStmt:
                {
                    std::shared_ptr<ReturnStatement> rs = std::dynamic_pointer_cast<ReturnStatement>(st);

                    if (!rs) {
                        std::cout<<"couldn't typecast"<<std::endl;
                        exit(0);
                    }

                    uniquifyPass(*rs);
                    break;
                }

                case IfStmt:
                {
                    std::shared_ptr<IfStatement> is = std::dynamic_pointer_cast<IfStatement>(st);
                    
                    if (!is) {
                        std::cout<<"couldn't typecast"<<std::endl;
                        exit(0);
                    }

                    uniquifyPass(*is);
                    break;
                }
            }
        }
    }

    void uniquifyPass(Function& func) {
        
        // rename the function_name
        auto fname = func.getName();
        Token curToken;
        curToken.ttype = tok_identifier;
        curToken.value = funcRenameTbl[fname];
        func.setNameToken(curToken);


        // renaming of the params
        auto &params = func.getParams();
        std::vector<std::shared_ptr<Identifier>> newvec;

        for(auto p : params) {
           std::string name = p->getName();
           renameTbl[name] = genSym(name);

           Token curToken2;
           curToken2.ttype = tok_identifier;
           curToken2.value = renameTbl[name];
           std::shared_ptr<Identifier> newIdent(new Identifier(curToken2, curToken2.value));
           
           newvec.push_back(newIdent);
        }

        params = newvec;

        auto body = func.getBody();
        uniquifyPass(*body);
    }


    Program& uniquifyPass(Program& prg) {
        
        for(auto &st : prg.getStatements()) {
            // this will get the function statements
            std::shared_ptr<Function> fs = std::dynamic_pointer_cast<Function>(st);

            if (!fs) {
                std::cout << "couldn't typecast" << std::endl;
                exit(0);
            }

            auto fname = fs->getName();
            if(fname != "choco_main") {
                
                funcRenameTbl[fname] = genSym(fname);
                uniquifyPass(*fs);
            }
        }

        return prg;
    }

}