#ifndef AST_H
#define AST_H

#include "lexer.cpp"

#include <vector>
#include <utility>
#include <unordered_map>

#include <string>
#include <memory>


namespace choco {

enum Kind {
    IntegerLiteralExp,
    IdentifierExp,
    PrefixExpressionExp,
    InfixExpressionExp,
    BooleanExp,
    CallExp,
    
    ExprStmt,
    FuncStmt,
    RetStmt,
    VarDeclStmt,
    IfStmt,
    AssignStmt,
    BlkStmt
};

class Node {
    public:
        virtual std::string TokenValue() const = 0;
        virtual std::string String() const = 0;
        virtual ~Node() = default;
};

class Expression : public Node {
    public:
        virtual void expressionNode() = 0;
        virtual Kind getKind() = 0;
};

class Statement : public Node {
    public :
        virtual void statementNode() = 0;
        virtual Kind getKind() = 0;
};


class IntegerLiteral : public Expression {
    public:
        IntegerLiteral(Token token, int value)
            : token(token), value(value) {}

        std::string TokenValue() const override {
            return token.value;
        }

        std::string String() const override {
            
            return token.value;
        }

        uint32_t getVal() {
            return value;
        }

        void expressionNode() override {}

        Kind getKind() {
            return IntegerLiteralExp;
        }

    private:
        Token token;
        uint32_t value;
};

class Identifier : public Expression {
    public:
        Identifier(Token token, std::string name)
            : token(token), name(name) {}
        
        std::string TokenValue() const override {
            return token.value;
        }

        std::string getName() {
            return name;
        }

        std::string String() const override {
            return name;
        }

        void expressionNode() override {}

        Kind getKind() {
            return IdentifierExp;
        }

    private:
        Token token;
        std::string name;
};


class PrefixExpression : public Expression {
    public:
        PrefixExpression(Token token, std::string op, std::shared_ptr<Expression> right)
            : token(token), op(op), right(right) {}

        std::string TokenValue() const override {
            return token.value;
        }

        std::string getOp() {
            return op;
        }
    
        std::shared_ptr<Expression> getRight() {
            return right;
        }

        std::string String() const override {
            std::string str;

            str+="(";
            str+=op;
            str+=right->String();
            str+=")";

            return str;
        }

        void expressionNode() override {}
        
        Kind getKind() {
            return PrefixExpressionExp;
        }

    private:
        Token token;
        std::string op;
        std::shared_ptr<Expression> right;
};

class InfixExpression : public Expression {
    public:
        InfixExpression(Token token, std::shared_ptr<Expression> left, std::string op, std::shared_ptr<Expression> right)
            : token(token), left(left), op(op), right(right) {}

        std::string TokenValue() const override {
            return token.value;
        }

        std::shared_ptr<Expression> getLeft() {
            return left;
        }

        std::string getOp() {
            return op;
        }

        std::shared_ptr<Expression> getRight() {
            return right;
        }

        Kind getKind()
        {
            return InfixExpressionExp;
        }

        std::string String() const override {
            std::string str;

            str+="(";
            str+=left->String();
            str+=" "+op+" ";
            str+=right->String();
            str+=")";

            return str;
        }
        
        void expressionNode() override {}


    private:
        Token token;
        std::shared_ptr<Expression> left;
        std::string op;
        std::shared_ptr<Expression> right;
};

class Boolean : public Expression {
    public:
        Boolean(Token token, bool value)
            : token(token), value(value) {}

        std::string TokenValue() const override {
            return token.value;
        }

        bool getValue() {
            return value;
        }

        std::string String() const override {
            return TokenValue();
        }

        Kind getKind()
        {
            return BooleanExp;
        }


        void expressionNode() override {}

    private:
        Token token;
        bool value;
};


class CallExpression : public Expression {
    public:
        CallExpression(Token token, std::shared_ptr<Identifier> ident ,std::vector<std::shared_ptr<Expression>> args)
            : token(token), ident(ident), args(args) {}

        std::string TokenValue() const override {
            return token.value;
        }

        std::string getCallee() const {
            return ident->getName();
        }

        std::vector<std::shared_ptr<Expression>> getArgs() {
            return args;
        }

        Kind getKind()
        {
            return CallExp;
        }


        std::string String() const override {
            std::string str;
            
            str+=getCallee()+"(";
            // print the expression list also here with commas
            for(auto& exp : args) {
                str+=exp->String();
                str+=", ";
            }
            str+=")";

            return str;
        }

        void expressionNode() override {}

    private:
        Token token;
        std::shared_ptr<Identifier> ident;
        std::vector<std::shared_ptr<Expression>> args;
};

class BlockStatement : public Statement {
    public:
        BlockStatement(Token token, std::vector<std::shared_ptr<Statement>> statements)
            : token(token), statements(statements) {} // token stored "{" token, marking the beginning of a block

        std::string TokenValue() const override {
            return token.value;
        }

        std::vector<std::shared_ptr<Statement>>& getStatements() {
            return statements;
        }

        std::string String() const override {
            std::string str;

            for(auto &st : statements) {
                str+=st->String();
            }

            return str;
        }

        Kind getKind() override {
            return BlkStmt;
        }

        void statementNode() override {}

    private:
        Token token;
        std::vector<std::shared_ptr<Statement>> statements;
};


class ExpressionStatement : public Statement
{
public:
    std::shared_ptr<Expression> exp;
    ExpressionStatement(Token token, std::shared_ptr<Expression> exp)
        : token(token), exp(exp) {}

    // first token of the expression
    std::string TokenValue() const override
    {
        return token.value;
    }

    std::shared_ptr<Expression> getExp() {
        return exp;
    }

    std::string String() const override
    {
        std::string str;

        if (exp != nullptr)
        {
            str += exp->String();
        }

        return str;
    }

    Kind getKind()
    {
        return ExprStmt;
    }

    void statementNode() override {}

private:
    Token token;
    
};

class IfStatement : public Statement
{
public:
    IfStatement(Token token, std::shared_ptr<Expression> cond, std::shared_ptr<BlockStatement> then, std::shared_ptr<BlockStatement> els)
        : token(token), cond(cond), then(then), els(els) {}

    std::string TokenValue() const override
    {
        return token.value;
    }

    std::shared_ptr<Expression> getCond()
    {
        return cond;
    }

    std::shared_ptr<BlockStatement> getThen()
    {
        return then;
    }

    std::shared_ptr<BlockStatement> getEls()
    {
        return els;
    }

    bool hasReturnInThen() {
        auto thenLastStmt = then->getStatements().back();

        if(thenLastStmt->getKind() == RetStmt) {
            return true;
        }

        return false;
    }

    bool hasReturnInBothBranches()
    {
        if (els != nullptr)
        {
            auto thenLastStmt = then->getStatements().back();
            auto elsLastStmt = els->getStatements().back();

            if ((thenLastStmt->getKind() == RetStmt) && (elsLastStmt->getKind() == RetStmt))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        return false;
    }

    bool hasReturnInOneBranch()
    {
        if (els != nullptr)
        {
            auto thenLastStmt = then->getStatements().back();
            auto elsLastStmt = els->getStatements().back();

            if ((thenLastStmt->getKind() == RetStmt) && (elsLastStmt->getKind() != RetStmt))
            {
                return true;
            }
            else if ((thenLastStmt->getKind() != RetStmt) && (elsLastStmt->getKind() == RetStmt))
            {
                return true;
            }
        }
        return false;
    }

    std::string String() const override
    {
        std::string str;

        str += "if(";
        str += cond->String() + ")\n{\n";
        str += then->String() + "\n}\n";
        if (els)
        {
            str += "else {\n";
            str += els->String() + "\n}\n";
        }

        return str;
    }

    Kind getKind()
    {
        return IfStmt;
    }

    void statementNode() override {}

private:
    Token token;
    std::shared_ptr<Expression> cond;
    std::shared_ptr<BlockStatement> then;
    std::shared_ptr<BlockStatement> els;
};

class ReturnStatement : public Statement {
    public:
        ReturnStatement(Token token, std::shared_ptr<Expression> returnValue)
            : token(token), returnValue(returnValue) {}

        std::string TokenValue() const override {
            return token.value;
        }

        std::shared_ptr<Expression> getReturnValue() {
            return returnValue;
        }

        std::string String() const override {
            std::string str;
            str += TokenValue() + " ";

            if(returnValue != nullptr) {
                str += returnValue->String();
            }

            str += ";\n";

            return str;
        }

        Kind getKind()
        {
            return RetStmt;
        }

        void statementNode() override {}

    private:
        Token token;
        std::shared_ptr<Expression> returnValue;
};



class Function : public Statement {
    public:
        Function(Token token, Token name, std::vector<std::shared_ptr<Identifier>> params, std::shared_ptr<BlockStatement> body)
            : token(token), name(name), params(params), body(body) {}
        
        std::string TokenValue() const override {
            return token.value;
        }

        std::string getName() {
            return name.value;
        }
        
        std::vector<std::shared_ptr<Identifier>> getParams() {
            return params;
        }

        std::shared_ptr<BlockStatement> getBody() {
            return body;
        }

        void gatherReturns(std::shared_ptr<BlockStatement> body)
        {
            if(body) {
                for (auto &st : body->getStatements())
                {
                    if (st->getKind() == RetStmt)
                    {
                        std::shared_ptr<ReturnStatement> rs = std::dynamic_pointer_cast<ReturnStatement>(st);
                        auto rv = rs->getReturnValue();
                        if(rv) {
                            retStmtsWithOperands[rs] = 1;
                        } else {
                            retStmtsWithOperands[rs] = 0; 
                        }
                    }
                    else if (st->getKind() == IfStmt)
                    {
                        std::shared_ptr<IfStatement> is = std::dynamic_pointer_cast<IfStatement>(st);
                        gatherReturns(is->getThen());
                        auto els = is->getEls();
                        if(els) {
                            gatherReturns(els);
                        } else {
                            continue;
                        }
                    }
                }
                return;
            }
            else {
                return;
            }
        }

        std::unordered_map<std::shared_ptr<ReturnStatement>, int>& getReturnsMap() {
            return retStmtsWithOperands;
        }


        bool hasReturn() {
            // Does this function end with a return statement
            // used in generating the return type of the function in MLIRGen.cpp

            for(auto st : body->getStatements()) {
                if(st->getKind() == RetStmt) {
                    return true;
                } else if (st->getKind() == IfStmt) {
                    std::shared_ptr<IfStatement> is = std::dynamic_pointer_cast<IfStatement>(st);
                    if(is->hasReturnInOneBranch())
                        return true;
                }
            }
            return false;
        }

        std::string String() const override {
            std::string str;

            str+=TokenValue()+" ";
            str+=name.value+"(";
            
            for(auto& ident : params) {
                str+=ident->String();
                str+=", ";
            }

            str+=")\n{\n";
            str+=body->String()+"\n}\n";

            return str;
        }

        Kind getKind()
        {
            return FuncStmt;
        }

        void statementNode() override {}

    private:
        Token token;
        Token name;
        std::vector<std::shared_ptr<Identifier>> params;
        std::shared_ptr<BlockStatement> body;
        std::unordered_map<std::shared_ptr<ReturnStatement>, int> retStmtsWithOperands;
};


class VarDeclStatement : public Statement {
    public:
        VarDeclStatement(Token token, std::shared_ptr<Identifier> name, std::shared_ptr<Expression> value)
            : token(token) , name(name) , value(value) {}

        std::string TokenValue() const override {
            return token.value;
        }

        std::shared_ptr<Identifier> getIdent() {
            return name;
        }

        std::shared_ptr<Expression> getValueExpr() {
            return value;
        }

        std::string String() const override {
            std::string str;
        
            str += TokenValue() + " ";
            str += name->String();
            str += " = ";

            if(value != nullptr) {
                str += value->String();
            }

            str += ";\n";

            return str;
        }

        Kind getKind()
        {
            return VarDeclStmt;
        }

        void statementNode() override {}

    private:
        Token token;
        std::shared_ptr<Identifier> name;
        std::shared_ptr<Expression> value;
};

class AssignStatement : public Statement {
    public :
        AssignStatement(Token token, std::shared_ptr<Identifier> ident, std::shared_ptr<Expression> right)
            : token(token), ident(ident), right(right) {} // token stores the name of the identifier that is being assigned to

        std::string TokenValue() const override {
            return token.value;
        }

        std::shared_ptr<Identifier> getIdent() {
            return ident;
        }

        std::shared_ptr<Expression> getRight() {
            return right;
        }

        std::string String() const override {

            std::string str;

            str+=ident->String();
            str+="=";
            str+=right->String();
            str+=";\n";

            return str; 
        }

        Kind getKind()
        {
            return AssignStmt;
        }

        void statementNode() override {}

    private:
        Token token;
        std::shared_ptr<Identifier> ident;
        std::shared_ptr<Expression> right;
};



class Program : public Node {
    public:
        Program(std::vector<std::shared_ptr<Statement>> statements) :
            statements(statements) {}

        std::string TokenValue() const override {
            if (!statements.empty()) {
                return statements[0]->TokenValue();
            } else {
                return "";
            }
        }

        std::vector<std::shared_ptr<Statement>>& getStatements() {
            return statements;
        }

        void addFuncStatement(std::shared_ptr<Function> func) {
            statements.push_back(func);
        }

        std::string String() const override {
            std::string str;

            for(auto &st : statements) {
                str+=st->String();
            }
            return str;
        }
        
    private:
        std::vector<std::shared_ptr<Statement>> statements;
        
};



}

#endif //AST_H