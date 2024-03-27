#include "lexer.cpp"
#include "AST.h"


#include <vector>
#include <memory>
#include <functional>



namespace choco {

    class Parser {
        public:
            enum Constants {
                LOWEST = 1,
                EQUALS,         // ==
                LESSGREATER,    // > or <
                SUM,            // +
                PRODUCT,        // *
                PREFIX,         // -X or !X
                CALL            // fn(X)
            };

            
            Parser(Lexer &l) : lexer(l) {
                // set the cur_token and peek_token when initializing the parser
                curToken = choco::nextToken(lexer);
                peekToken = choco::nextToken(lexer);

            }

            void nextToken() {
                curToken = peekToken;
                peekToken = choco::nextToken(lexer);
            }

            bool curTokenIs(TokenType t) {
                return curToken.ttype == t;
            }

            bool peekTokenIs(TokenType t) {
                return peekToken.ttype == t;
            }

            // advances the tokens while making assertions about the token-type
            bool expectPeek(TokenType t) {
                if (peekTokenIs(t)) {
                    nextToken();
                    return true;
                } else {
                    peekError(t);
                    return false;
                }
            }

            int peekPrecedence() {
                auto it = precedences.find(peekToken.ttype);
                
                if (it != precedences.end())
                {
                    return it->second;
                }
                
                return LOWEST;
            }

            int curPrecedence() {
                auto it = precedences.find(curToken.ttype);

                if (it != precedences.end())
                {
                    return it->second;
                }

                return LOWEST; // for things like ')' we'd return LOWEST
            }


            std::shared_ptr<VarDeclStatement> parseVarDecl() {
                
                Token var_token = curToken;

                if(!expectPeek(tok_identifier))
                    return nullptr;

                std::shared_ptr<Identifier> ident(new Identifier(curToken, curToken.value));

                if(!expectPeek(tok_assign))
                    return nullptr;

                
                nextToken();

                std::shared_ptr<Expression> exp = parseExpression(LOWEST);

                if (peekTokenIs(tok_semicolon))
                {
                    nextToken();
                }
                else
                {
                    peekError(tok_semicolon);
                    return nullptr;
                }

                std::shared_ptr<VarDeclStatement> stmt(new VarDeclStatement(var_token, ident, exp));

                return stmt;
            }
            
            std::shared_ptr<ReturnStatement> parseReturn() {
                Token ret_token = curToken;
                nextToken();

                // i.e. no return value
                if(curTokenIs(tok_semicolon)) {
                    //nextToken();
                    std::shared_ptr<ReturnStatement> stmt(new ReturnStatement(ret_token, nullptr));
                    return stmt;
                }

                std::shared_ptr<Expression> return_value = parseExpression(LOWEST);

                if (peekTokenIs(tok_semicolon))
                {
                    nextToken();
                }
                else
                {
                    peekError(tok_semicolon);
                    return nullptr;
                }

                std::shared_ptr<ReturnStatement> stmt(new ReturnStatement(ret_token, return_value));

                return stmt;
            }

            std::shared_ptr<Identifier> parseIdentifier()
            {
                std::shared_ptr<Identifier> exp(new Identifier(curToken, curToken.value));
                return exp;
            }

            std::shared_ptr<IntegerLiteral> parseIntegerLiteral()
            {
                uint32_t value = stoi(curToken.value);
                std::shared_ptr<IntegerLiteral> lit(new IntegerLiteral(curToken, value));
                return lit;
            }

            std::shared_ptr<PrefixExpression> parsePrefixExpression() {
                Token cur = curToken;
                std::string op = curToken.value;

                nextToken();

                std::shared_ptr<Expression> right = parseExpression(PREFIX);

                std::shared_ptr<PrefixExpression> pre(new PrefixExpression(curToken, op, right));


                return pre;
            }

            bool checkPrefixParseFns(TokenType tt) {
                switch (tt)
                {
                case tok_identifier:
                    return true;

                case tok_numeric:
                    return true;

                case tok_not:
                    return true;

                case tok_minus:
                    return true;

                case tok_open_paren:
                    return true;
                }

                return false;
            }

            std::shared_ptr<Boolean> parseBoolean() {
                std::shared_ptr<Boolean> exp(new Boolean(curToken, curTokenIs(tok_true)));
                return exp;
            }


            std::shared_ptr<Expression> prefixParseFns(TokenType tt) {
                switch (tt)
                {
                case tok_identifier:
                    return parseIdentifier();

                case tok_numeric:
                    return parseIntegerLiteral();

                case tok_not:
                    return parsePrefixExpression();

                case tok_minus:
                    return parsePrefixExpression();

                case tok_open_paren:
                    return parseGroupedExpression();

                case tok_true:
                    return parseBoolean();

                case tok_false:
                    return parseBoolean();
                }
                
                return nullptr;
            }

            std::shared_ptr<Expression> parseGroupedExpression() {
                nextToken();

                std::shared_ptr<Expression> exp = parseExpression(LOWEST);

                if(!expectPeek(tok_close_paren))
                    return nullptr;

                return exp;
            }

            std::shared_ptr<InfixExpression> parseInfixExpression(std::shared_ptr<Expression> left) {
                Token cur = curToken;
                int prec = curPrecedence();
                nextToken();
                std::shared_ptr<Expression> right = parseExpression(prec);

                std::shared_ptr<InfixExpression> exp(new InfixExpression(cur, left, cur.value, right));

                return exp;
            }

            bool checkInfixParseFns(TokenType tt) {
                switch (tt)
                {
                case tok_plus:
                    return true;
                case tok_minus:
                    return true;
                case tok_mult:
                    return true;
                case tok_lt:
                    return true;
                case tok_gt:
                    return true;
                case tok_eq:
                    return true;
                case tok_not_eq:
                    return true;
                case tok_open_paren:
                    return true;
                }

                return false; // only if a token is passed for which there is no parseInfixFunction
            }

            std::shared_ptr<Expression> infixParseFns(TokenType tt, std::shared_ptr<Expression> left)
            {
                switch (tt)
                {
                    case tok_plus: 
                        return parseInfixExpression(left);
                    case tok_minus:
                        return parseInfixExpression(left);
                    case tok_mult:
                        return parseInfixExpression(left);
                    case tok_lt:
                        return parseInfixExpression(left);
                    case tok_gt:
                        return parseInfixExpression(left);
                    case tok_eq:
                        return parseInfixExpression(left);
                    case tok_not_eq:
                        return parseInfixExpression(left);
                    case tok_open_paren:
                        return parseCallExpression(left);
                }

                return nullptr; // only if a token is passed for which there is no parseInfixFunction
            }

            std::shared_ptr<CallExpression> parseCallExpression(std::shared_ptr<Expression> funcIdentExpr) {
                Token cur = curToken;
                
                std::vector<std::shared_ptr<Expression>> args = parseCallArguments();

                std::shared_ptr<Identifier> ident = std::dynamic_pointer_cast<Identifier>(funcIdentExpr);

                std::shared_ptr<CallExpression> call(new CallExpression(cur, ident, args));

                return call;
            }

            std::vector<std::shared_ptr<Expression>> parseCallArguments() {
                
                std::vector<std::shared_ptr<Expression>> args;

                // If the function doesn't accept any arguments
                if(peekTokenIs(tok_close_paren)) {
                    nextToken();
                    return args;
                }

                // If the function accepts multiple arguments
                nextToken();
                // for the first argument
                std::shared_ptr<Expression> exp1 = parseExpression(LOWEST);
                args.push_back(exp1);

                while(peekTokenIs(tok_comma)) {
                    nextToken();
                    nextToken();
                    std::shared_ptr<Expression> exp2 = parseExpression(LOWEST);
                    args.push_back(exp2);
                }

                if(!expectPeek(tok_close_paren)) {
                    // just fill args with nullptr
                    std::cout<<"close the paren in the arguments list"<<std::endl;
                    args.push_back(nullptr);
                    return args;
                }

                return args;
            }

            std::shared_ptr<Expression> parseExpression(int precedence) {
                
                // check if there is an associated prefixParseFunction for the current token
                if(checkPrefixParseFns(curToken.ttype) == false)
                    return nullptr;

                std::shared_ptr<Expression> leftExp = prefixParseFns(curToken.ttype);
                
                while (!peekTokenIs(tok_semicolon) && precedence < peekPrecedence()) {

                    if(checkInfixParseFns(peekToken.ttype) == false)
                        return leftExp;

                    nextToken();

                    leftExp = infixParseFns(curToken.ttype, leftExp);
                }

                return leftExp;
            }

            std::shared_ptr<AssignStatement> parseAssignStatement() {
                Token cur = curToken; // the identifier token

                std::shared_ptr<Identifier> ident(new Identifier(cur, cur.value));

                if (!expectPeek(tok_assign))
                    return nullptr;

                nextToken();

                std::shared_ptr<Expression> exp = parseExpression(LOWEST);

                if (peekTokenIs(tok_semicolon))
                {
                    nextToken();
                }
                else
                {
                    peekError(tok_semicolon);
                    return nullptr;
                }

                std::shared_ptr<AssignStatement> stmt(new AssignStatement(cur, ident, exp));
                return stmt;
            }

            std::shared_ptr<ExpressionStatement> parseExpressionStatements() {
                Token cur = curToken;

                std::shared_ptr<Expression> exp = parseExpression(LOWEST);

                if(peekTokenIs(tok_semicolon)) {
                    nextToken();
                } else {
                    peekError(tok_semicolon);
                    return nullptr;
                }

                std::shared_ptr<ExpressionStatement> stmt(new ExpressionStatement(cur, exp));
                
                return stmt;
            }

            std::shared_ptr<BlockStatement> parseBlockStatement() {
                Token cur = curToken;

                std::vector<std::shared_ptr<Statement>> stmts;

                nextToken();

                while(!curTokenIs(tok_close_brace) && !curTokenIs(tok_eof)) {
                    std::shared_ptr<Statement> st = parseStatement();

                    if(st != nullptr)
                        stmts.push_back(st);

                    nextToken();
                }

                std::shared_ptr<BlockStatement> bs(new BlockStatement(cur, stmts));

                return bs;
            }

            std::shared_ptr<IfStatement> parseIfStatement() {
                Token cur = curToken;

                if(!expectPeek(tok_open_paren))
                    return nullptr;

                nextToken();

                std::shared_ptr<Expression> cond = parseExpression(LOWEST);

                if(!expectPeek(tok_close_paren))
                    return nullptr;
                
                if(!expectPeek(tok_open_brace))
                    return nullptr;

                std::shared_ptr<BlockStatement> then = parseBlockStatement();

                std::shared_ptr<BlockStatement> els;

                if(peekTokenIs(tok_else)) {
                    nextToken();

                    if(!expectPeek(tok_open_brace))
                        return nullptr;

                    els = parseBlockStatement();
                } else {
                    els = nullptr;
                }

                std::shared_ptr<IfStatement> st(new IfStatement(cur, cond, then, els));

                return st;
            }

            std::vector<std::shared_ptr<Identifier>> parseParamList() {
                
                std::vector<std::shared_ptr<Identifier>> identifiers;

                // in case of empty param list
                if(peekTokenIs(tok_close_paren)) {
                    nextToken();
                    return identifiers;
                }

                    
                // if the param list is not empty
                // Get the first element of the param list
                nextToken();
                std::shared_ptr<Identifier> ident(new Identifier(curToken, curToken.value));
                identifiers.push_back(ident);

                while(peekTokenIs(tok_comma)) {
                    nextToken();
                    nextToken();
                    std::shared_ptr<Identifier> ident(new Identifier(curToken, curToken.value));
                    identifiers.push_back(ident);
                }

                if(!expectPeek(tok_close_paren))
                    // we should return nullptr here
                    return identifiers;

                return identifiers;
            }


            std::shared_ptr<Function> parseFunctionStatement() {
                Token cur = curToken; // this is the token containing 'def'

                if(!expectPeek(tok_identifier))
                    return nullptr;

                // cur = foo, peek = (
                Token ident = curToken; // this is the token containing the function name 

                if(!expectPeek(tok_open_paren))
                    return nullptr;


                // parse the parameters list
                std::vector<std::shared_ptr<Identifier>> param_list = parseParamList();

                if(!expectPeek(tok_open_brace))
                    return nullptr;

                // parse the body which must end with a return statement
                std::shared_ptr<BlockStatement> body = parseBlockStatement();

                std::shared_ptr<Function> func(new Function(cur, ident, param_list, body));
                
                return func;
            }
            
            std::shared_ptr<Statement> parseStatement() {
                switch(curToken.ttype) {
                    case tok_var:
                        return parseVarDecl();
                    
                    case tok_return:
                        return parseReturn();

                    case tok_if:
                        return parseIfStatement();
                    
                    case tok_def:
                        return parseFunctionStatement();

                    // i.e. statement does not start with 'var' keyword
                    case tok_identifier:
                    {
                        if(peekToken.ttype == tok_assign)
                            return parseAssignStatement();
                        
                        if(peekToken.ttype == tok_open_paren)
                            return parseExpressionStatements();
                    }
                        

                    default:
                        return parseExpressionStatements();
                }
            }

            std::shared_ptr<Program> parseProgram() {
                std::vector<std::shared_ptr<Statement>> stmts;
                

                while(curToken.ttype != tok_eof) {

                    std::shared_ptr<Statement> stmt = parseStatement();

                    if(stmt) {
                        stmts.push_back(stmt);
                    }

                    nextToken();
                }

                std::shared_ptr<Program> ast(new Program(stmts));
                return ast;
            }

            void peekError(TokenType t) {
                std::string err = "expected next token to be "+tokToString(t)+" got "+tokToString(peekToken.ttype)+" instead";
                errors.push_back(err);
            }

            std::vector<std::string> getErrors() {
                return errors;
            }

            
        private : 
            std::vector<std::string> errors;
            Lexer lexer;
            Token curToken;
            Token peekToken;

            std::unordered_map<TokenType, int> precedences = {
                {tok_eq, EQUALS},
                {tok_not_eq, EQUALS},
                {tok_lt, LESSGREATER},
                {tok_gt, LESSGREATER},
                {tok_plus, SUM},
                {tok_minus, SUM},
                {tok_mult, PRODUCT},
                {tok_open_paren, CALL}
            };
            
    };    
}

