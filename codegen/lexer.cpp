#ifndef LEXER_CPP
#define LEXER_CPP


#include <string>
#include <memory>
#include <map>
#include <vector>
#include <iostream>

namespace choco{

    enum TokenType {
        tok_assign, // =
        tok_plus,
        tok_minus,
        tok_mult,
        tok_lt,
        tok_gt,
        tok_not,
        tok_eq,     // ==
        tok_not_eq, // !=

        tok_open_paren,
        tok_close_paren,
        tok_open_brace,
        tok_close_brace,

        tok_eof,
        tok_semicolon,
        tok_comma,
        tok_open_sbracket,
        tok_close_sbracket,

        tok_var,    // for variable declarations
        tok_def,    // for function decls
        tok_return, 
        tok_if,
        tok_else,

        tok_identifier,
        tok_numeric,
        tok_true,
        tok_false,

        tok_illegal
    };


    struct Token {
        TokenType ttype;
        std::string value;
    };


    struct Lexer {
        // We have two position pointers because we need to peek one char ahead
        std::string input;
        int curCursor; // points to current char
        int nextCursor; // after next char
        char ch; // currently has the char pointer to by 'curCursor'

        // constructor takes care of initializing these variables
        Lexer(std::string code)
            : input(code), curCursor(0), nextCursor(1), ch(input[curCursor]) {}

        
        std::map<std::string, TokenType> keywords{{"var", tok_var}, {"def", tok_def}, 
                                                {"return", tok_return}, {"if", tok_if}, {"else", tok_else}, 
                                                {"true", tok_true}, {"false", tok_false}};

        void skipWhitespace() {
            while(ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                readChar();
            }
        }

        char peekChar() {
            if (nextCursor >= input.length())
                return 0;
            
            return input[nextCursor];
        }

        // Purpose of read char is to read the next char in the stream
        void readChar() {
            if (nextCursor >= input.length())
                ch = 0;
            else
                ch = input[nextCursor];

            curCursor = nextCursor;
            nextCursor += 1;
        }

        TokenType lookupIdent(std::string word) {
            if (TokenType t = keywords[word]) {
                return t;
            }

            return tok_identifier;
        }

        std::string readIdentifier() {
            int cur = curCursor;
            
            while (isalpha(ch) || (ch == '_')) {
                readChar();
            }

            return input.substr(cur, curCursor - cur);
        }

        std::string readNumber() {
            int cur = curCursor;

            while(isdigit(ch)) {
                readChar();
            }

            return input.substr(cur, curCursor - cur);
        }
        
    };

    static Token newToken(TokenType ty, char ch)
    {
        Token tok;
        tok.ttype = ty;
        tok.value = std::string(1, ch);
        return tok;
    }

    static std::string tokToString(TokenType t)
    {
        switch (t)
        {
        case tok_assign:
            return "tok_assign";
        case tok_plus:
            return "tok_plus";
        case tok_minus:
            return "tok_minus";
        case tok_mult:
            return "tok_mult";
        case tok_lt:
            return "tok_lt";
        case tok_gt:
            return "tok_gt";
        case tok_not:
            return "tok_not";
        case tok_eq:
            return "tok_eq";
        case tok_not_eq:
            return "tok_not_eq";

        case tok_open_paren:
            return "tok_open_paren";
        case tok_close_paren:
            return "tok_close_paren";
        case tok_open_brace:
            return "tok_open_brace";
        case tok_close_brace:
            return "tok_close_brace";

        case tok_eof:
            return "tok_eof";
        case tok_comma:
            return "tok_comma";
        case tok_semicolon:
            return "tok_semicolon";
        case tok_open_sbracket:
            return "tok_open_sbracket";
        case tok_close_sbracket:
            return "tok_close_sbracket";

        case tok_var:
            return "tok_var";
        case tok_def:
            return "tok_def";
        case tok_return:
            return "tok_return";
        case tok_if:
            return "tok_if";
        case tok_else:
            return "tok_else";

        case tok_identifier:
            return "tok_identifier";
        case tok_numeric:
            return "tok_numeric";
        case tok_illegal:
            return "tok_illegal";
        }

        std::cout<<"Unregistered/Unsupported token encountered while lexing"<<std::endl;
        return nullptr;
    }

    static Token nextToken(Lexer &l) {
        Token tok;

        l.skipWhitespace();

        switch(l.ch) {
            case '=' :
                if(l.peekChar() == '=') {
                    char c = l.ch;
                    l.readChar();   // move the cursor 
                    tok.ttype = tok_eq;
                    tok.value = std::string(1, c) + std::string(1, l.ch);
                }
                else
                    tok = newToken(tok_assign, l.ch); // this is how we convert a character to a std::string

                break;
            case '+':
                tok = newToken(tok_plus, l.ch);
                break;
            case '-':
                tok = newToken(tok_minus, l.ch);
                break;
            case '*':
                tok = newToken(tok_mult, l.ch);
                break;
            case '<':
                tok = newToken(tok_lt, l.ch);
                break;
            case '>':
                tok = newToken(tok_gt, l.ch);
                break;
            case '!':
                if (l.peekChar() == '=') {
                    char c = l.ch;
                    l.readChar();
                    tok.ttype = tok_not_eq;
                    tok.value = std::string(1, c) + std::string(1, l.ch);
                } else
                    tok = newToken(tok_not, l.ch);

                break;
            case ';' :
                tok = newToken(tok_semicolon, l.ch);
                break;
            case ',':
                tok = newToken(tok_comma, l.ch);
                break;
            case '(':
                tok = newToken(tok_open_paren, l.ch);
                break;
            case ')':
                tok = newToken(tok_close_paren, l.ch);
                break;
            case '{':
                tok = newToken(tok_open_brace, l.ch);
                break;
            case '}':
                tok = newToken(tok_close_brace, l.ch);
                break;
            case '[':
                tok = newToken(tok_open_sbracket, l.ch);
                break;
            case ']':
                tok = newToken(tok_close_sbracket, l.ch);
                break;
            case 0 :
                tok.ttype = tok_eof;
                tok.value = "";
                break;

            default :
                // Here we read in identifiers like "foo_bar", "var", "def"
                //  we need to be able to tell user-defined identifiers apart from language keywords
                if(isalpha(l.ch) || (l.ch == '_')) {
                    tok.value = l.readIdentifier();
                    tok.ttype = l.lookupIdent(tok.value);
                    return tok;
                } 
                else if (isdigit(l.ch)) {
                    tok.ttype = tok_numeric;
                    tok.value = l.readNumber();
                    return tok;
                }
                else {
                    tok = newToken(tok_illegal, l.ch);
                }
        }

        // Advance cursor by 1 and read the next char in 'ch'
        l.readChar();

        return tok;
    }

}


// Write text suite for the lexer. It will contain the testing function

#endif // LEXER_CPP