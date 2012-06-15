///Expression templates sample code (no external documentation yet).
///Author: Ugo Varetto 
///  
///#define STORE_EXPRESSIONS if you want to store expressions into objects for delayed evaluation
///#define EVAL_TEXT to print the textual representation of expressions
///Skip to the 'TEST CODE' section for examples of definitions and evaluations of expressions

// OUTPUT without EVAL_TEXT #defined
//Expression templates 1
//----------------------------------
//-41.3
//-41.3
//-41.3
//-82.6
//12.1138
//
//Expression templates 2 - placeholders
//----------------------------------
//-41.3
//-41.3
//-41.3
//-82.6
//12.1138
//1.36109e+006
//
//Compile time algorithms
//----------------------------------
//124251

//OUTPUT with EVAL_TEXT #defined
//Expression templates 1
//----------------------------------
//(-3)*12.1+1-2*3
//(-3)*12.1+1-2*3
//(-3)*12.1+1-2*3
//2*(-3)*12.1+1-2*3
//12.1+exp(2*(-3)*12.1+1-2*3/19.3)
//
//Expression templates 2 - placeholders
//----------------------------------
//(-3)*12.1+1-2*3
//(-3)*12.1+1-2*3
//(-3)*12.1+1-2*3
//2*(-3)*12.1+1-2*3
//12.1+exp(2*(-3)*12.1+1-2*3/19.3)
//(-2*(-3)*12.1+1-2*3)*12.1+1-2*2*(-3)*12.1+1-2*3+(-2*(-3)*12.1+1-2*3)*12.1+1-2*2*
//(-3)*12.1+1-2*3*(-2*(-3)*12.1+1-2*3)*12.1+1-2*2*(-3)*12.1+1-2*3*(-2*(-3)*12.1+1-
//2*3)*12.1+1-2*2*(-3)*12.1+1-2*3/(-2*(-3)*12.1+1-2*3)*12.1+1-2*2*(-3)*12.1+1-2*3+
//(-2*(-3)*12.1+1-2*3)*12.1+1-2*2*(-3)*12.1+1-2*3
//
//Compile time algorithms
//----------------------------------
//124251

//NOTE #1: The implementation relies on the compiler to inline everything
//and on this:
//"Unless bound to a reference or used to initialize a named object,
//a temporary object is destroyed at the end of the full expression in
//which it was created."
//"The C++ Programming Language" third edition, page 254, section 10.4.10, 2nd paragraph

//NOTE #2: The end goal here is *NOT* to perform actual evaluation but run-time
//generation of OpenCL and CUDA CODE in textual form 

//NOTE #3: It is possible to keep const references to values only if expressions are not
//stored for delayed evaluation - see note #1

//TODO: explore the use of an Haskell-like monadic approach (i.e. overload ->, <-, >>, >>= and
//pass copies (references) of (to) context around)

#include <vector>
#include <list>
#include <iostream>
#include <iterator>
#include <string>
#include <map>
#include <cmath>
#include <stdexcept>
#include <sstream>


typedef double ScalarType;

///Simple string wrapper.
struct String
{
    std::string str_;
    String() : str_( "UNINITIALIZED" )  {}
    String( const std::string& s ) : str_( s ) {}
    String( ScalarType d )
    {
        std::ostringstream iss;
        if( d < 0. ) iss << '(' << d << ')';
        else iss << d;
        str_ = iss.str();
    }
    friend std::ostream& operator<<( std::ostream& os, const String& s )
    {
        os << s.str_;
        return os;
    }
};

//#define EVAL_TEXT

#ifdef EVAL_TEXT
typedef String EvalType;
#define STORE_EXPRESSIONS
#else
typedef double EvalType; 
#endif


//==============================================================================

///Base class for all expression types; Implements compile-time polymorphism when
///used with the CRTP and optionally the Barton-Nackman trick to add functionality
///to derived classes.
///Each derived class must then be derived from this class as:
///<code>
///struct DerivedExpression : Expr< DerivedExpression >
///</code>
template < class DerivedT >
struct Expr
{
    typedef DerivedT Type;
    operator const DerivedT&() const
    {
        return static_cast< const DerivedT& >( *this );
    }
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const 
    {
        return static_cast< const DerivedT& >( *this ).Eval( ctx );
    }
};



#define DEFINE_STR_TWO_OPERAND_OPERATOR( OP_ ) \
    /*inline*/ String operator OP_( const String& s1, const String& s2 ) \
    { \
        return s1.str_ + #OP_ + s2.str_; \
    } \
    /*inline*/ String operator OP_( const String& s1, ScalarType v ) \
    { \
        return s1.str_ + #OP_ + String( v ).str_; \
    } \
     /*inline*/ String operator OP_( ScalarType v, const String& s2 ) \
    { \
        return String( v ).str_ + #OP_ + String( s2 ).str_; \
    } 

#define DEFINE_STR_ONE_OPERAND_OPERATOR( OP_ ) \
    /*inline*/ String operator OP_( const String& s1 ) \
    { \
    return std::string( "(" ) + #OP_ + s1.str_ + ')'; \
    }

#define DEFINE_STR_FUN_OPERATOR( FUNNAME_, FUN_ ) \
    /*inline*/ String FUNNAME_( const String& s1 ) \
    { \
        return std::string( #FUN_ ) + '(' +  s1.str_ + ')'; \
    }

DEFINE_STR_TWO_OPERAND_OPERATOR( + )
DEFINE_STR_TWO_OPERAND_OPERATOR( - )
DEFINE_STR_TWO_OPERAND_OPERATOR( * )
DEFINE_STR_TWO_OPERAND_OPERATOR( / )
DEFINE_STR_ONE_OPERAND_OPERATOR( - )
DEFINE_STR_ONE_OPERAND_OPERATOR( + )
//==============================================================================
// OPERATIONS
//==============================================================================
// This is the code that gets executed at run-timee. 
// The following function objects are called by Eval() method of the expression
// they are embedded in and hopefulled inlined. 

/// unary '-'
struct Neg
{
    EvalType operator()( EvalType v ) const { return -v; } 
};
/// unary '+'
struct Plus
{
    EvalType operator()( EvalType v ) const { return v; } 
};
/// +
struct Add
{
    EvalType operator()( EvalType v1, EvalType  v2 ) const { return v1 + v2; } 
};
/// -
struct Sub
{
    EvalType operator()( EvalType v1, EvalType  v2 ) const { return v1 - v2; } 
};
/// x
struct Mul
{
    EvalType operator()( EvalType v1, EvalType  v2 ) const { return v1 * v2; } 
};
/// /
struct Div
{
    EvalType operator()( EvalType v1, EvalType  v2 ) const { return v1 / v2; } 
};

template < class T > T Exponential( const T& v )
{
    return std::exp( v );
}

template <> String Exponential< String >( const String& v )
{
    return "exp(" + v.str_ + ")";
}

template < class T > T Cos( const T& v )
{
    return std::cos( v );
}

template <> String Cos< String >( const String& v )
{
    return "cos(" + v.str_ + ")";
}


/// Exponential
struct ExpFun
{
    EvalType operator()( EvalType v ) const { return Exponential( v ); }
};

/// Cosine
struct CosFun
{
    EvalType operator()( EvalType v ) const { return Cos( v ); }
};

//------------------------------------------------------------------------------
///Context class to hold run-time information; used to store variables and 
///constants. Could be used for storing memoized results as well
struct Context
{
    typedef std::map< std::string, EvalType > MapType;
    typedef MapType::key_type KeyType;
    typedef MapType::mapped_type MappedType;
    struct VariableTag {};
    struct ConstTag {};
    MapType varMap_;
    MapType constMap_;
    const MappedType& operator()( VariableTag,  KeyType i ) const
    {
        MapType::const_iterator it = varMap_.find( i );
        if( it == varMap_.end() ) throw std::runtime_error( "Element not found" );
        return it->second;
    }
    MappedType& operator()( VariableTag, KeyType i )
    {
        MapType::iterator it = varMap_.find( i );
        if( it == varMap_.end() ) throw std::runtime_error( "Element not found" );
        return it->second;
    }
    void Set( VariableTag, const KeyType& varName, EvalType varValue )
    {
        varMap_[ varName ] = varValue;
    }
};


//------------------------------------------------------------------------------
///Compile-time context: categories of data are selected by type tags; data
///references are selected with a placeholder id. Data are stores into a vector;
///using a compile-time fixed size array should allow for easier inlining
struct PlaceHolderContext
{
    struct VariableTag {};
    struct ConstTag {};
    
    typedef int Key;

    // doesn't need to be inside context
    template < int ID > struct PlaceHolder {};

    std::vector< EvalType > variables_;
    std::vector< EvalType > constants_;
    template < int ID >
    EvalType& operator()( VariableTag, PlaceHolder< ID > )
    {
        return variables_[ ID ];
    }
    template < int ID >
    const EvalType& operator()( VariableTag, PlaceHolder< ID > ) const
    {
        return variables_[ ID ];
    }
    template < int ID >
    const EvalType& operator()( ConstTag, PlaceHolder< ID > ) const
    {
        return constants_[ ID ];
    }
   
};


//==============================================================================
// EXPRESSIONS
//==============================================================================
///Variable expression class: loads/stores data from/to context
template < class E > struct AssignExpression;
struct VarExpression : Expr< VarExpression >
{
    typedef std::string VarKey;
    VarKey varKey_;
    VarExpression( const VarKey& varKey ) :  varKey_( varKey ) {}
    template < class CtxT >  EvalType Eval( CtxT& ctx ) const 
    { 
        return ctx( typename CtxT::VariableTag(), varKey_ );
    }
    AssignExpression< EvalType > operator=( EvalType v );
    /*{
        return AssignExpression< EvalType >( *this, v );
    }*/
    template < class Ex >
    AssignExpression< Ex > operator=( const Expr<Ex>& e );
    /*{
        return AssignExpression< E >( *this, e );
    }*/
};

///Assign expression: assign new value to variable in context at evaluation time
template < class E >
struct AssignExpression : Expr< AssignExpression< E > >
{
#ifndef STORE_EXPRESSIONS
    typedef const E& ExpressionType;
    typedef const VarExpression& VarExpressionType;
#else 
    typedef E ExpressionType;
    typedef VarExpression VarExpressionType;
#endif
    VarExpressionType v_;
    ExpressionType e_;
    AssignExpression( const VarExpression& v, const E& e ) : v_( v ), e_( e ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const
    {
        ctx( typename CtxT::VariableTag(), v_.varKey_ ) = e_.Eval( ctx );
        return ctx( typename CtxT::VariableTag(), v_.varKey_ );
    }
};
template <>
struct AssignExpression< EvalType > : Expr< AssignExpression< EvalType > >
{
#ifndef STORE_EXPRESSIONS
    typedef const VarExpression& VarExpressionType;
#else
    typedef VarExpression VarExpressionType;
#endif
    VarExpressionType v_;
    EvalType e_;
    AssignExpression( const VarExpression& v, EvalType e ) : v_( v ), e_( e ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const
    {
        ctx( typename CtxT::VariableTag(), v_.varKey_ ) = e_;
        return e_;
    }
};


//Define here methods of VarExpression requiring knowledge of AssignExpression class
inline AssignExpression< EvalType > VarExpression::operator=( EvalType v )
{
    return AssignExpression< EvalType >( *this, v );
}
template < class E >    
inline AssignExpression< E > VarExpression::operator=( const Expr<E>& e )
{
    return AssignExpression< E >( *this, e );
}


///Assign expression with place holders, easier since the assignment instance
///needs only to store an id
template < int ID, class E >
struct PHAssignExpression : Expr< PHAssignExpression< ID, E > >
{
#ifndef STORE_EXPRESSIONS
    typedef const E& ExpressionType;
#else
    typedef E ExpressionType;
#endif
    ExpressionType e_;
    PHAssignExpression( const E& e ) : e_( e ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const
    {
        ctx( typename CtxT::VariableTag(), typename CtxT::template PlaceHolder< ID >() ) = e_.Eval( ctx );
        return ctx( typename CtxT::VariableTag(), typename CtxT::template PlaceHolder< ID >() );
    }
};
template < int ID >
struct PHAssignExpression< ID, EvalType > : Expr< PHAssignExpression< ID, EvalType > >
{
    EvalType e_;
    PHAssignExpression( EvalType e ) : e_( e ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const
    {
        ctx( typename CtxT::VariableTag(), typename CtxT::template PlaceHolder< ID >() ) = e_;
        return e_;
    }
};


//Variable expression class: loads/stores data from/to context
template < int ID >
struct PlaceHolderExpression : Expr< PlaceHolderExpression< ID > >
{
    template < class CtxT >  EvalType Eval( CtxT& ctx ) const 
    { 
        return ctx( typename CtxT::VariableTag(), typename CtxT::template PlaceHolder< ID >() );
    }
    PHAssignExpression< ID, EvalType > operator=( EvalType v )
    {
        return PHAssignExpression< ID, EvalType >( v );
    }
    template < class E >
    PHAssignExpression< ID, E > operator=( const Expr< E >& e )
    {
        return PHAssignExpression< ID, E >( e );
    }
};

///Constant expression
struct ConstExpression : Expr< ConstExpression >
{
    EvalType v_;
    ConstExpression( EvalType v ) : v_( v ) {}   
    template < class CtxT >
    EvalType Eval( CtxT& ) const
    {
        return v_; 
    }
};

///Function expression: Applies funtion to evaluated expression
template< class F, class E >
struct FunExpression : Expr< FunExpression< F, E > > 
{
#ifndef STORE_EXPRESSIONS
    typedef const E& ExpressionType;
#else
    typedef E ExpressionType;
#endif
    F f_;
    ExpressionType e_;
    FunExpression( const E& e ) : e_( e ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const { return f_( e_.Eval( ctx ) ); }
};


///Prefix operator expression e.g. unary '-'
template < class OP, class E > 
struct OneOpExpression : Expr< OneOpExpression< OP, E  > >
{
#ifndef STORE_EXPRESSIONS
    typedef const E& ExpressionType;
#else
    typedef E ExpressionType;
#endif
    OP op_;
    ExpressionType e_;
    OneOpExpression( const E& e ) : e_( e ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const
    {
        return op_( e_.Eval( ctx ) );
    }
};

template< class OP >
struct OneOpExpression< OP, EvalType > : Expr< OneOpExpression< OP, EvalType > >
{
    OP op_;
    EvalType v_;//for on the fly evaluation const& works too
    OneOpExpression( EvalType v ) : v_( v ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ) const 
    {
        return op_( v_ );
    } 
};


///Binary operator expression e.g. '+'
template < class OP, class E1, class E2 >
struct TwoOpExpression : Expr< TwoOpExpression< OP, E1, E2 > >
{
#ifndef STORE_EXPRESSIONS
    typedef const E1& FirstExprType;
    typedef const E2& SecondExprType;
#else
    typedef E1 FirstExprType;
    typedef E2 SecondExprType;
#endif
    OP op_;
    FirstExprType e1_;
    SecondExprType e2_;
    TwoOpExpression( const E1& e1, const E2& e2 ) : e1_( e1 ), e2_( e2 ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const
    {
        return op_( e1_.Eval( ctx ), e2_.Eval( ctx ) );
    }
};

template < class OP, class E1 >
struct TwoOpExpression< OP, E1, EvalType > : Expr< TwoOpExpression< OP, E1, EvalType > >
{
#ifndef STORE_EXPRESSIONS
    typedef const E1& FirstExprType;
#else
    typedef E1 FirstExprType;
#endif
    OP op_;
    FirstExprType e1_;
    EvalType v_;//for on the fly evaluation const& works too
    TwoOpExpression( const E1& e1, EvalType v ) : e1_( e1 ), v_( v ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const
    {
        return op_( e1_.Eval( ctx ), v_ );
    }
};

template < class OP, class E2 >
struct TwoOpExpression< OP, EvalType, E2 > : Expr< TwoOpExpression< OP, EvalType, E2 > >
{
#ifndef STORE_EXPRESSIONS
    typedef const E2& SecondExprType;
#else
    typedef E2 SecondExprType;
#endif
    OP op_;
    EvalType v_;//for on the fly evaluation const& works too
    SecondExprType e2_; 
    TwoOpExpression( EvalType v, const E2& e2 ) : v_( v ), e2_( e2 ) {}
    template < class CtxT >
    EvalType Eval( CtxT& ctx ) const
    {
        return op_( v_, e2_.Eval( ctx ) );
    }
};

//==============================================================================
// OPERATORS
//==============================================================================
///Macro to simplify the task of generating expressions with binary operators
#define DEF_TWO_OPERAND_OPERATOR( OPERATOR__, OPERATION__ ) \
    template < typename LHS, typename RHS > \
    /*inline*/ TwoOpExpression< OPERATION__, LHS, RHS >  operator OPERATOR__( const Expr< LHS >& lhs, const Expr< RHS >& rhs ) \
    { \
        return TwoOpExpression< OPERATION__, LHS, RHS >( static_cast< const LHS& >( lhs ), static_cast< const RHS& >( rhs ) ); \
    } \
    template < typename LHS > \
    /*inline*/ TwoOpExpression< OPERATION__, LHS, EvalType >  operator OPERATOR__( const Expr< LHS >& lhs, const EvalType& rhs ) \
    { \
        return TwoOpExpression< OPERATION__, LHS, EvalType >( static_cast< const LHS& >( lhs ),  rhs ); \
    } \
    template < typename RHS > \
    /*inline*/ TwoOpExpression< OPERATION__, EvalType, RHS >  operator OPERATOR__( const EvalType& lhs, const Expr< RHS >& rhs ) \
    { \
        return TwoOpExpression< OPERATION__, EvalType, RHS >( lhs, static_cast< const RHS& >( rhs ) ); \
    } 

///Macro to generate expressions with unary prefix operators
#define DEF_ONE_OPERAND_OPERATOR( OPERATOR__, OPERATION__ ) \
    template < typename OPERAND > \
    /*inline*/ OneOpExpression< OPERATION__, OPERAND >  operator OPERATOR__( const Expr< OPERAND >& operand ) \
    { \
        return OneOpExpression< OPERATION__, OPERAND >( static_cast< const OPERAND& >( operand ) ); \
    }

#define DEF_ONE_OPERAND_FUNCTION( FUNCTION__, FUNCTION_NAME__ ) \
    template < class OPERAND > \
    /*inline*/ FunExpression< FUNCTION__, OPERAND >  FUNCTION_NAME__( const Expr< OPERAND >& operand ) \
    { \
        return FunExpression< FUNCTION__, OPERAND >( static_cast< const OPERAND& >( operand ) ); \
    }

//==============================================================================
// RUN-TIME
//==============================================================================
///Interface to expression type: used at run-time to store expressions
///of any type; note that the virtual 'Eval()' function is called only
///once to evaluate the entire complie-time syntax tree.
template < class CtxT >
struct IExpression
{
    virtual EvalType Eval( CtxT& ) const = 0;
    virtual ~IExpression() {}
};

///Implementation of IExpression interface to hold instances of any
///expression types
template < class ExprT, class CtxT >
struct Expression : IExpression< CtxT > 
{
    ExprT e_;
    Expression( const ExprT& e ) : e_( e ) {}
    EvalType Eval( CtxT& ctx ) const { return e_.Eval( ctx ); }
};

///Class uses to store expressions of any kind through IExpression
///implementations
template < class ContextT >
struct ExpressionWrapper
{
    typedef IExpression< ContextT > ExpressionType;
    ExpressionType* pExpr_; //USE SMART POINTERS WITH CUSTOM ALLOCATORS!
    ExpressionWrapper( ExpressionType* pExpr ) : pExpr_( pExpr ) {}
    ExpressionWrapper() : pExpr_( 0 ) {}
    template < class RHS > ExpressionWrapper( const Expr< RHS >& rhs ) 
        : pExpr_( new Expression< RHS, ContextT >( static_cast< const RHS& >( rhs ) ) ) {} 
    EvalType Eval( ContextT& ctx ) const { return pExpr_->Eval( ctx ); }
    template < class RHS > const ExpressionWrapper& operator=( const Expr< RHS >& rhs )
    {
        delete pExpr_;
        pExpr_ = new Expression< RHS, ContextT >( static_cast< const RHS& >( rhs ) );
        return *this;
    }
    ~ExpressionWrapper()
    {
        delete pExpr_;
    }
};

//==============================================================================
// TEST CODE
//==============================================================================

//------------------------------------------------------------------------------
// 1) Expression templates
//------------------------------------------------------------------------------
//Define operators and functions as expressions.
DEF_ONE_OPERAND_OPERATOR( -, Neg )
DEF_ONE_OPERAND_OPERATOR( +, Plus )
DEF_TWO_OPERAND_OPERATOR( +, Add )
DEF_TWO_OPERAND_OPERATOR( -, Sub )
DEF_TWO_OPERAND_OPERATOR( *, Mul )
DEF_TWO_OPERAND_OPERATOR( /, Div )
DEF_ONE_OPERAND_FUNCTION( ExpFun, Exp )

///Evaluate and print passed expression
template < class E, class CtxT > void eval( const E& e, CtxT& ctx )
{
    std::cout << e.Eval( ctx ) << std::endl;
}

///Generate and return an expression for delayed evaluation.
ExpressionWrapper< Context > GenerateExpression()
{
    ConstExpression c = 2.0;
    return  c - 2.0 + Exp( c );
}


///Test expression templates
void ExpressionTemplatesTest1()
{
    typedef VarExpression Var;
    Context ctx; //context
    Var x( "x" ); //bind x var to "x" key
    Var y( "y" ); //bind y var to "y" key
       
    ctx.Set( Context::VariableTag(), "x", 3.0 ); //associate value with "x" variable key
    ctx.Set( Context::VariableTag(), "y", 12.1 ); //associate value with "y" variable key
        

#define EXPRESSION -x*y + 1. - 2. * x
  
    // 1 - On the fly evaluation
    const EvalType evaluatedExpression = ( EXPRESSION ).Eval( ctx );
    std::cout << evaluatedExpression << std::endl;
    // 2 - Store expression into wrapper for subsequent evaluation
    ExpressionWrapper< Context > ew = EXPRESSION;
    std::cout << ew.Eval( ctx ) << std::endl;
    // 3 - pass expression to function 
    eval( EXPRESSION, ctx );
    // 4 - assignment + on the fly evaluation
    std::cout << ( x = 2.f * ( EXPRESSION ) ).Eval( ctx ) << std::endl;
    // 5 - store new expression with function call
    ExpressionWrapper< Context > ew2 = y + Exp( x / 19.3f);
    std::cout << ew2.Eval( ctx ) << std::endl; 
    //std::cout << ew2.Eval( ctx ) << std::endl;
    // 6 - long expression
    // FOLLOWING EXPRESSION GIVES: 'operator +' : decorated name length exceeded, name was truncated
    // see e.g.: http://msdn.microsoft.com/en-us/library/074af4b6%28VS.80%29.aspx
    //std::cout << ( EXPRESSION + EXPRESSION*EXPRESSION*EXPRESSION/EXPRESSION+EXPRESSION).Eval( ctx ) << std::endl;  
}

///Test expression templates with place holders to handle variables
void ExpressionTemplatesTest2()
{
    
    PlaceHolderContext ctx; //context
    PlaceHolderExpression< 0 > x; //bind x var to "x" value
    PlaceHolderExpression< 1 > y; //bind y var to "y" value
       
    ctx.variables_.push_back( 3.0 ); //associate value with placeholder 0
    ctx.variables_.push_back( 12.1 ); //associate value with placeholder 1

#undef EXPRESSION    
#define EXPRESSION (-x*y + 1.0 - 2.0 * x) 

    // 1 - On the fly evaluation
    const EvalType evaluatedExpression = EXPRESSION.Eval( ctx );
    std::cout << evaluatedExpression << std::endl;
    // 2 - Store expression into wrapper for subsequent evaluation
    ExpressionWrapper< PlaceHolderContext > ew = EXPRESSION;
    std::cout << ew.Eval( ctx ) << std::endl;
    // 3 - pass expression to function 
    eval( EXPRESSION, ctx );
    // 4 - assignment + on the fly evaluation
    std::cout << (x = 2. * EXPRESSION).Eval( ctx ) << std::endl;
    // 5 - store new expression with function call in previous wrapper instance
    ew = y + Exp( x / 19.3 ); // Exp is a function wrapper
    std::cout << ew.Eval( ctx ) << std::endl;
    // 6 - long expression
    std::cout << ( EXPRESSION + EXPRESSION*EXPRESSION*EXPRESSION/EXPRESSION+EXPRESSION).Eval( ctx ) << std::endl;  
}


//------------------------------------------------------------------------------
// 2) Compile-time algorithms
//------------------------------------------------------------------------------

///Sample compile time Loop construct
template < int N, class ExprT, class ContextT >
struct Loop
{
    static ContextT& Eval( ContextT& ctx )
    {
        return  ExprT::template Execute< N >( Loop< N-1, ExprT, ContextT>::Eval( ctx ) );  
    }   
}; 
template< class ExprT, class ContextT >
struct Loop< 0, ExprT, ContextT >
{
    static ContextT& Eval( ContextT& ctx )
    {
        return ExprT::template Execute< 0 >( ctx );
    }
};
struct Ctx
{
    ScalarType v_;
    Ctx() : v_( ScalarType() ) {}
    friend std::ostream& operator<<( std::ostream& os, const Ctx& ctx )
    {
        os << ctx.v_;
        return os;
    }   
};
struct Sum
{
    template < int N >    
    static Ctx& Execute( Ctx& ctx )
    {
        ctx.v_ += N;
        return ctx;
    } 
};


///Test compile time algorithms
void CompileTimeAlgorithmsTest()
{
    Ctx ctx;
    //Loop; warning need to explicitly set -ftemplate-depth-n with n = 500 
    std::cout << Loop< 498, Sum, Ctx >::Eval( ctx ) << std::endl;   
}
//------------------------------------------------------------------------------

//==============================================================================
// main - entry point
//==============================================================================
int main(int argc, char** argv)
{
    std::cout << "Expression templates 1" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    ExpressionTemplatesTest1();
    std::cout << "\nExpression templates 2 - placeholders" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    ExpressionTemplatesTest2();
    std::cout << "\nCompile time algorithms" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    CompileTimeAlgorithmsTest();
    return 0;
}
