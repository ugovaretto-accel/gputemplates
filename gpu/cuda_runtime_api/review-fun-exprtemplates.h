#ifndef EXPRTEMPLATES_

#ifdef __CUDACC__
#define CUDAAPI /*__host__*/ __device__ 
#else
#define CUDAAPI 
#endif

#ifdef DOUBLE_PRECISION
// use this for double precision
typedef double ScalarType;
typedef double EvalType;
#else
typedef float ScalarType;
typedef float EvalType;
#endif


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
    CUDAAPI operator const DerivedT&() const
    {
        return static_cast< const DerivedT& >( *this );
    }
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ctx ) const 
    {
        return static_cast< const DerivedT& >( *this ).Eval( ctx );
    }
};


//==============================================================================
// OPERATIONS & FUNCTIONS
//==============================================================================
// This is the code that gets executed at run-timee. 
// The following function objects are called by Eval() method of the expression
// they are embedded in and hopefulled inlined. 

/// unary '-'
struct Neg
{
    CUDAAPI EvalType operator()( EvalType v ) const { return -v; } 
};
/// unary '+'
struct Plus
{
    CUDAAPI EvalType operator()( EvalType v ) const { return v; } 
};
/// +
struct Add
{
    CUDAAPI EvalType operator()( EvalType v1, EvalType  v2 ) const { return v1 + v2; } 
};
/// -
struct Sub
{
    CUDAAPI EvalType operator()( EvalType v1, EvalType  v2 ) const { return v1 - v2; } 
};
/// x
struct Mul
{
    CUDAAPI EvalType operator()( EvalType v1, EvalType  v2 ) const { return v1 * v2; } 
};
/// /
struct Div
{
    CUDAAPI EvalType operator()( EvalType v1, EvalType  v2 ) const { return v1 / v2; } 
};

/// Cosine
template < class T > 
CUDAAPI T Cos( const T& v )
{
    return cos( v );
}

/// Exponential
template < class T > 
CUDAAPI T Exponential( const T& v )
{
    return exp( v );
}


/// Exponential
struct ExpFun
{
    CUDAAPI EvalType operator()( EvalType v ) const { return Exponential( v ); }
};

/// Cosine
struct CosFun
{
    CUDAAPI EvalType operator()( EvalType v ) const { return Cos( v ); }
};


//==============================================================================
// EXPRESSIONS, problem assign expression requires way of extracting reference
//              to location to store result
//==============================================================================

template < int I, int J, typename E >
struct AssignExpression : Expr< AssignExpression< I, J, E > >
{
#ifndef STORE_EXPRESSIONS
    typedef const E& ExpressionType;
#else
    typedef E ExpressionType;
#endif
    ExpressionType e_;
    CUDAAPI AssignExpression( const E& e ) : e_( e ) {}
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ctx ) const
    {
        ctx( typename CtxT::VariableTag(), typename CtxT::template PlaceHolder< I , J >() ) = e_.Eval( ctx );
        return ctx( typename CtxT::VariableTag(), typename CtxT::template PlaceHolder< I, J >() );
    }
};




template < int I, int J >
struct AssignExpression< I, J, EvalType > : Expr< AssignExpression< I, J, EvalType > >
{
    EvalType e_; //should be another expression, if not expression is evaluated when instances are created, but which is the type ? 
	             //Cannot use 'auto' on the GPU anyway
    CUDAAPI AssignExpression( EvalType e ) : e_( e ) {}
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ctx ) const
    {
        ctx( typename CtxT::VariableTag(), typename CtxT::template PlaceHolder< I, J >() ) = e_;
        return e_;
    }
};


//Variable expression class: loads/stores data from/to context
template < int I, int J > // assuming 2D index
struct VarExpression : Expr< VarExpression< I, J > >
{
    template < class CtxT >
	CUDAAPI EvalType Eval( CtxT& ctx ) const 
    { 
        return ctx( typename CtxT::VariableTag(), typename CtxT::template PlaceHolder< I, J >() );
    }
    CUDAAPI AssignExpression< I, J, EvalType > operator=( EvalType v )
    {
        return AssignExpression< I, J, EvalType >( v );
    }
    template < class E >
    CUDAAPI AssignExpression< I, J, E > operator=( const Expr< E >& e )
    {
        return AssignExpression< I, J, E >( e );
    }
};

///Constant expression
struct ConstExpression : Expr< ConstExpression >
{
    EvalType v_;
    CUDAAPI ConstExpression( EvalType v ) : v_( v ) {}   
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ) const
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
    CUDAAPI FunExpression( const E& e ) : e_( e ) {}
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ctx ) const { return f_( e_.Eval( ctx ) ); }
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
    CUDAAPI OneOpExpression( const E& e ) : e_( e ) {}
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ctx ) const
    {
        return op_( e_.Eval( ctx ) );
    }
};

template< class OP >
struct OneOpExpression< OP, EvalType > : Expr< OneOpExpression< OP, EvalType > >
{
    OP op_;
    EvalType v_;//for on the fly evaluation const& works too
    CUDAAPI OneOpExpression( EvalType v ) : v_( v ) {}
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ) const 
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
    CUDAAPI TwoOpExpression( const E1& e1, const E2& e2 ) : e1_( e1 ), e2_( e2 ) {}
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ctx ) const
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
    CUDAAPI TwoOpExpression( const E1& e1, EvalType v ) : e1_( e1 ), v_( v ) {}
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ctx ) const
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
    CUDAAPI TwoOpExpression( EvalType v, const E2& e2 ) : v_( v ), e2_( e2 ) {}
    template < class CtxT >
    CUDAAPI EvalType Eval( CtxT& ctx ) const
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
    inline /*CUDAAPI*/ TwoOpExpression< OPERATION__, LHS, RHS >  operator OPERATOR__( const Expr< LHS >& lhs, const Expr< RHS >& rhs ) \
    { \
        return TwoOpExpression< OPERATION__, LHS, RHS >( static_cast< const LHS& >( lhs ), static_cast< const RHS& >( rhs ) ); \
    } \
    template < typename LHS > \
    inline /*CUDAAPI*/ TwoOpExpression< OPERATION__, LHS, EvalType >  operator OPERATOR__( const Expr< LHS >& lhs, const EvalType& rhs ) \
    { \
        return TwoOpExpression< OPERATION__, LHS, EvalType >( static_cast< const LHS& >( lhs ),  rhs ); \
    } \
    template < typename RHS > \
    inline /*CUDAAPI*/ TwoOpExpression< OPERATION__, EvalType, RHS >  operator OPERATOR__( const EvalType& lhs, const Expr< RHS >& rhs ) \
    { \
        return TwoOpExpression< OPERATION__, EvalType, RHS >( lhs, static_cast< const RHS& >( rhs ) ); \
    } 

///Macro to generate expressions with unary prefix operators
#define DEF_ONE_OPERAND_OPERATOR( OPERATOR__, OPERATION__ ) \
    template < typename OPERAND > \
    inline OneOpExpression< OPERATION__, OPERAND >  operator OPERATOR__( const Expr< OPERAND >& operand ) \
    { \
        return OneOpExpression< OPERATION__, OPERAND >( static_cast< const OPERAND& >( operand ) ); \
    }

#define DEF_ONE_OPERAND_FUNCTION( FUNCTION_NAME__, PARAM1__, EXPRESSION__ ) \
    template < typename ExprT, typename RetExprT > \
    inline RetExprT FUNCTION_NAME__( const ExprT& PARAM1__, const RetExprT& retExpr = EXPRESSION__ ) \
    { \
        return retExpr; \
    }

#define DEF_TWO_OPERAND_FUNCTION( FUNCTION_NAME__, PARAM1__, PARAM2__, EXPRESSION__ ) \
    template < typename Expr1T, typename Expr2T, typename RetExprT > \
    inline RetExprT FUNCTION_NAME__( const Expr1T& PARAM1__, const Expr2T& PARAM2__, const RetExprT& retExpr = EXPRESSION__ ) \
    { \
        return retExpr; \
    }




#endif // EXPRTEMPLATES_
