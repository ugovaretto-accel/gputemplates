#include <cassert>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <boost/typeof/std/ostream.hpp>
#include <boost/mpl/assert.hpp>
#include <cmath>


// Include all of Proto
#include <boost/proto/proto.hpp>

using namespace boost;
// Create some namespace aliases
namespace mpl = boost::mpl;
namespace fusion = boost::fusion;
namespace proto = boost::proto;

// Allow unqualified use of Proto's wildcard pattern
using proto::_;

template < int S, typename T >
struct Array
{
    T data_[ S ];
    template < int  E >
    const T& At() const { return data_[ E ]; }
    template < int E >
    T& At() { return data_[ E ]; }
};


//==============================================================================
//1) define terminals, build expression and evaluate. Operator overloads
//   apply automatically operators to instances of types contained into terminals
//   e.g. '<<' operator is applied to terminal< ostream > which in turn applies it to ostream instance

proto::terminal< std::ostream& >::type cout_= { std::cout };
template < typename Expr >
void Evaluate( Expr const& expr )
{
    proto::default_context ctx;
    proto::eval( expr, ctx );
}

void Test1()
{
    Evaluate( cout_ << "Ciao" << "!" );
}
//==============================================================================
//2) use placeholders to identify variables; placeholder should carry information
//   on how to access data which is used in context::operator() invoked by eval()
template <int I> struct Placeholder {};
proto::terminal< Placeholder< 0 > >::type const _1 = {{}};
proto::terminal< Placeholder< 1 > >::type const _2 = {{}};

struct CalculatorContext : proto::callable_context< CalculatorContext const >
{
    Array< 2, double > args_;
    typedef double result_type;
    template < int I >
    double operator()( proto::tag::terminal, Placeholder< I > ) const
    {
        return args_.At< I >();
    }
};


void Test2()
{
    CalculatorContext ctx;
    ctx.args_.At< 0 >() = 45.;
    ctx.args_.At< 1 >() = 50.;
    const double d = proto::eval( ( _2 + _1 ) / _2 * 100., ctx );
    std::cout << d << std::endl;
    
}
//==============================================================================
//3) Generate callable function objects which evaluate expressions:
//   i.   declare expression wrapper: this is the function object
//   ii.  define domain associated with expression wrapper
//   iii. define expression wrapper: behaves as expressions + overloaded operator()
//   the domain is what informs Proto of the existence of a wrapper 
template < typename Expr > struct Calculator;

struct CalculatorDomain : proto::domain< proto::generator< Calculator > >
{};

template < typename Expr >
struct Calculator
	: proto::extends<Expr, Calculator< Expr >, CalculatorDomain >
{
	typedef proto::extends< Expr, Calculator< Expr >, CalculatorDomain >
		base_type;
	Calculator( const Expr& expr = Expr() ) : base_type( expr )
	{}

	typedef double result_type;

	double operator()( double a1 = 0., double a2 = 0. ) const
	{
		CalculatorContext ctx;
		ctx.args_.At< 0 >() = a1;
		ctx.args_.At< 1 >() = a2;
		return proto::eval( *this, ctx );
	}
};

template < typename T > struct Typeis { typedef T type; };

template < typename T >
inline Typeis< T > ComputeType( const T& T ) { return Typeis< T >(); }


void Test3()
{
	// when using an expression wrapper wrap terminal declarations with
	// wrappe type; type is an expression wrapper which is in turn an expression
	Calculator< proto::terminal< Placeholder< 0 > >::type > const _1;
	Calculator< proto::terminal< Placeholder< 1 > >::type > const _2;
	// since placeholders are defined within an expression wrapper
	// an expression which used such placeholders as variables will
	// generate an instance of an expression wrapper which in this case
	// has an overloaded operator() accepting two parameters
	double result = ( ( _2 - _1 ) / _2 * 100. )( 45.0, 50.0 );
	assert( result == ( 50.0 - 45.0 ) / 50.0 * 100. );
}

void Test3b()
{
	Calculator< proto::terminal< Placeholder< 0 > >::type > const _1;
	Calculator< proto::terminal< Placeholder< 1 > >::type > const _2;
	double a1[ 4 ] = { 56, 84, 37, 69 };
	double a2[ 4 ] = { 65, 120, 60, 70 };
	double a3[ 4 ] = { 0, 0, 0, 0 };
	// expression wrapper creates a function object which can be passed to an algorithm
	std::transform( a1, a1 + 4, a2, a3, ( _2 - _1 ) / _2 * 100. );
	std::copy( a3, a3 + 4, std::ostream_iterator< double >( std::cout, "," ) );
}

//==============================================================================
// 4) Define grammar for compile time check. Compile time check should happen
//    before invoking eval()


struct CalculatorGrammar :
	proto::or_<
	    proto::plus< CalculatorGrammar, CalculatorGrammar >,
	    proto::minus< CalculatorGrammar, CalculatorGrammar >,
	    proto::multiplies< CalculatorGrammar, CalculatorGrammar >,
	    proto::divides< CalculatorGrammar, CalculatorGrammar >,
	    proto::terminal< proto::_ > // <- matches any terminal 
	>
{};

// declare another expression wrapper
template < typename Expr > struct CalculatorChecked;

// define a new domain associated with wrapper
struct CalculatorDomainChecked : proto::domain< proto::generator< CalculatorChecked > >
{};

// define expression wrapper; check expression before evaluation
template < typename Expr >
struct CalculatorChecked
	: proto::extends<Expr, CalculatorChecked< Expr >, CalculatorDomainChecked >
{
	typedef proto::extends< Expr, CalculatorChecked< Expr >, CalculatorDomainChecked >
		base_type;
	CalculatorChecked( const Expr& expr = Expr() ) : base_type( expr )
	{}

	typedef double result_type;

	double operator()( double a1 = 0., double a2 = 0. ) const
	{
		BOOST_MPL_ASSERT( ( proto::matches< Expr, CalculatorGrammar > ) ); 
		CalculatorContext ctx;
		ctx.args_.At< 0 >() = a1;
		ctx.args_.At< 1 >() = a2;
		return proto::eval( *this, ctx );
	}
};

// it is possible to simply pass the grammar as a parameter to the domain:
//struct calculator_domain
//  : proto::domain< proto::generator<calculator>, calculator_grammar >
//{};

//==============================================================================

// 5) Define function

struct CosFct
{
	typedef double result_type; //required
	typedef double arg_type;
	double operator()( arg_type const& x ) const
	{
		return std::cos( x );
	}
};

template < typename FT, typename ArgT >
    typename proto::result_of::make_expr<
    typename proto::tag::function,
    FT,
    ArgT const& >::type const
Fun( ArgT const& arg )
{
	return proto::make_expr< proto::tag::function, CalculatorDomain >( FT(), boost::ref( arg ) );
}

void Test4()
{
	Calculator< proto::terminal< Placeholder< 0 > >::type > const _1;
	Calculator< proto::terminal< Placeholder< 1 > >::type > const _2;

	std::cout << Fun<CosFct>( _1 + _2 )( 0.2, 0.4 ) << std::endl;
}

//------------------------------------------------------------
struct SinFct
{
	typedef double result_type;
	double operator()( double const& d ) const
	{
		return std::sin( d );
	}
};


struct CalculatorDomainSinTag;

// Create tag to identify function type
struct SinTag {};

template< typename ArgT >
typename proto::result_of::make_expr<
    SinTag,
    CalculatorDomainSinTag,
	SinFct,
	ArgT >::type
Sin( ArgT const& arg )
{
	
	return proto::make_expr< SinTag, CalculatorDomainSinTag >( SinFct(), arg );
}


// Context
// Expression Wrapper: is-a expression, depends-on Domain, uses Context
// Domain: depends-on Expression Generator
// Expression Generator: depends-on Expression Wrapper


// Context
struct CalculatorContextSinTag : proto::callable_context< CalculatorContextSinTag const >
{
    Array< 2, double > args_;
    typedef double result_type;
    template < int I >
    double operator()( proto::tag::terminal, Placeholder< I > ) const 
    {
        return args_.At< I >();
    }
    template < typename Expr1, typename Expr2 >
	result_type operator()( SinTag, Expr1& e1, Expr2& e2 ) const
	{
		return proto::eval( e1, *this )( proto::eval( e2, *this ) );
	}
};

template < typename Expr > struct CalculatorSinTag;

struct CalculatorDomainSinTag :
	proto::domain< proto::generator< CalculatorSinTag > >
{};

// Expression wrapper
template < typename Expr >
struct CalculatorSinTag 
	: proto::extends< Expr, CalculatorSinTag< Expr >, CalculatorDomainSinTag >
{

	typedef proto::extends< Expr, CalculatorSinTag< Expr >, CalculatorDomainSinTag >
		base_type;
	
	CalculatorSinTag( const Expr& expr = Expr() ) : base_type( expr )
	{
		proto::display_expr( expr );
	}

	typedef double result_type;

	double operator()( double a1 = 0., double a2 = 0. ) const
	{
		CalculatorContextSinTag ctx;
		ctx.args_.At< 0 >() = a1;
		ctx.args_.At< 1 >() = a2;
		return proto::eval( *this, ctx );
	}
};



//template < typename Expr >
//struct Calculator
//	: proto::extends<Expr, Calculator< Expr >, CalculatorDomain >
//{
//	typedef proto::extends< Expr, Calculator< Expr >, CalculatorDomain >
//		base_type;
//	Calculator( const Expr& expr = Expr() ) : base_type( expr )
//	{}
//
//	typedef double result_type;
//
//	double operator()( double a1 = 0., double a2 = 0. ) const
//	{
//		CalculatorContext ctx;
//		ctx.args_.At< 0 >() = a1;
//		ctx.args_.At< 1 >() = a2;
//		return proto::eval( *this, ctx );
//	}
//};



void Test4b()
{

	CalculatorSinTag< proto::terminal< Placeholder< 0 > >::type > const _1;
	CalculatorSinTag< proto::terminal< Placeholder< 1 > >::type > const _2;

	std::cout << Sin( _1 + _2 )( 0.3, 0.4 ) << std::endl;
}

template < typename Expr >
struct SinExpr
{
	BOOST_PROTO_BASIC_EXTENDS( Expr, SinExpr, CalculatorDomain );
	BOOST_PROTO_EXTENDS_FUNCTION();
};


void Test4c()
{
	// without assignment to {} it
	// gives warning since it's an unininitialized const (no assignment operator)
	SinExpr< proto::terminal< SinFct >::type > const Sin = {}; 

	std::cout << Sin( _1 + _2 )( 0.5 ) << std::endl;
}

//==============================================================================



int main( int , char** )
{
	Test4c();
#ifdef _MSC_VER
	getchar();
#endif
	return 0;
}