function out = cf2gaby(cf,a,b,y,varargin)
%CF2GABY conditional expectations given characteristic function
%
%   CF2GABY(CF,A,B,Y) 
%   Given the discounted characteristic function CF of a stochastic process 
%   X, this function returns the discounted conditional expectation of 
%   exp(A'*X) given (b'*X)<=Y. 
%
%   WARNING: CF2GABY will pass (NX)x(K) arguments to CF, thus CF should be
%   expect an input of dimension (NX)x(K) and return an (1)x(K) output.
%
%   CF2GABY(CF,A,B,Y,MAX) 
%   Integrates the characteristic function to MAX (default MAX=250).
%
%   Example: Black-Scholes model with corresponding characteristic function
%   CF. One way to obtain the option price for a given strike level K is by
%   employing CF2GABY:
%
%   S0      = 100;
%   x0      = log(S0);
%   rf      = 0.05;
%   tau     = 1;
%   sigma   = 0.25;
%   cf      = @(u) exp(-rf*tau+i.*u.*x0+i.*u.*tau*(rf-1/2*sigma^2)-1/2*u.^2*sigma^2);
%   K       = 105
%   C       = cf2gaby(cf,1,-1,-log(K)) - K*cf2gaby(cf,0,-1,-log(K))

%   Author:     matthias.held@web.de
%   Date:       2015-05-03

uMax            = 250;
if ~isempty(varargin)
    uMax        = varargin{1};
end

aFun            = @(u) repmat(a,1,size(u,2));
out             = cf(-i*a)/2;
out             = out -1/pi*quadgk(@(u) imag(cf(-i*aFun(u)+b*u).*exp(-i.*u.*y))./u,0,uMax);
end

