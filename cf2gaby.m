function [out Y] = cf2gaby(cf,a,b,y,varargin)
%CF2GABY conditional expectations given characteristic function
%
%   C = CF2GABY(CF,A,B,Y) 
%   Given the (discounted) characteristic function of a stochastic process,
%   this function returns the expectation of exp(A'X) given (b'X<=Y) for
%   all points in Y. By default, CF2GABY will use FFT methods.
%
%   [C Y] = CF2GABY(CF,A,B,[]) or CF2GABY(CF,A,B) 
%   If Y is left empty, CF2GABY will generate points around zero or around
%   the level x0 specified in the AUX structure (see below).
%
%   [C Y] = CF2GABY(CF,A,B,Y,AUX) or CF2GABY(CF,A,B,[],AUX)
%   Tune the method via the AUX structure with fields:
%   aux.N       number of points for FFT evaluation         (2^13)
%   aux.u0      lower bound for numerical integration       (1e-8)
%   aux.uMax    upper bound for numerical integration       (200)
%   aux.x0      log of spot underlying                      (0)
%   aux.quad    For quadrature method, set to 'true' or 1   (0)
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
%   Date:       2015-06-11

if ~exist('y');y=[];end
uMax            = 200;
u0              = 1e-8;
N               = 2^13;
x0              = 0; 
useFFT          = 1;

if length(varargin)>=1 & isstruct(varargin{1})
    varargin=varargin{1};
    if isfield(varargin,'uMax')
        uMax            = varargin.uMax;
    end
    if isfield(varargin,'u0')
        u0              = varargin.u0;
    end
    if isfield(varargin,'N')
        N               = varargin.N;
    end
    if isfield(varargin,'x0')
        x0              = varargin.x0;
    end
    if isfield(varargin,'quad') ...
       &&(  (ischar(varargin.quad) && strcmp(varargin.quad,'true')) ...
          || (isnumeric(varargin.quad)&&varargin.quad==1))
      useFFT        = 0;
    end
end

if useFFT == 1;
    jj              = [0:N-1]';
    du              = (uMax+u0)/(N-1);
    u               = u0 + jj*du;
    nY              = N;
    if isempty(y)
        dy          = 2/(N-1);
        Y0          = x0 - N/2*dy;
        Y           = Y0 +jj*dy;
    elseif isscalar(y)
        dy          = 2/(N-1);
        Y0          = y;
        Y           = Y0 + jj*dy;
        nY          = 1;
    elseif ismatrix(y)
        dy          = mean(diff(y));
        if std(diff(y))>1e-8
            warning('Variable spacing of y may deny FFT applicability.');
        end
        Y0          = y(1);
        Y           = Y0 + jj*dy;
        nY          = length(y);
    else
        'Wrong y submitted. Expecting: empty [], scalar or vector';
    end
 
    alpha           = du*dy/2/pi;

    if ismatrix(a)||ismatrix(b)
    a = reshape(a,length(a),1);
    b = reshape(b,length(b),1);
	a = repmat(a,1,N);
    end
    
    try z = cf(-a.'*i+u*b.');catch err1;end
    if exist('err1')
        try z = cf((-a.'*i+u*b.').').' ;catch err2;end
    end
    if ~exist('z')
        error('problem with the char fun, dimension mismatch');
    end
    z               = z./u.*du.*exp(-i*jj*du*Y0);
    z([1,end])      = 1/2*z([1;end]);
    z               = frft(z,alpha);
    z               = imag(exp(-i*u0*Y).*z);
    out             = cf(-i*a(:,1))/2-1/pi*z;
    out             = out(1:nY,:);
else
    
    aFun            = @(u) repmat(a,1,size(u,2));
    cf              = @(u) reshape(cf(u),1,size(u,2));
    out             = cf(-i*a)/2;
    out             = out -1/pi*quadgk(@(u) ...
                        imag(cf(-i*aFun(u)+b*u).*exp(-i.*u.*y))./u,0,uMax);    
end
end

function f = frft(x,a)
% Fractional Fourier Transform (FRFT, Chourdakis (2008))
% The FRFT allows to seperate the spacing of the frequency domain
% integration variable u and that of the time domain variable
% x. (In our case, the x variable is the log asset return.)
% a = (du * dx) / (2 * pi);

[N nx]      = size(x);
e1          = repmat(exp(-pi.*i.*a .* ([0:(N-1)]').^2),1,nx);
e2          = repmat(exp( pi.*i.*a .* ([N:-1:1]').^2),1,nx);
z1          = [x.*e1 ; zeros(N,nx)];
z2          = [1./e1 ; e2];
fz1         = fft(z1);
fz2         = fft(z2);
fz          = fz1.*fz2;
ifz         = ifft(fz);
f           = e1.*ifz(1:N,:);
end
 





