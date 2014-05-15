function [C K] = cf2call(cf,varargin)
%CF2CALL compute call option prices from characteristic function
%
%   [C K] = CF2CALL(CF) 
%   returns call option prices C on a grid of corresponding strikes K given 
%   the characteristic function specified by the function handle CF. Using
%   real argument u, CF(u) is the characteristic function of X. With 
%   complex argument u=-v*i, CF(u) returns the moment generating function. 
%
%   [C K] = CF2CALL(CF,AUX) 
%   obtains option prices for optional FRFT algorithm parameters stored in 
%   the stucture AUX with fields
%   aux.N       number of points for FRFT evaluation        (2^13)
%   aux.uMax    range of integration of the char. function  (0:100)
%   aux.damp    dampening factor in the Carr Madan approach (1.5)
%   aux.dx      discretization of log strike range          (2/N)
%   aux.x0      log of spot underlying                      (0)
%   aux.K       vector of strike evaluation points
%
%   [C K] = CF2CALL(CF,AUX,PAR)
%   [C K] = CF2CALL(CF,[],PAR) 
%   Specify a parameter structure PAR to hand over to CF(u,PAR).
%
%   Example: Black Scholes
%   par         = struct('x0', log(100), 'rf', 0.05, 'sigma', 0.25)
%   tau         = 0.5
%   aux.K       = [50:1:150]'
%   aux.x0      = par.x0
%   cf          = @(u) exp(-par.rf*tau)*cflib(u,tau,par,'BS')
%   C           = cf2call(cf,aux);
%   arbBound    = max(100-aux.K*exp(-par.rf*tau),0);
%   plot(aux.K,[C arbBound])
%   

%   Author:     matthias.held@web.de
%   Date:       2014-05-01

damp            = 1.5;
uMax            = 100;
N               = 2^13;
x0              = 0;
K0              = [];
if length(varargin)==2 & isstruct(varargin{2})
    par = varargin{2};
    cf = @(u) cf(u,par);
end
if length(varargin)>=1 & isstruct(varargin{1})
    varargin=varargin{1};
    if isfield(varargin,'damp')
        damp           = varargin.damp;
    end
    if isfield(varargin,'uMax')
        uMax            = varargin.uMax;
    end
    if isfield(varargin,'N')
        N               = varargin.N;

    end
    if isfield(varargin,'dx')
        dx              = varargin.dx;
    end
    if isfield(varargin,'x0')
        x0              = varargin.x0;
    end    
    if isfield(varargin,'K') & ~isempty(varargin.K)
        K0              = varargin.K;
        K0              = reshape(K0,1,length(K0));
    end
end
du              = uMax/(N-1);
dx              = 2/N;
u               = [0:N-1]'*du;
xMin            = -N*dx/2+x0;
alpha           = du*dx/2/pi;
x               = [0:N-1]'*dx + xMin;

try z = cf(u-i*(damp+1));catch err1;end
if exist('err1')
   try z = cf((u-i*(damp+1)).').';catch err2;end
end
if ~exist('z')
    error('problem with the char fun, dimension mismatch');
end
z               = z./repmat((damp^2+damp-u.^2+i*(2*damp+1)*u),1,size(z,2));
z               = repmat(exp(-u*i*(xMin)),1,size(z,2)).*z.*du;

z([1 end],:)    = 0.5*z([1 end],:);
z               = real(frft(z,alpha));
z               = repmat(exp(-damp*x),1,size(z,2)).*z/pi;

K               = exp(x);
C               = z;
if ~isempty(K0)
    [~, idx]        = sort(abs(bsxfun(@minus,K,K0)));
    Cl              = C(idx(1,:),:);
    dC              = C(idx(2,:),:)-C(idx(1,:),:);
    dK              = repmat(K(idx(2,:))-K(idx(1,:)),1,size(C,2));
    C               = Cl + dC./dK.*repmat((K0'-K(idx(1,:))),1,size(C,2));
    K               = reshape(K0,length(K0),1);
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
 
 
 
