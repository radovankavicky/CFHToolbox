function [C P] = cf2spread(cf,K,varargin)
%CF2SPREAD European spread option prices from characteristic function
%
%   C = CF2SPREAD(CF,K) 
%   returns prices C of spread calls with strikes K. The options pay
%   max(S1-S2-K,0). CF is a characteristic function expecting 2D-array 
%   arguments U and V that correspond to S1 and S2, CF returns a 2D-array. 
%
%   [~, P] = CF2SPREAD(CF,K) 
%   returns prices C of spread puts with strikes K. The options pay
%   max(K-(S1-S2),0).
%
%   [C P] = CF2SPREAD(CF,K,AUX) 
%   obtains prices of spread calls or puts for optional FFT parameters
%   stored in the stucture AUX with fields
%   AUX.N       number of points for FRFT evaluation        (2^10)
%   AUX.uMax    upper and lower integration limit           (200)
%   aux.x1Limit log asset range around discounted forward   (3)
%   aux.x2Limit log asset range around discounted forward   (3)
%
%   If AUX.N exceeds available memory, the function throws an error. 
%   For longer maturities, set AUX.x1Limit and AUX.x2Limit to higher values 
%   if necessary, say (3*MATURITY).
%
%   Example: Spread call option on the spread of two bivariate lognormally
%            distributed stocks:
%
%   S1          = 100;
%   S2          = 105;
%   K           = [0:0.5:15];
%   tau         = 0.5;
%   rf          = 0.05;
%   s1          = 0.20;
%   s2          = 0.25;
%   rho         = 0.65;
%   cf          = @(u,v) exp(-rf*tau+i*u*(log(S1)+tau*(rf-1/2*s1^2)) ...
%                                   +i*v*(log(S2)+tau*(rf-1/2*s2^2)) ...
%                       -1/2*(u.^2*s1^2+v.^2*s2^2+2*u.*v.*s1*s2*rho)*tau);
%   plot(K,cf2spread(cf,K));

%   Author:     matthias.held@web.de
%   Date:       2014-06-13

uMax            = 200;
N               = 2^10;
override        = 0;

if length(varargin)==1 & isstruct(varargin{1})
    varargin=varargin{1};
    if isfield(varargin,'uMax')
        uMax            = varargin.uMax;
    end
    if isfield(varargin,'N')&&(~isempty(varargin.N))
        N               = varargin.N;
    end
    if isfield(varargin,'override')&&(varargin.override==1)
        override        = 1;
    end
    if isfield(varargin,'x1Limit')&&(~isempty(varargin.x1Limit))
        x1Limit          = varargin.x1Limit;
    end
    if isfield(varargin,'x2Limit')&&(~isempty(varargin.x2Limit))
        x2Limit          = varargin.x2Limit;
    end
end

[~,sys]         = memory;
if (sys.PhysicalMemory.Available<(N^2)*320)&&(override == 0)
    error(['Memory requirement exceeds available memory size.' ...
           ' Set AUX.override to [1] to override this message.']);
end

jj              = [0:N-1]';
[du dv]         = deal((2*uMax)/N);
[u0 v0]         = deal(-N/2*du);
[u v]           = deal(u0 + jj*du);

if ~exist('x1Limit')
    x1Limit         = 3;
end
dx1             = x1Limit/N;
x10             = log(cf(-i,0))-N/2*dx1;
x1              = x10 + jj*dx1;

if ~exist('x2Limit')
    x2Limit         = 3;
end
dx2             = x2Limit/N;
x20             = log(cf(0,-i))-N/2*dx2;
x2              = x20 + jj*dx2;

a1              = du*dx1/2/pi;
a2              = dv*dx2/2/pi;

u               = repmat(u,1,N);
v               = repmat(v',N,1);

x1              = repmat(x1,1,N);
x2              = repmat(x2',N,1);

% z is an array.    from top to bottom: change in x1 and u. 
%                   from left to right: change in x2 and v.
% Hence: the first FFT (top to bottom) is on x1, the second FFT is on x2, 
% i.e. on z' (left to right).
pre             = 1/(2*pi)^2*exp(-i*u0*x1-i*v0*x2);
z               = cf(u,v)*du*dv.*exp(-i*x10*(u-u0)-i*x20*(v-v0));
z               = real(pre.*(frft(frft(z,a1).',a2).'));
[rK cK]         = size(K);
[C P]           = deal(zeros(rK,cK));
for k = 1:length(K)
    cMesh       = 1*(exp(x1)-exp(x2)-K(k)>=0);
    C(k)        = sum(sum(z.*cMesh.*(exp(x1)-exp(x2)-K(k))))*dx1*dx2;
    P(k)        = sum(sum(z.*(1-cMesh).*(-(exp(x1)-exp(x2)-K(k)))))*dx1*dx2;
end
   
end

function f = frft(x,a)
% Fractional Fourier Transform (FRFT, Chourdakis (2008))
% The FRFT allows to seperate the spacing of the frequency domain 
% integration variable u and that of the time domain variable x.
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
