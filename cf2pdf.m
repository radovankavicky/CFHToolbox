function [f x] = cf2pdf(cf,varargin)
%CF2PDF recover probability density function from characteristic function
%
%   [F X] = CF2PDF(CF) 
%   returns the normalized probability density function F and corresponding 
%   sampling points X for a characteristic function CF via the Fractional 
%   Fourier Transform (FRFT). 
%   For real argument u, CF(u) yields the characteristic function of X. 
%   For complex argument u=-v*i, CF(u) is the moment generating function.
%
%   [F X] = CF2PDF(CF,AUX) 
%   obtains the pdf using optional FRFT algorithm parameters stored in the 
%   stucture AUX with fields
%   aux.N       number of evaluation points     (2^13)
%   aux.uMax    upper limit of integration      (200)
%   aux.dx      sample point step size          (3/N)
%   aux.x0      log of center of evaluation     (0)
%   aux.x       range of pdf evaluation points. (N points centered at x0)
%               If used, the values dx,x0,N are 
%               omitted and set to values that 
%               correspond to aux.x.
%
%   Example: pdf corresponding to Heston's SV model, with N = 512
%   tau         = 0.5
%   par         = struct(   'rf', 0.05, 'x0', log(100), 'v0', 0.2*0.2, ...
%                           'kappa', 0.85, 'theta', 0.25*0.25, ...
%                           'sigma', 0.10, 'rho', -0.7)
%   aux.x       = log(100)+linspace(-0.75,0.75,512);
%   cf          = @(u) exp(-par.rf*tau)*cflib(u,tau,par,'Heston')
%   [f x]       = cf2pdf(cf,aux);
%   plot(x,f)

%   Author: matthias.held@web.de
%   Date:   2014-05-27

uMax            = 200;
N               = 2^13;
dx              = 3/N;
x0              = 0;

if length(varargin)>=1 & isstruct(varargin{1})
    varargin=varargin{1};
    if isfield(varargin,'uMax')
        uMax            = varargin.uMax;
    end
    if isfield(varargin,'N')
        N               = varargin.N;
    end
    if isfield(varargin,'dx')
        dx              = varargin.dx;
    end
    if isfield(varargin,'x')
        x               = varargin.x;
        N               = length(x);
        x               = reshape(x,N,1);
        xMin            = x(1);
        dx              = x(2)-x(1);
    end
    if isfield(varargin,'x0')
        x0              = varargin.x0;
    end
end

du              = uMax/(N-1);
u               = [0:N-1]'*du;
if ~exist('xMin')
xMin            = -N*dx/2+x0;
end
if ~exist('x')
x               = [0:N-1]'*dx + xMin;
end
alpha           = du*dx/2/pi;

try z = cf(u);catch err1;end
if exist('err1')
   try z = cf(u.').';catch err2;end
end
if ~exist('z')
    error('Dimension mismatch in CF.');
end
z               = repmat(exp(-u*i*(xMin)),1,size(z,2)).*z.*du;
z([1 end],:)    = 0.5*z([1 end],:);
f               = real(frft(z,alpha))/pi;
f               = f/cf(0);
f(f<0)          = 0;
f               = f.*dx;

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
 
 
 
