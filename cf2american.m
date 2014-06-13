function [V D] = cf2american(cf,K,tau,type,varargin)
%CF2AMERICAN American options from characteristic function of returns
%
%   V = CF2AMERICAN(CF,K,TAU,TYPE)
%   returns the normalized American option price of TYPE='Call' or 'Put'
%   with time to maturity TAU. K is the normalized strike level (a scalar).
%   The characteristic function of the return distribution, CF(U,T), is a 
%   function handle that expects the inputs U and T, where T is the time 
%   horizon of the return distribution. All normalization is with respect
%   to the spot asset price.
%   
%   [V D] = CF2AMERICAN(CF,K,TAU,TYPE)
%   Returns the normalized American option price V and the corresponding
%   option delta for an American option of TYPE='Call' or 'Put' with time
%   to maturity TAU.
%
%   V = CF2AMERICAN(CF,TAU,TYPE,AUX)
%   returns the normalized American option price using optional parameters 
%   stored in the structure AUX with fields:
%   aux.NFFT    number of points for FRFT evaluation    (2^13)
%   aux.NSTEP   number of time steps til maturity       (500)
%   aux.damp    damping factor                          (2)
%   aux.xLimit  log return range = 0 +/- 0.5*xLimit     (max(3,3*tau))
%
%   Important note: The underlying distribution must be i.i.d., i.e. from
%   the class of Lévy models. Models such as the Heston model, or models
%   with stochastic factor dynamics can not be solved via CF2AMERICAN.
%
%   Example: American put option in Black scholes world:
%
%   par         = struct('rf',0.05,'q',0.03,'sigma',0.25,'x0',0);
%   tau         = 0.75;
%   S           = 100;
%   K           = 95;
%   cf          = @(u,tau) cflib(u,tau,par,'BS');
%   Put         = S*cf2american(cf,K/S,tau,'Put')

    %   Author:     matthias.held@web.de
%   Date:       2014-05-01

damp            = 2;
NFFT            = 2^13;

if length(varargin)>=1 & isstruct(varargin{1})
    varargin=varargin{1};
    if isfield(varargin,'damp')
        damp           = varargin.damp;
    end
    if isfield(varargin,'NFFT')&&(~isempty(varargin.NFFT))
        NFFT           = varargin.NFFT;
    end
    if isfield(varargin,'NSTEP')&&(~isempty(varargin.NSTEP))
        NSTEP           = varargin.NSTEP;
    end
    if isfield(varargin,'xLimit')&&(~isempty(varargin.xLimit))
        xLimit          = varargin.xLimit;
    end
end

if ~exist('NSTEP') 
    NSTEP           = 500;
end
if ~exist('dt')
	dt              = tau/NSTEP;
end
if ~exist('xLimit')
    xLimit          = 3*max(1,tau);
end
    dx              = (xLimit)/(NFFT-1);

cf              = @(u) cf(u,dt);

if strcmpi(type,'call'); 
    s = 1; 
elseif strcmpi(type,'put') s=-1;
else
    error('Specify type=''Call'' or type=''Put''');
end

damp            = s*(-damp);
uSelect         = [50:50:5000]';

try z = cf(uSelect);catch err1;end
if exist('err1')
   try z = cf(uSelect.').';catch err2;end
end
if ~exist('z')
    error('problem with the char fun, dimension mismatch');
end
uMax            = uSelect(min(find(abs(z)<=1e-8)));
w               = [1/2 ; ones(NFFT-2,1) ; 1/2];
du              = uMax/(NFFT-1);

if du>0.25;
    warning(['Large steps in image space required: dU=' num2str(du) '.'...
             ' You might consider increasing (aux.N).'])
end

jj              = [0:NFFT-1]';
u               = jj*du;

x0              = -NFFT/2*dx;
x               = x0 + jj*dx;

a               = du*dx/(2*pi); 

V               = max(s*(exp(x)-K),0);

if exist('err1')
   try cf = cf((damp*i-u).').';end
else
    try cf = cf(damp*i-u);end    
end
cf                  = reshape(cf,NFFT,1);

for k = 1:NSTEP
    vHat            = exp(i*u*x0).*frft(exp(damp*x).*V.*w.*dx,-a);
    V               = exp(-damp*x).*1/pi.*real(frft(exp(-i*u*x0).*vHat.*cf.*du.*w,a));
    V               = max(V,s*(exp(x)-K));
end
    D               = diff(V(NFFT/2+[2 0]))/diff(exp(x(NFFT/2+[2 0])));
    V               = V(NFFT/2+1);
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