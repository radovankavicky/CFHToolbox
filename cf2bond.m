function [P Y alpha beta] = cf2bond(tau,x0,K0,K1,H0,H1,R0,R1,L0,L1,jump,varargin)
%CF2BOND zero bond pricing in an affine jump-diffusion framework 
%
%   [P Y] = CF2BOND(TAU,X0,K0,K1,H0,H1,R0,R1) 
%   Returns the bond prices P and yields Y for maturities TAU corresponding 
%   to a system that consists solely of diffusive risks. The interest rate
%   is driven by the constant R0 and a linear combination of the process
%   variables R1*X. For example, if only the kth element of X represents
%   the interest rate, set R1(k)=1.
%
%   [P Y] = CF2BOND(TAU,X0,K0,K1,H0,H1,R0,R1,L0,L1,jump) 
%   Returns the bond prices and yields for maturities TAU when when there
%   are diffusive and (multiple) jump risks present.
%
%   Leave any component empty [] if you do not require it in your model.
%
%   The underlying process of X is affine jump-diffusive:
%   dX=M(X)*dt+S(X)*dW+J*dZ, where
%   M(X)        = K0 + K1*X,    process drift
%   S(X)*S(X)'  = H0 + H1*X,    process diffusive variance
%   L(X)        = L0 + L1'*X,   jump intensities
%   R(X)        = R0 + R1'*X,   interest rate process
%   jump(c)     = E(exp(c*J)),  moment generating function of the 
%                               multivariate jump distribution
%
%   The vector x0 contains the initial state.
%
%   Single vs. multiple jumps: The number of jumps (nJ) is defined by the 
%   length of L0 or the number of columns in L1. The JUMP transform is a 
%   user supplied function that expects an array of dimension (dim(X)xK) 
%   and returns an array of dimension (nJxK). Each output row of JUMP
%   corresponds to one jump transform applied to the input array.
%
%   % Example: Vasicek interest rate model: dr=kappa*(theta-r)*dt+sigma*dW
%
%   kappa   = 2.00;
%   theta   = 0.055;
%   sigma   = 0.02;
%   r0      = 0.035;
%   tau     = [1/52:1/52:10]';
%   [~,Y]   = cf2bond(tau,r0,kappa*theta,-kappa,sigma^2,[],[],1);
%   plot(tau,Y);title('Vasicek annualized yields');

%   Author: matthias.held@web.de 
%   Date:   2014 05 05

if ~exist('K0');K0=[];end
if ~exist('K1');K1=[];end
if ~exist('H0');H0=[];end
if ~exist('H1');H1=[];end
if ~exist('R0');R0=[];end
if ~exist('R1');R1=[];end
if ~exist('L0');L0=[];end
if ~exist('L1');L1=[];end
if ~exist('jump');jump=[];end

if ~isempty(varargin) && length(varargin) == 1;nyear = varargin{1};
else;nyear = 1000;
end
nx              = length(x0);
[out1 out2]     = size(tau);
ntau            = length(tau);
tau             = reshape(tau,1,ntau);
x0              = reshape(x0,nx,1);
K0              = eReshape(K0,nx,1);
K1              = eReshape(K1,nx,nx);
H0              = eReshape(H0,nx,nx);
H1              = eReshape(H1,nx,nx,nx);
R0              = eReshape(R0,1,1);
R1              = eReshape(R1,nx,1);

% check size of jumps and reshape accordingly
nJ              = max([1 length(L0) size(L1,2)]);
L0              = eReshape(L0,1,nJ);
L1              = eReshape(L1,nx,nJ);

%dt              = tau./ceil(tau*nyear);
dt              = tau./200;
dtB             = repmat(dt,nx,1);
beta            = [zeros(nx,ntau)];
alpha           = zeros(1,ntau);

if isa(jump,'function_handle')
    tempX   = rand(nx,2*nx);
    try tempY = jump(tempX);catch err;end
    if ~exist('tempY');tempY=[];end
    if exist('err') | any(size(tempY)~=[nJ 2*nx])
       % uh oh, things will become PRETTY slow from here!
       warning(['Flawed user supplied jump transform detected.' ...
                ' Should accept an input of size (NX)x(K), where' ...
                ' the order of inputs corresponds to the ordering' ...
                ' of the process variables. Output should be (1)x(K).' ...
                char(10) char(10) ...
                'Transform may take longer than usual.']);
       jump = @(u) arrayfun(@(k) jump(u(:,k)),1:size(u,2));
    end
else
    jump            = @(u) ones(1,size(u,2));
end

ricatti         = @(beta) cf_ricatti(beta,K0,K1,H0,H1,L0,L1,jump,R0,R1);
time            = tau;

for k = 1:(max(tau./dt))
    time            = time - dt;
    timeFilterA     = time>=-dt;
    timeFilterB     = repmat(timeFilterA,nx+1,1);
    KK1             = ricatti(beta).*timeFilterB;
    beta1           = beta + dtB/2.*KK1(2:end,:);
    KK2             = ricatti(beta1).*timeFilterB;
    beta2           = beta + dtB/2.*KK2(2:end,:);
    KK3             = ricatti(beta2).*timeFilterB;
    beta3           = beta + dtB.*KK3(2:end,:);
    KK4             = ricatti(beta3).*timeFilterB;
    alpha           = alpha+dt/6.*(KK1(1,:)+2*KK2(1,:)+2*KK3(1,:)+KK4(1,:));
    beta            = beta+dtB/6.*(KK1(2:end,:)+2*KK2(2:end,:)+2*KK3(2:end,:)+KK4(2:end,:));
end, clear KK* beta1 beta2 beta3
P               = exp(alpha.' + beta.'*[x0]);
P               = reshape(P,out1,out2);
Y               = -log(P)./reshape(tau,size(P));
end

function out = cf_ricatti(beta,K0,K1,H0,H1,L0,L1,jump,R0,R1)
%Ricatti equation from Duffie, Singleton, Pan (2000)
nB              = size(beta,1);
nu              = size(beta,2);
out(1,:)        = K0'*beta + 1/2*sum((beta.'*H0).*beta.',2).'+L0*(jump(beta)-1)-R0*ones(1,nu);
for k=1:nB
    temp(k,:)   = sum((beta.'*H1(:,:,k)).*beta.',2).';
end
out(2:nB+1,:)   = K1'*beta + 1/2*temp+L1*(jump(beta)-1)-repmat(R1,1,nu);
end

function out = eReshape(in,r,c,d)
%ERESHAPE transforms (empty) input to (zero) array of desired dimension
if ~exist('d');d=1;end
if isempty(in)
    out = zeros(r,c,d);
else
    out = reshape(in,r,c,d);
end
end
