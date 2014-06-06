function [out alpha beta] = cfaffine(u,x0,tau,K0,K1,H0,H1,R0,R1,L0,L1,jump,ND,varargin)
%CFAFFINE characteristic function (cf) of an affine jump-diffusion process
%
%   W = CFAFFINE(U,X0,TAU,K0,K1,H0,H1,R0,R1) 
%   Returns the cf when there are only diffusive risks present.
%   You may leave any unused parameter empty [].
%
%   W = CFAFFINE(U,X0,TAU,K0,K1,H0,H1,R0,R1,L0,L1,jump)
%   Returns the cf when there are diffusive risks and jumps present.
%   You may leave any unused parameter empty [].
%
%   For real argument u, CFAFFINE(u,..) returns the characteristic function 
%   of the stochastic process. For complex arugment u=-v*i, CFAFFINE(u,..) 
%   returns the moment generating function of the stochastic process. 
%
%   W = CFAFFINE(U,X0,TAU,K0,K1,H0,H1,R0,R1,L0,L1,jump,ND)
%   If U should be applied column-wise set ND=1; for a row-wise application
%   of U set ND=2. Type 'doc cfaffine' for more information.
%
%   The underlying N-dimensinoal process of X is affine jump-diffusive:
%   dX=M(X)*dt+S(X)*dW+J*dZ, where
%   M(X)        = K0 + K1*X,    process drift (vector/matrix)
%   S(X)*S(X)'  = H0 + H1*X,    process diffusive variance (matrix/tensor)
%   L(X)        = L0 + L1'*X,   jump intensity (scalar/vector)
%   R(X)        = R0 + R1'*X,   interest rate process (scalar/vector)
%   jump(c)     = E(exp(c*J)),  moment generating function of the 
%                               multivariate jump distribution
%
%   The |Nx1| vector x0 contains the initial state, the first entry being 
%   the variable whose characteristic function is required. If ND is used, 
%   the setup of x0 has to correspond with U.
%
%   The jump transform is a user supplied function that expects an array of
%   dimension |NxK| and returns an array of dimension |1xK|. The first
%   input corresponds to the first variable, the second input relates to 
%   the second process variable, etc. 
%
%   Example: Black-Scholes world with constant interest rates:
%
%   x0      = log(1);
%   tau     = 1;
%   sigma   = 0.25;
%   rf      = 0.05;
%   cf      = @(u) cfaffine(u,x0,tau,rf-1/2*sigma^2,[],sigma^2,[],rf)
%   cf(0)   % the discount factor
%   [f x]   = cf2pdf(cf);
%   plot(x,f)

%   Author : matthias.held@web.de 
%   Date : 2014 05 28

if ~exist('K0');K0=[];end
if ~exist('K1');K1=[];end
if ~exist('H0');H0=[];end
if ~exist('H1');H1=[];end
if ~exist('R0');R0=[];end
if ~exist('R1');R1=[];end
if ~exist('L0');L0=[];end
if ~exist('L1');L1=[];end
if ~exist('jump');jump=[];end
if ~exist('ND');ND=[];end
if ~isempty(varargin) && length(varargin) == 1;nyear = varargin{1};
else;nyear = 200;
end
nx              = length(x0);
[out1 out2]     = size(u);
if ND==1 % u is (Nx)*(Nu)
    nu              = out2;
    u               = u*i;
    beta            = u;
    out1            = 1;
elseif ND==2 % u is (Nu)*(Nx)
    nu              = out1;
    u               = u.'*i;
    beta            = u;
    out2            = 1;
else % u is (out1)*(out2)
    nu              = length(u);
    u               = reshape(u,1,nu)*i;
    beta            = [u ; zeros(nx-1,nu)];
end

x0              = reshape(x0,nx,1);
K0              = eReshape(K0,nx,1);
K1              = eReshape(K1,nx,nx);
H0              = eReshape(H0,nx,nx);
H1              = eReshape(H1,nx,nx,nx);
R0              = eReshape(R0,1,1);
R1              = eReshape(R1,nx,1);
dt              = tau/(ceil(tau*nyear));
% check size of jumps and reshape accordingly
nJ              = max([1 length(L0) size(L1,2)]);
L0              = eReshape(L0,1,nJ);
L1              = eReshape(L1,nx,nJ);


alpha           = zeros(1,nu);
% check whether jump exists as a function handle, and check whether its
% output is of desired dimension. e
if isa(jump,'function_handle')
    tempX   = rand(nx,2*nx);
    try tempY = jump(tempX);catch err;end
    if ~exist('tempY');tempY=[];end
    if exist('err') | any(size(tempY)~=[nJ 2*nx])
       % uh oh, things will become PRETTY slow from here!
       warning(['Possible dimensino mismatch in user supplied jump' ...
              ' transform detected.' ...
              ' JUMP(C) Should accept an input of size (NX)x(K), where' ...
              ' the order of inputs corresponds to the ordering' ...
              ' of the process variables. Output should be (NJ)x(K).' ...
              char(10) char(10) ...
              'Transform may take longer than usual.']);
       jump = @(u) arrayfun(@(k) jump(u(:,k)),1:size(u,2));
    end
else
    jump            = @(u) ones(1,size(u,2));
end

ricatti         = @(beta) cf_ricatti(beta,K0,K1,H0,H1,L0,L1,jump,R0,R1);
for k = tau-dt:-dt:0
    KK1             = ricatti(beta);
    beta1           = beta + dt/2*KK1(2:end,:);
    KK2             = ricatti(beta1);
    beta2           = beta + dt/2*KK2(2:end,:);
    KK3             = ricatti(beta2);
    beta3           = beta + dt*KK3(2:end,:);
    KK4             = ricatti(beta3);
    alpha           = alpha+dt/6*(KK1(1,:)+2*KK2(1,:)+2*KK3(1,:)+KK4(1,:));
    beta            = beta+dt/6*(KK1(2:end,:)+2*KK2(2:end,:)+2*KK3(2:end,:)+KK4(2:end,:));
end, clear KK* beta1 beta2 beta3
out             = exp(alpha.' + beta.'*[x0]);
out             = reshape(out,out1,out2);
end


function out = cf_ricatti(beta,K0,K1,H0,H1,L0,L1,jump,R0,R1)
%cf_ricatti ricatti equation from Duffie, Singleton, Pan (2000)
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
    out = in; %    out = reshape(in,r,c,d);
end
end
