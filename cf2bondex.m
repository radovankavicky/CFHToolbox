function [out A B a b] = cf2bondex(A,B,tau,x0,K0,K1,H0,H1,R0,R1,L0,L1,jump,gradJump,varargin)
%CF2BONDEX extended transform 
%
%   [P] = CF2BONDEX(A,B,TAU,X0,K0,K1,H0,H1,R0,R1,L0,L1,JUMP,GRADJUMP) 
%   Returns the discounted risk neutral expectation of a combination of 
%   state variables, P=E[(A+B*X)*(df)], where df is the implied discount 
%   factor for maturity TAU. Leave any component not required in your model 
%   empty []. 
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
%   The vector x0 contains the |(NX)x(1)| initial state.
%
%   In this version, only one jump component is supported in CF2BONDEX.
%   JUMP is the moment generating function of the jump distribution,
%   expecting input of dimension |(NX)x(K)| and returning |(1)x(K)|. 
%   GRADJUMP returns the gradient of the moment generating function,
%   expecting an input |(NX)x(K)| and returning |(NX)x(K)|. If your model
%   has a JUMP component, you also have to specificy GRADJUMP.
%
%   % Example: CDS pricing. Vasicek zero rate and CIR default intensity
%   % can jump itself (exponentially) with constant intensity.What is the 
%   % corresponding CDS spread for different times to maturity?
%   r0          = 0.05;
%   kappaR      = 0.07;
%   thetaR      = 0.04;
%   sigmaR      = 0.05;
%   lambda0     = 0.08;
%   kappaL      = 0.45;
%   thetaL      = 0.05;
%   sigmaL      = 0.20;
%   intensity   = 0.20;
%   muJ         = 0.10;
%   jump        = @(c) 1./(1-c(2,:)*muJ);
%   gradJump    = @(c) [zeros(1,size(c,2));muJ./(1-c(2,:)*muJ).^2 ];
%   x0          = [r0 ; lambda0];
%   K0          = [kappaR*thetaR kappaL*thetaL]';
%   K1          = [-kappaR 0 ; 0 -kappaL];
%   H0          = [sigmaR^2 0 ; 0 0];
%   H1          = zeros(2,2,2);
%   H1(2,2,2)   = sigmaL^2;
%   R1          = [1 1]';
%   L0          = intensity          
%   recovery    = 0.4;
%   dt          = 0.001;
%   tau         = [0:dt:5];
%   protection  = cumsum(cf2bondEx(0,[0;1],tau,x0,K0,K1,H0,H1,[],R1,L0,[],jump,gradJump))*dt;
%   premium     = cumsum(cf2bond(tau,x0,K0,K1,H0,H1,[],R1,L0,[],jump))*dt;
%   spread      = (1-recovery)*protection./premium;
%   plot(tau,spread);title('risk-neutral CDS spread');

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
if ~exist('gradJump');gradJump=[];end

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
L0              = eReshape(L0,1,nJ) ;
L1              = eReshape(L1,nx,nJ);

%dt              = tau./ceil(tau*nyear);
dt              = tau./200;
dtB             = repmat(dt,nx,1);
beta            = [zeros(nx,ntau)];
alpha           = zeros(1,ntau);

B               = repmat(B,1,ntau);
A               = repmat(A,1,ntau);

if isa(jump,'function_handle')
    if ~isa(gradJump,'function_handle')
        error('jump gradient missing');
    end
    
    tempX           = rand(nx,2*nx);
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

if isa(gradJump,'function_handle')
    tempX   = rand(nx,2*nx);
    try tempY = gradJump(tempX);catch err;end
    if ~exist('tempY');tempY=[];end
    if exist('err') | any(size(tempY)~=[nJ*nx 2*nx])
       % uh oh, things will become PRETTY slow from here!
       warning(['Flawed user supplied jump transform detected.' ...
                ' Should accept an input of size (NX)x(K), where' ...
                ' the order of inputs corresponds to the ordering' ...
                ' of the process variables. Output should be (1)x(K).' ...
                char(10) char(10) ...
                'Transform may take longer than usual.']);
       gradJump = @(u) arrayfun(@(k) gradJump(u(:,k)),1:size(u,2));
    end
else
    gradJump            = @(u) zeros(nx,size(u,2));
end


ricatti         = @(beta,B) cf_ricatti(beta,B,K0,K1,H0,H1,L0,L1,jump,gradJump,R0,R1);
time            = tau;

for k = 1:(max(tau./dt))
    time            = time - dt;
    timeFilterA     = time>=-dt;
    timeFilterB     = repmat(timeFilterA,2*(nx+1),1);
    
    KK1             = ricatti(beta,B).*timeFilterB;
    beta1           = beta + dtB/2.*KK1(2:nx+1,:);
    B1              = B + dtB/2.*KK1(nx+3:2*nx+2,:);
    
    
    KK2             = ricatti(beta1,B1).*timeFilterB;
    beta2           = beta + dtB/2.*KK2(2:nx+1,:);
    B2              = B + dtB/2.*KK2(nx+3:2*nx+2,:);
    
    KK3             = ricatti(beta2,B2).*timeFilterB;
    beta3           = beta + dtB.*KK3(2:nx+1,:);
    B3              = B + dtB.*KK3(nx+3:2*nx+2,:);   
    
    KK4             = ricatti(beta3,B3).*timeFilterB;
    
    alpha           = alpha+dt/6.*(KK1(1,:)+2*KK2(1,:)+2*KK3(1,:)+KK4(1,:));
    beta            = beta+dtB/6.*(KK1(2:nx+1,:)+2*KK2(2:nx+1,:)+2*KK3(2:nx+1,:)+KK4(2:nx+1,:));
    A               = A + dt/6.*(KK1(nx+2,:)+2*KK2(nx+2,:) + 2*KK3(nx+2,:) + KK4(nx+2,:));
    B               = B + dtB/6.*(KK1(nx+3:2*nx+2,:)+2*KK2(nx+3:2*nx+2,:)+2*KK3(nx+3:2*nx+2,:)+KK4(nx+3:2*nx+2,:));
end, clear KK* beta1 beta2 beta3
out             = (A + x0.'*B).*exp(alpha + x0.'*beta);
a = alpha;
b = beta;
end

%Corresponding ricatti equation from the Duffie, Singleton, Pan (2000).
function [out] = cf_ricatti(beta,B,K0,K1,H0,H1,L0,L1,jump,gradJump,R0,R1)
% this vector ricatti iterates on the system [alpha ; beta ; A ; B] where 
% alpha and A are 1x1 and beta,B are (nx)x1. The first element in
% outExp is alpha, the first element in outAbs is A. 
%
nx              = size(beta,1);
nu              = size(beta,2);
j1              = jump(beta)-1;
j2              = sum(gradJump(beta).*B);
% operate on alpha and beta
out(1,:)        = K0'*beta + 1/2*sum((beta.'*H0).*beta.',2).'+L0*(j1)-R0*ones(1,nu);
out(nx+2,:)     = K0'*B + sum((beta.'*H0).*B.',2).'+L0*(j2);
for k=1:nx
    temp(k,:)   = sum((beta.'*H1(:,:,k)).*beta.',2).';
    temp2(k,:)  = sum((beta.'*H1(:,:,k)).*B.',2).';
end
out(2:nx+1,:)   = K1'*beta + 1/2*temp+L1*(j1)-repmat(R1,1,nu);
out(nx+3:2*nx+2,:)   = K1'*B + temp2 + L1*(j2);
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
