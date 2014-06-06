function [out A B alpha beta] = cfaffineEx(u,v,x0,tau,K0,K1,H0,H1,R0,R1,L0,L1,jump,gradJump,ND,varargin)
%CFAFFINEEX extended characteristic function  of AJD process
%
%   W = CFAFFINEEX(U,V,X0,TAU,K0,K1,H0,H1,R0,R1) 
%   Returns the ecf when there are only diffusive risks present.
%   You may leave any unused parameter empty [].
%
%   W = CFAFFINEEX(U,V,X0,TAU,K0,K1,H0,H1,R0,R1,L0,L1,jump,jumpGrad)
%   Returns the ecf when there are diffusive risks and jumps present.
%   You may leave any unused parameter empty [].
%
%   W = CFAFFINE(U,V,X0,TAU,K0,K1,H0,H1,R0,R1,L0,L1,jump,gradJump,ND)
%   If U should be applied column-wise set ND=1; for a row-wise application
%   of U set ND=2. Type 'doc cfaffineEx' for more information.
%
%   The underlying N-dimensinoal process of X is affine jump-diffusive:
%   dX=M(X)*dt+S(X)*dW+J*dZ, where
%   M(X)        = K0 + K1*X,    process drift (vector/matrix)
%   S(X)*S(X)'  = H0 + H1*X,    process diffusive variance (matrix/tensor)
%   L(X)        = L0 + L1'*X,   jump intensity (scalar/vector)
%   R(X)        = R0 + R1'*X,   interest rate process (scalar/vector)
%   jump(c)     = E(exp(c*J)),  moment generating function of the 
%                               multivariate jump distribution
%   gradJump(c) = d j(c)/ dc    gradient of the jump transform function
% 
%   The |Nx1| vector x0 contains the initial state, the first entry being 
%   the variable whose characteristic function is required. If ND is used, 
%   the setup of x0 has to correspond with U.
%
%   The jump transform is a user supplied function that expects an array of
%   dimension (dim(X)xK) and returns an array of dimension (1xK). The first
%   input corresponds to the first variable, the second input relates to 
%   the second process variable, etc. 
%
%   The gradient of the jump transform is a user supplied function that an 
%   array of dimension (dim(X)xK) and returns an array of the same dimension.
%
%   Example: Undiscounted expectation of average CIR short rate over 5 yrs:
%
%   r0          = 0.08;
%   y0          = 0;
%   x0          = [r0 y0]';
%   kR          = 2.50;
%   tR          = 0.04;
%   sR          = 0.10;
%   tau         = 5;
%   K0          = [kR*tR 0]';
%   K1          = [-kR 0 ; 1 0];
%   H1          = zeros(2,2,2); 
%   H1(1,1,1)   = sR^2;
%   v           = [0 1]';
%   avg         = 1/tau*cfaffineEx([0 0]',v,x0,tau,K0,K1,[],H1,[],[],[],[],[],[],1)

%   Author : matthias.held@web.de 
%   Date : 2014 06 06


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

B               = repmat(reshape(v,nx,1),1,nu);
A               = zeros(1,nu);
alpha           = zeros(1,nu);

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






% check whether jump exists as a function handle, and check whether its
% output is of desired dimension. e

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

ricatti         = @(beta) cf_ricatti(beta,K0,K1,H0,H1,L0,L1,jump,R0,R1);
ricatti         = @(beta,B) cf_ricatti(beta,B,K0,K1,H0,H1,L0,L1,jump,gradJump,R0,R1);

for k = tau-dt:-dt:0
    KK1             = ricatti(beta,B);
    beta1           = beta + dt/2*KK1(2:nx+1,:);
    B1              = B + dt/2.*KK1(nx+3:2*nx+2,:);
    KK2             = ricatti(beta1,B1);
    beta2           = beta + dt/2*KK2(2:nx+1,:);
    B2              = B + dt/2.*KK2(nx+3:2*nx+2,:);
    KK3             = ricatti(beta2,B2);
    beta3           = beta + dt.*KK3(2:nx+1,:);
    B3              = B + dt.*KK3(nx+3:2*nx+2,:);   
    KK4             = ricatti(beta3,B3);
   
    
    alpha           = alpha+dt/6*(KK1(1,:)+2*KK2(1,:)+2*KK3(1,:)+KK4(1,:));
    beta            = beta+dt/6*(KK1(2:nx+1,:)+2*KK2(2:nx+1,:)+2*KK3(2:nx+1,:)+KK4(2:nx+1,:));
    A               = A + dt/6*(KK1(nx+2,:)+2*KK2(nx+2,:) + 2*KK3(nx+2,:) + KK4(nx+2,:));
    B               = B + dt/6*(KK1(nx+3:2*nx+2,:)+2*KK2(nx+3:2*nx+2,:)+2*KK3(nx+3:2*nx+2,:)+KK4(nx+3:2*nx+2,:));
    
    
end, clear KK* beta1 beta2 beta3
out             = (A.'+B.'*x0).*exp(alpha.' + beta.'*[x0]);
out             = reshape(out,out1,out2);
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
    out = in; %    out = reshape(in,r,c,d);
end
end
