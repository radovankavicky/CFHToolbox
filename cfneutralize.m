function [K0 K1] = cfneutralize(K0,K1,H0,H1,R0,R1,Q0,Q1,L0,L1,jump)
%CFNEUTRALIZE risk neutral drift coefficients for given dynamics
%
%   [K0 K1] = CFNEUTRALIZE(K0,K1,H0,H1,R0,R1,Q0,Q1,L0,L1,jump)
%   yields the risk neutral drift parameters K0 and K1 that correspond to
%   arbitrage free pricing given pre-adjusted drift parameters K0/K1, risk 
%   neutral variance parameters H0/H1, interest rate parameters R0/R1, jump
%   intensity parameters L0/L1 and dividend yield parameters Q0/Q1.
% 
%   The first entries/rows should correspond to the variable whose drift 
%   will be adjusted.
%
%   Example: Heston Model with constant dividend yield q
%
%   mu          = 0.15;
%   q           = 0.04;
%   rf          = 0.05;
%   kappa       = 0.55;
%   theta       = 0.04;
%   sigma       = 0.10;
%   rho         = -0.7;
%   K0          = [mu ; kappa*theta];
%   K1          = [0 0 ; 0 -kappa];
%   H1(:,:,1)   = [1 rho*sigma ; rho*sigma sigma^2];
%   R0          = rf;
%   Q0          = q;
%   [K0 K1]     = cfneutralize(K0,K1,[],H1,R0,[],Q0)
       
%   Author:     matthias.held@web.de
%   Date:       2014-05-01

if ~exist('K0');K0=[];end
if ~exist('K1');K1=[];end
if ~exist('H0');H0=[];end
if ~exist('H1');H1=[];end
if ~exist('R0');R0=[];end
if ~exist('R1');R1=[];end
if ~exist('L0');L0=[];end
if ~exist('L1');L1=[];end
if ~exist('Q0');Q0=[];end
if ~exist('Q1');Q1=[];end
if ~exist('jump');jump=[];end

% find the size of the state space
dimVec          = max( size([K0 R1 L1 Q1],1));
dimMat          = max( size([K1 H0],1));
dimTen          = size(H1,1);
nx              = max([dimVec dimMat dimTen]);

K0              = eReshape(K0,nx,1);
K1              = eReshape(K1,nx,nx);
H0              = eReshape(H0,nx,nx);
H1              = eReshape(H1,nx,nx,nx);
R0              = eReshape(R0,1,1);
R1              = eReshape(R1,nx,1);

% number of assets is defined by length of Q0
nA              = max([1 length(Q0) size(Q1,2)]);
Q0              = eReshape(Q0,1,nA);
Q1              = eReshape(Q1,nx,nA);
% check size of jumps and reshape accordingly
nJ              = max([1 length(L0) size(L1,2)]);
L0              = eReshape(L0,1,nJ);
L1              = eReshape(L1,nx,nJ);

if isa(jump,'function_handle')
    tempX = rand(nx,2*nx); try tempY = jump(tempX);catch err;end
    if ~exist('tempY');tempY=[];end
    if exist('err') | any(size(tempY)~=[nJ 2*nx])
        % uh oh, things will become PRETTY slow from here!
        jump = @(u) arrayfun(@(k) jump(u(:,k)),1:size(u,2));
    end
else
    jump            = @(u) ones(1,size(u,2));
end

c0              = [ eye(nA) ; zeros(nx-nA,nA)];
m               = zeros(nx,nJ);
m(1:nA,:)       = jump(c0)'-1;
K0(1:nA,:)      = R0-Q0'-1/2*diag(H0(1:nA,1:nA))-m(1:nA,:)*L0';
    for k = 1:nA
        K1(k,:)         = R1'-Q1(:,k)'-1/2*reshape(H1(k,k,:),nx,1)' ...
                            -m(k,:)*L1';
    end
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
