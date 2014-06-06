function out = cfjump(c,par,type)
%CFJUMP library of commonly encountered jump transforms
%
%   CFJUMP(C,PAR,TYPE) 
%   yields evaluations at C of the moment generating function of a jump
%   distribution specified in TYPE using parameter structure PAR. If TYPE
%   is appended by GRAD, CFJUMP returns the gradient of the jump transform.
%
%   The input C is supposed to be of dimension (NX)x(K), where NX denotes
%   the number of process variables. If, for example, your model contains
%   stochastic volatility and jumps only in the underlying itself, NX=2 and
%   the parameters of the jump distribution have to be set such that there
%   is an infinite point mass at zero variance jumps. The output
%   corresponds to the number of jump components: (NJ)x(K).
%
%   Supported jump specifications: Merton type jumps, exponential jumps
%
%   'Merton': Multivariate normally distributed jumps in the log spot
%   process. 
%   par.MuJ     (NX)x(1) vector of mean jump sizes
%   par.SigmaJ  (NX)x(NX) matrix of jump covariances
% 
%   'Exponential': Exponentially distributed jumps on the positive axis
%   par.MuJ     (NX)x(1) vector of mean jump sizes
%
%   'DoubleExponential': Double exponentially distributed jumps over reals
%   pUp         (NX)x(1) vector of upward jump probability
%   mUp         (NX)x(1) vector of mean upward jump sizes (0<mUp<1)
%   mDown       (NX)x(1) vector of mean downward jump
%
%   Example: Gradient of Merton type jumps at different points c:
%   par.MuJ     = [0.1 0 -0.3]';
%   par.SigmaJ  = [0.1 0 0.05 ; 0 0 0 ; 0.05 0 0.2];
%   c           = [zeros(3,1) eye(3) rand(3,4)];
%   cfjump(c,par,'MertonGrad')
%

%   Author:     matthias.held@web.de
%   Date:       2015-06-06

[nx nu]     = size(c);

if strcmp(type,'Merton')
    MuJ             = par.MuJ;
    SigmaJ          = par.SigmaJ;
    [~,~,nJ]        = size(SigmaJ);
    for k = 1:nJ
        % LDL decomposition yields (sqrt(D)*L')'*(sqrt(D)*L') = SigmaJ
        [L D]           = ldl(SigmaJ(:,:,k));
        out(k,:)        = exp(MuJ(:,k)'*c + 1/2*sum(((sqrt(D)*L')*c).^2,1));
    end
    
elseif strcmp(type,'MertonGrad')
    MuJ             = par.MuJ;
    SigmaJ          = par.SigmaJ;
    [~,~,nJ]        = size(SigmaJ);
    MuJ             = repmat(MuJ,1,nu);
    out             = (MuJ+SigmaJ*c).*repmat(cfjump(c,par,'Merton'),nx,1);
    
elseif strcmp(type,'Exponential')
    MuJ             = par.MuJ;
    out             = ones(1,nu);
    for k = 1:nx
        out             = out.*(1./(1-MuJ(k)*c(k,:)));
    end
    
elseif strcmp(type,'ExponentialGrad')
    MuJ             = par.MuJ;
    out             = zeros(nx,nu);
    for k = 1:nx
        out(k,:)        = MuJ(k)./(1-MuJ(k)*c(k,:));
    end
    out             = out.*repmat(cfjump(c,par,'Exponential'),nx,1);

elseif strcmp(type,'DoubleExponential')
    pUp             = par.pUp;
    mUp             = par.mUp;
    mDown           = par.mDown;
    out             = ones(1,nu);
    for k = 1:nx
        out             = out.*(pUp(k)*1./(1-mUp(k)*c(k,:))+(1-pUp(k))*1./(1+mDown(k)*c(k,:)));
    end
    
elseif strcmp(type,'DoubleExponentialGrad')
    pUp             = par.pUp;
    mUp             = par.mUp;
    mDown           = par.mDown;
    out             = ones(nx,nu);
    for k = 1:nx
        tpar            = par;
        tpar.mUp(k)     = 0;
        tpar.mDown(k)   = 0;
        out(k,:)        = cfjump(c,tpar,'DoubleExponential');
        out(k,:)        = out(k,:).*(pUp(k)*mUp(k)./(1-c(k,:).*mUp(k)).^2 ...
                        - (1-pUp(k))*mDown(k)./(1+c(k,:).*mDown(k)).^2);
    end
end

end
    