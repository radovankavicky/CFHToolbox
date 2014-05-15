function out = cfjump(c,par,type)
%CFJUMP library of commonly encountered jump transforms
%
%   CFJUMP(C,PAR,TYPE) 
%   yields evaluations at C of the moment generating function of a jump
%   distribution specified in TYPE using parameter structure PAR.
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
%   par.MuJ     Contains a (NX)x(1) vector of mean jump sizes
%   par.SigmaJ  Contains a (NX)x(NX) matrix of jump covariances
% 
%   'Exponential': Exponentially istributed jumps on the positive subspace.
%   par.MuJ     Contains a (NX)x(1) vector of mean jump sizes

%   Author:     matthias.held@web.de
%   Date:       2015-05-02

[nx nu]     = size(c);
if strcmp(type,'Merton')
    MuJ             = par.MuJ;
    SigmaJ          = par.SigmaJ;
    [~,~,nJ]        = size(SigmaJ);
        
    for k = 1:nJ
        % use ldl(SigmaJ) instead f
        [L D]           = ldl(SigmaJ(:,:,k));
        out(k,:)        = exp(MuJ(:,k)'*c + 1/2*sum(((sqrt(D)*L')*c).^2,1));
    end
elseif strcmp(type,'Exponential')
    MuJ             = par.MuJ;
    out             = 1./(1-MuJ'*c);
end
 
end
    