function [out lOut] = cflib(u,tau,par,type)
%CFLIB library of commonly used characteristic functions in finance
%
%   CFLIB(U,TAU,PAR,TYPE) 
%   yields evaluations of the characteristic function of a model specified
%   in the string TYPE at points in U, given parameter structure PAR and
%   time to maturity TAU. The resulting CF is already discounted. If TAU is
%   a vector, CFLIB returns an array of dimensions |U|x|Tau|.
%
%   For real argument u, CFLIB(u,..) is the characteristic function of X.
%   For complex argument u=-v.*i, CFLIB(u) yields the moment generating
%   function of X.
%
%   Supported Models (parameters below): Black Scholes, BS with Merton
%   style jumps, Heston's Stochastic Volatility, Heston with Merton jumps
%
%   'BS': Black-Scholes  option pricing
%   rf      constant annualized risk free rate
%   x0      log of spot asset price
%   sigma   spot volatility
%
%   'BSJump': Black Scholes with lognormal jumps in the spot process
%   rf      constant annualized risk free rate
%   x0      log of spot asset price
%   sigma   diffusive volatility of spot process
%   muJ     expected jump size
%   sigmaJ  jump volatility
%   lambda  jump intensity
%
% 'Heston':	Heston's stochastic volatility model
%   rf      constant annualized risk free rate
%   x0      log of spot asset price
%   v0      spot variance level
%   kappa   mean reversion speed
%   theta   mean interest rate level
%   sigma   volatility of interest rate
%   rho     correlation between innovations in variance and spot process
%
%   'HestonJump': Heston with jumps. Requires 'Heston' and the following:
%   muJ     expected jump size
%   sigmaJ  jump volatility
%   lambda  jump intensity
%
%   'Kou': Kou's model with asymmetric double exponential jump distribution
%   rf      constant annualized risk free rate
%   x0      log of spot asset price
%   sigma   diffusive volatility of spot process
%   lambda  jump intensity
%   pUp     probability of upward jump 
%   mUp     mean upward jump (set 0 < mUp < 1 for finite expectation)
%   mDown   mean downward jump
%
%
%   Example: Black Scholes
%
%   par     = struct('x0', log(100), 'rf', 0.05, 'sigma', 0.25);
%   cflib(0,1,par,'BS') % the 1-year discount factor 
%   cflib(-i,1,par,'BS') % the 1-year risk neutral expectation of spot

%   Author:   matthias.held@web.de
%   Date:     2014-06-04


nt = length(tau);
nu = length(u);
if isrow(u);u = u.';end
Tau = tau;

lOut            = zeros(nu,nt);

if strcmp(type,'BS')
    rf          = par.rf;
    sigma       = par.sigma;
    x0          = par.x0;
    for t = 1:nt
    tau = Tau(t);
    lOut(:,t)       = -rf*tau + u.*i.*x0 + u.*i.*tau.*(rf-1/2*sigma^2) ...
                        - 1/2*u.^2.*sigma.^2*tau;
    end
elseif strcmp(type,'BSJump')
    rf              = par.rf;
    sigma           = par.sigma;
    muJ             = par.muJ;
    sigmaJ          = par.sigmaJ;
    lambda          = par.lambda;
    x0              = par.x0;
    m               = exp(muJ+1/2*sigmaJ^2)-1;
    for t = 1:nt
    tau = Tau(t);
    alpha           = -rf*tau + u.*i.*tau.*(rf-1/2*sigma^2-lambda*m) ...
                      - 1/2*u.^2*sigma^2*tau ...
                      + lambda*tau*(exp(muJ.*u.*i-1/2*u.^2*sigmaJ^2)-1);
    lOut(:,t)       = alpha + u.*i.*x0;
    end
    
elseif strcmp(type,'Heston') || strcmp(type,'HestonJump')
    kappa           = par.kappa;
    theta           = par.theta;
    sigma           = par.sigma;
    rho             = par.rho;
    rf              = par.rf;
    x0              = par.x0;
    v0              = par.v0;
    lambda          = 0;
    muJ             = 0;
    sigmaJ          = 0;
    if strcmp(type,'HestonJump')
        lambda          = par.lambda;
        muJ             = par.muJ;
        sigmaJ          = par.sigmaJ;
    end
    a               = -1/2.*u.*i.*(1-u.*i);
    b               = rho.*sigma.*u.*i - kappa;
    c               = 1/2.*sigma^2;
    m               = -sqrt(b.^2 - 4.*a.*c);
    for t = 1:nt
    tau = Tau(t);
    emt             = exp(m.*tau);
    beta2           = (m-b)./sigma^2 .* (1-emt)./(1-(b-m)./(b+m).*emt);
    alpha           = -rf*tau ...
                    + (rf-lambda*(exp(muJ+1/2*sigmaJ^2)-1)).*u.*i.*tau ...
                    + kappa.*theta.*(m-b)./sigma.^2.*tau ...
                    + kappa.*theta./c.*log( (2.*m) ./ ((m-b).*emt+b+m));
    alpha           = alpha+tau*lambda*(exp(u.*i.*muJ - 1/2.*u.^2.*sigmaJ^2)-1);
    lOut(:,t)       = alpha + u.*i.*x0 + beta2.*v0;
    end
    
elseif strcmp(type,'Kou')
    rf              = par.rf;
    sigma           = par.sigma;
    x0              = par.x0;
    lambda          = par.lambda;
    pUp             = par.pUp;
    mDown           = 1/par.mDown;
    mUp             = 1/par.mUp;
    comp            = @(x) pUp*mUp./(mUp-x) + (1-pUp)*mDown./(mDown+x)-1;
    for t = 1:nt
    tau = Tau(t);
    alpha           = -rf*tau + u*i*(rf-1/2*sigma^2-lambda*comp(1)) ...
                        -1/2*u.^2*sigma^2 + lambda*comp(u*i);
	lOut(:,t)       = alpha + u*i*x0;
    end
end 

out             = exp(lOut);

end


