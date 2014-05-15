function [out lOut] = cflib(u,tau,par,type)
%CFLIB library of commonly used characteristic functions in finance
%
%   CFLIB(U,TAU,PAR,TYPE) 
%   yields evaluations of the characteristic function of a model specified
%   in the string TYPE at points in U, given parameter structure PAR and
%   time to maturity TAU. The resulting CF is discounted. 
%
%   For real argument u, CFLIB(u,..) is the characteristic function of X.
%   For complex argument u=-v.*i, CFLIB(u) yields the moment generating
%   function of X.
%
%   Supported Models (parameters below): Black Scholes, BS with Merton
%   style jumps, Heston's Stochastic Volatility, Heston with Merton jumps
%
% 'BS': Black-Scholes  option pricing
% rf        constant annualized risk free rate
% x0        log of spot asset price
% sigma     spot volatility
%
% 'BSJump': Black Scholes with lognormal jumps in the spot process
% rf        constant annualized risk free rate
% x0        log of spot asset price
% sigma     spot volatility
% muJ       expected jump size
% sigmaJ    jump volatility
% lambda    jump intensity
%
% 'Heston':	Heston's stochastic volatility model
% rf        constant annualized risk free rate
% x0        log of spot asset price
% v0        spot variance level
% kappa     mean reversion speed
% theta     mean interest rate level
% sigma     volatility of interest rate
% rho       correlation between innovations in variance and spot process
%
% 'HestonJump': Heston with jumps. Requires 'Heston' and the following:
% muJ       expected jump size
% sigmaJ    jump volatility
% lambda    jump intensity
%
% Example:
% tau       = 1;
% par       = struct('x0', log(100), 'rf', 0.05, 'sigma', 0.25);
% cflib(0,tau,par,'BS') % the discount factor 
% cflib(1,tau,par,'BS') % the risk neutral expectation of the spot asset
% cflib(1*i,tau,par,'BS') % the CF of X, evaluated at u=1*i

% Author:   matthias.held@web.de
% Date:     2014-04-30

u = u.*i;

if strcmp(type,'BS')
    rf          = par.rf;
    sigma       = par.sigma;
    x0          = par.x0;
    lOut        = -rf*tau + u.*x0 + u.*tau.*(rf-1/2*sigma^2) ...
                    + 1/2*u.^2.*sigma.^2*tau;

elseif strcmp(type,'BSJump')
    rf              = par.rf;
    sigma           = par.sigma;
    muJ             = par.muJ;
    sigmaJ          = par.sigmaJ;
    lambda          = par.lambda;
    x0              = par.x0;
    m               = exp(muJ+1/2*sigmaJ^2)-1;
    alpha           = -rf*tau + u.*tau.*(rf-1/2*sigma^2-lambda*m) ...
                      + 1/2*u.^2*sigma^2*tau ...
                      + lambda*tau*(exp(muJ.*u+1/2*u.^2*sigmaJ^2)-1);
    lOut            = alpha + u.*x0;

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
    a               = -1/2.*u.*(1-u);
    b               = rho.*sigma.*u - kappa;
    c               = 1/2.*sigma^2;
    m               = -sqrt(b.^2 - 4.*a.*c);
    emt             = exp(m.*tau);
    beta2           = (m-b)./sigma^2 .* (1-emt)./(1-(b-m)./(b+m).*emt);
    alpha           = -rf*tau + (rf-lambda*(exp(muJ+1/2*sigmaJ^2)-1)).*u.*tau ...
                    + kappa.*theta.*(m-b)./sigma.^2.*tau ...
                    + kappa.*theta./c.*log( (2.*m) ./ ((m-b).*emt+b+m));
    alpha           = alpha+tau*lambda*(exp(u.*muJ + 1/2.*u.^2.*sigmaJ^2)-1);
    lOut            = alpha + u.*x0 + beta2.*v0;
end 
out             = exp(lOut);



