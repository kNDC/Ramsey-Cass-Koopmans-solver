clear all; close all; clc;

%% Parameters
% H/h preferences
rho = 0.04;
theta = 1;

% Technology
nu = 1/3;
gamma = 1;
delta = 0.05;

% Growth rates
g = 0.01;
n = 0.01;

% Shocks
uK = 1;
uL = 1;

%% Functions
syms ks;
if (gamma == 1) fs = uK*uL*ks.^nu;
else fs = (nu*(uK*ks).^(1-1/gamma) + (1-nu)*uL.^(1-1/gamma)).^(gamma/(gamma-1));
end

f_p1s = diff(fs, ks);
f_p2s = diff(f_p1s, ks);

f = matlabFunction(fs, "vars", ks);
f_p1 = matlabFunction(f_p1s, "vars", ks);
f_p2 = matlabFunction(f_p2s, "vars", ks);

if (theta == 1) u = @(c) log(c);
else u = @(c) c.^(1-theta)/(1-theta);
end

%% Steady state
% SS for Ramsey, GR for Solow
% Numerical solution using unidimensional Newton-Raphson

fprintf("Solving for the steady state...\n");

if (rho <= n + (1-theta)*g)
    fprintf("The parameterisation admits no solution!\n\n")
    return
end

metric=1; i=0; kSS=1;
while (metric >= eps)
    kSS = kSS - (f_p1(kSS)-delta - rho - theta*g)/f_p2(kSS);
    metric = abs(f_p1(kSS)-delta - rho - theta*g); i=i+1;
end
cSS = f(kSS) - (delta+g+n)*kSS;

metric=1; i=0; kGR=1;
while (metric>=1e-10)
    kGR = kGR - (f_p1(kGR)- (delta+g+n))/f_p2(kGR);
    metric = abs(f_p1(kGR) - (delta+g+n)); i=i+1;
end
cGR = f(kGR) - (delta+g+n)*kGR;

%% Grid, differentiation operators
Ik = 2000; kmin = 1e-5*kSS; kmax = 5*kSS;
k = linspace(kmin, kmax, Ik)';
dk = (kmax-kmin)/Ik;

d1Mx = spdiags(repmat([-1 sparse(1, 1) 1], Ik, 1), -1:1, Ik, Ik)/2;
d1Mx(1, 1)=-1; d1Mx(1, 2)=1; d1Mx(Ik, Ik-1)=-1; d1Mx(Ik, Ik)=1;
d1Mx = d1Mx/dk;

%% Solution - see notes in the solver
[c, V, ~] = RCK_solver(k, rho, theta, nu, gamma, delta, g, n, 0);
s = f(k) - (delta+n+g)*k - c;

%% Extracting lifetime utility/value function V
figure;
plot(k, V, "linewidth", 1.2, "color", [1 0.3 0]);
xlim([0 kmax]); ylim([V(1) V(Ik)]);

xlabel("$\tilde{k}$", "interpreter", "latex");
ylabel("$V\!\left(\tilde{k}\right)$", "interpreter", "latex");
title("Value function~$V\!\left(\tilde{k}\right)$", "interpreter", "latex", "fontsize", 12);

%% Phase diagram
figure;
plot(k, f(k) - (delta+g+n).*k, "linewidth", 1.2, "color", [0.4 0 1]);
xlim([0 kmax]); ylim([0 3*cSS]); hold on;

plot(k, c, "linewidth", 1.2, "color", [1 0.3 0]);
line([kSS kSS], [0 3*cSS], "linewidth", 1.2, "color", [0.4 0 1]);
line([0 kSS], [cSS cSS], "linestyle", "--", "color", [0 0 0]);

xlabel("$\tilde{k}$", "interpreter", "latex");
ylabel("$\tilde{c}$", "interpreter", "latex");
title("Phase Diagram and Saddle Path", "interpreter", "latex", "fontsize", 12);

kgrid = repmat(linspace(0, kmax, 30), 30, 1);
cgrid = (3*cSS/kmax * speye(30)) * kgrid';
dkgrid = f(kgrid) - cgrid - (delta+g+n).*kgrid;
dkgrid(dkgrid==Inf) = NaN;
dcgrid = 1/theta*(f_p1(kgrid) - delta - rho - theta*g).*cgrid;
dcgrid(dcgrid==Inf) = NaN;

quiver(kgrid, cgrid, dkgrid, dcgrid, ...
    "color", [0.4 0.4 0.4], ...
    "AlignVertexCenters", "on", ...
    "MaxHeadSize", 0.1);

% Explosive and sub-optimal paths
streamline(kgrid, cgrid, dkgrid, dcgrid, 0.7*kSS, 0.5*pchip(k, c, 0.7*kSS));
streamline(kgrid, cgrid, dkgrid, dcgrid, 0.7*kSS, 0.8*pchip(k, c, 0.7*kSS));
streamline(kgrid, cgrid, dkgrid, dcgrid, 0.7*kSS, 1.1*pchip(k, c, 0.7*kSS));

scatter(0.7*kSS, 0.5*pchip(k, c, 0.7*kSS), "b");
scatter(0.7*kSS, 0.8*pchip(k, c, 0.7*kSS), "b");
scatter(0.7*kSS, 1.1*pchip(k, c, 0.7*kSS), "b");

streamline(kgrid, cgrid, dkgrid, dcgrid, 3*kSS, 0.5*pchip(k, c, 3*kSS));
streamline(kgrid, cgrid, dkgrid, dcgrid, 3*kSS, 0.8*pchip(k, c, 3*kSS));
streamline(kgrid, cgrid, dkgrid, dcgrid, 3*kSS, 1.1*pchip(k, c, 3*kSS));

scatter(3*kSS, 0.5*pchip(k, c, 3*kSS), "b");
scatter(3*kSS, 0.8*pchip(k, c, 3*kSS), "b");
scatter(3*kSS, 1.1*pchip(k, c, 3*kSS), "b");

hold off;

%% Adjustment paths
params = [rho theta nu delta g uK uL]; params0 = params;
names = ["$\rho$" "$\theta$" "$\nu$" "$\delta$" "$g$" "$A_K$" "$A_L$"];
ktMap = @(t, x) pchip(k, s, x);

IT = 100; % trajectory mesh size
a = 1e-1; % mesh distribution parameter
dT = 1/(IT-1); % mesh density

tmax = 100; t=(0:0.1:tmax)';
tmin = -floor(tmax/20);

for j=1:length(params)
    params(j) = params(j)/1.1;
    rho = params(1);
    theta = params(2);
    nu = params(3);
    delta = params(4);
    g = params(5);
    uK = params(6);
    uL = params(7);
    
    if (gamma==1) fs = uK*uL*ks.^nu;
    else fs = (nu*(uK*ks).^(1-1/gamma) + (1-nu)*uL.^(1-1/gamma)).^(gamma/(gamma-1));
    end
    f_p1s = diff(fs, ks);
    f_p2s = diff(f_p1s, ks);

    f = matlabFunction(fs, "vars", ks);
    f_p1 = matlabFunction(f_p1s, "vars", ks);
    f_p2 = matlabFunction(f_p2s, "vars", ks);
    
    metric=1; i=0; kSS0=1;
    while (metric>=1e-10)
        kSS0 = kSS0 - (f_p1(kSS0)-delta - rho - theta*g)/f_p2(kSS0);
        metric = abs(f_p1(kSS0)-delta - rho - theta*g); i=i+1;
    end    
    cSS0 = f(kSS0) - (delta+g+n)*kSS0;
    
    [t, kt] = ode45(ktMap, [0 tmax], kSS0); %ode23t(ktMap, [0 tmax], kSS0);
    ct = pchip(k, c, kt);
    
    figure;
    subplot(2, 1, 1); plot(t, kt, "linewidth", 1.2, "color", [1 0.3 0]);
    line([tmin 0], [kSS0 kSS0], "linewidth", 1.2, "color", [1 0.3 0]);
    line([tmin tmax], [kSS kSS], "linestyle", "--", "color", [0 0 0]);
    xlim([tmin tmax]);
    
    xlabel("$t$", "interpreter", "latex"); ylabel("$\tilde{k}\!\left(t\right)$", "interpreter", "latex");
    title(strcat("Adjustment paths following a 10\% increase in~", names(j)), ...
        "interpreter", "latex", "fontsize", 12);
    
    subplot(2, 1, 2); plot(t, ct, "linewidth", 1.2, "color", [1 0.3 0]);
    line([0 0], [cSS0 ct(1)], "linestyle", "--", "color", [1 0.3 0]);
    line([tmin 0], [cSS0 cSS0], "linewidth", 1.2, "color", [1 0.3 0]);
    line([tmin tmax], [cSS cSS], "linestyle", "--", "color", [0 0 0]);
    xlim([tmin tmax]);
    
    xlabel("$t$", "interpreter", "latex");
    ylabel("$\tilde{c}\!\left(t\right)$", "interpreter", "latex");
    params = params0;
end