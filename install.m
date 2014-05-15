%% install
addpath(pwd);
savepath;
builddocsearchdb([pwd '\html']);
delete('install.m');
disp('installation complete');