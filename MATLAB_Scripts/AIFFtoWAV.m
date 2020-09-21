% ================
% Landon Buell
% Read .aiff Files
% PHYS 799
% 2 Jan 2020
% ================

% clear workspace
clearvars;
clc;

            %%%% Establish All directory Paths %%%%
rootdir = 'C:\Users\Landon\Documents\GitHub\Buell-Senior-Thesis\MATLAB_Scripts'; 
readdir = 'C:\Users\Landon\Documents\audioAIF';          
outpath = 'C:\Users\Landon\Documents\audioWAV';

try                         % attempt to change dir
    chdir(outpath)          % change to path
catch                       % if failure, 
    mkdir(outpath)          % create the dir
end     

chdir(readdir);             % change to reading directory
files = dir('**\*.aif');    % all files in subfolder


for i = 1:length(files)                 % in each file:
    
    filename = files(i).name;           % isolate filename
    dir = files(i).folder;              % isolate file folder
    chdir(dir);                         % change to specific folder
    
    try                                         % try to read audio data
        [data,rate] = audioread(filename);      % read audio data
    catch
        disp("Data for file could not be read")
        disp(files(i).name)
        continue
    end 
    
    % data is L & R racks, rate is sample rate
    data = data.';      % transpose data array
    %disp(size(data))
    
    L = data(1,:);      % left audio track
    R = data(2,:);      % right audio track
    
    % Decontruct Input name
    outname = strrep(filename,'_','.');    % eliminate underscore  
    outname = strrep(outname,'-','');           % eliminate dashes
    splitName = strsplit(outname,".");          % split string at '.'
    inst = upper(char(splitName(1)));
    
    % Reconstruct Output name
    outputName = strjoin(splitName(1,2:end-1),".");
    outputName = char(strcat(inst,".",outputName,".wav"));
    
    left_outname = strcat(outname,'.L','.wav');
    right_outname = strcat(outname,'.R','.wav');
        
    chdir(outpath)
    audiowrite(left_outname,L,rate);        % write left track
    audiowrite(right_outname,R,rate);     % write right track
    
end 
disp("Program Complete")