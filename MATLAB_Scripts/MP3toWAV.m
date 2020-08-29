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
readdir = 'C:\Users\Landon\Documents\audioMP3';          
outpath = 'C:\Users\Landon\Documents\audioWAV2';

try                         % attempt to change dir
    chdir(outpath)          % change to path
catch                       % if failure, 
    mkdir(outpath)          % create the dir
end     

chdir(readdir);             % change to reading directory
files = dir('**\*.mp3');    % all files in subfolder


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
    
    outname = strrep(files(i).name,'_','.');    % eliminate underscore       
    splitName = strsplit(outname,".");          % split string at '.'
    outname = strcat(splitName{1},".",splitName{2});% create-output name   
    outname = strcat(outname,'.LR','.wav');
    outname = char(outname);
        
    chdir(outpath)
    audiowrite(outname,data,rate);        % write track to disk
    
end 
disp("Program Complete")