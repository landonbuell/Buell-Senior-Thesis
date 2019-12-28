% ================
% Landon Buell
% Read .aiff Files
% 
% 27 Dec 2019
% ================

% clear workspace
clearvars;
clc;

            %%%% Establish All directory Paths %%%%
root_dir = pwd;                                             % starting path
read_dir = 'C:\Users\Landon\Documents\aiff_audio';          % raw data path
wave_out = strrep(read_dir,'aiff_audio','waveforms');       % waveforms output
FFT_out = strrep(read_dir,'aiff_audio','frequencies');      % Frequency output
spectro_out = strrep(read_dir,'aiff_audio','spectrograms'); % spectrogram output
outdirs = {wave_out,FFT_out,spectro_out};       % all

            %%%% Organize File Folders %%%%

for N = 1:length(outdirs)           % each output path
    try                             % attempt to change dir
        chdir(string(outdirs(N)))   % change to path
    catch                           % if failure, 
        mkdir(string(outdirs(N)))   % create the dir
    end                     
end 

chdir(read_dir);                % change to reading directory
files = dir('**\*.aif');        % all '.aif' files in subfolders
disp('Number of files:')
disp(length(files))

            %%%% Read Each Audio File in Directory %%%%
            
features = 2^14;        % length to crop each audio file

for I = 1:length(files)
    
    filename = files(I).name;   % current file name
    filepath = files(I).folder; % current file path   
    chdir(filepath);            % change to specific path
    
    try                                         % try to read audio data
        [data,rate] = audioread(filename);      % read audio data  
        % data is L & R tracks, rate is audio sample rate
    catch                                       % If failure,
        disp("Data for file could not be read") % print message
        disp(filename)                          % print file name
        continue                                % skip to next iteration
    end 
    
    t = 1:length(data(1));
    tiledlayout(2,1)  
    % Top plot
    ax1 = nexttile;
    plot(ax1,t,data(1))
    % Bottom plot
    ax2 = nexttile;
    plot(ax2,t,data(2))

   

    
end 
