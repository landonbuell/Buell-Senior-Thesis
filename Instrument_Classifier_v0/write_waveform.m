function [L,R] = write_waveform(rawdata,filename)
% Write waveform of audio file out to CSV for external use

L = rdivide(rawdata(1,:),max(rawdata(1,:)));    % isolate & normalize left
R = rdivide(rawdata(2,:),max(rawdata(2,:)));    % isolate & normalize right

filename = filename+'.txt';
outfile = fopen(filename,'w');


end

