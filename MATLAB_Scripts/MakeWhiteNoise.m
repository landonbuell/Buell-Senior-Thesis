% Landon Buell
% Kevin Short
% PHYS 799
% 4 Nov 2020

        %%%% MAKE PURE SINE WAVES AS .WAV FILE %%%%%
        
nFiles = 200;
nSamples = 265216;
sampleRate = 44100;
t = (1:nSamples)/sampleRate;

exptPath = "C:\Users\Landon\Documents\audioNoiseWAV";
baseFileName = "WhiteNoise.";
f = 16.35;

chdir(exptPath);
for i = 1:nFiles
    
    waveform = rand(1,nSamples);
    %soundsc(waveform,sampleRate);
    
    freq = num2str(i);
    exptFileName = strcat(baseFileName,freq,".wav");
    audiowrite(exptFileName,waveform,sampleRate);

end