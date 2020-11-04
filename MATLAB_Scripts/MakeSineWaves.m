% Landon Buell
% Kevin Short
% PHYS 799
% 4 Nov 2020

        %%%% MAKE PURE SINE WAVES AS .WAV FILE %%%%%
        
nFiles = 512;
nSamples = 265216;
sampleRate = 44100;
t = (1:nSamples)/sampleRate;

exptPath = "C:\Users\Landon\Documents\SyntheticWAV";
baseFileName = "SineWave.pure.";
f = 32.70;

chdir(exptPath);
for i = 1:nFiles
    
    waveform = sin(2*pi*f*t);
    soundsc(waveform,sampleRate);
    
    freq = num2str(f);
    exptFileName = strcat(baseFileName,freq,"Hz.wav");
    audiowrite(exptFileName,waveform,sampleRate);
    
    f = f * 2^(1/12); 
end