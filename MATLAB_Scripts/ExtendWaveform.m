function newSignal = ExtendWaveform(rawSignal,N)
%ExtendWaveform - Append a waveform onto itself N times
%       rawSignal (arr): 1D Array-like object like to extend
%       N (int): Number of time to append a waveform to itself

newSignal = reshape(rawSignal,1,[]);

for i = 1:N
   newSignal = [newSignal,newSignal];
    
end

end

