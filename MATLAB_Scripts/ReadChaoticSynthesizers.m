% 
% Landon Buell
% Kevin Short
% PHYS 799
% 

% ================================

dataPath = "C:\\Users\\Landon\\Documents\\audioChaoticSynthesizerTXT\\PER2TO10";
exptPath = "C:\\Users\\Landon\Documents\\audioChaoticSynthesizerWAV\\PER2TO10";

files = dir(dataPath+'\*.txt');    % all files in subfolder

for i = 1:length(files)
    
    % Read matrix from file, Eliminate header
    filePath = files(i).folder + "\" + files(i).name;
    A = readmatrix(filePath);
    A(1,:) = [];
        
    % Get channel, and extend
    names = ["X.txt","Y.txt","Z.txt"];

    for j = 1:3
        % extend channel
        x = ExtendWaveform(A(:,j),7);
        outName = exptPath + "\" + fileName + names(j)
    end

end 





