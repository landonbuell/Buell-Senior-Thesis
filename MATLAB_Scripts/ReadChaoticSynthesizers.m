% 
% Landon Buell
% Kevin Short
% PHYS 799
% 

% ================================

homePath = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\MATLAB_Scripts';
dataPath = "C:\\Users\\Landon\\Documents\\audioChaoticSynthesizerTXT\\PER2TO10";
exptPath = "C:\\Users\\Landon\Documents\\audioChaoticSynthesizerWAV\\PER2TO10";

files = dir(dataPath+'\*.txt');    % all files in subfolder

for i = 1:length(files)
    
    % Read matrix from file, Eliminate header
    fileName = files(i).name;
    filePath = files(i).folder + "\" + fileName;
    A = readmatrix(filePath);
    A(1,:) = [];
    A(:,1) = [];
        
    % Get channel, and extend
    names = ["X.wav","Y.wav","Z.wav"];
    fileName = strrep(fileName,".txt","");
   
    for j = 1:3
        % extend channel
        chdir(homePath);
        x = ExtendWaveform(A(:,j),7);
        outName = strcat(fileName,names(j));
        % Export Audio
        chdir(exptPath);
        audiowrite(outName,x,44100)       
    end

end 





