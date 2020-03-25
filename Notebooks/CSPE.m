function [Xout,Ycps]=CSPE(indat,varargin)
%This function file will calculate the CSPE.
%The variable arguments allow for the data to be passed in windowed or unwindowed.
%If data is unwindowed, and a hanning window is to be used, 
%   varargin=['unwindowed',LEN,Step]
%If data is unwindowed, and a different window is to be used, 
%   varargin=['unwindowed',LEN,Step,'win',OVRLAP,METHOD]
%If the data is windowed, 
%   varargin=['windowed',indat_shifted,LEN,Step]
%If the data is already transformed to the frequency domain
%   varargin=['freq_transformed',FFT_of_indat_shifted,LEN,Step]
%If the CSPE is being called just to update certain affected bins after
%subtracting out a reconstructed peak, then rrange=range of bins involved and
%   varargin=['update',FFT_of_indat_shifted,LEN,Step,rrange]
%USAGE: [Xout,Ycps]=CSPE(indat,varargin)

% dat=zeros(2*LEN,1); 
% dat1=zeros(2*LEN,1); 

[PaddedLEN,Ndim] = size(indat);
if (PaddedLEN < Ndim)
    fprintf('WARNING: PaddedLEN is less than the Ndims, so make sure indat is in column-oriented form, PaddedLEN x Ndim.\n');
end


if strcmp('unwindowed',varargin{1})
    LEN=varargin{2};
    Step=varargin{3};
    hh=hanning(LEN,'periodic');
    if length(varargin)==6
        if strcmp('win',varargin{4})
            OVRLAP=varargin{5};
            METHOD=varargin{6};
            hh=PrepareAnalysisWindow(LEN,OVRLAP,METHOD);
        end
    end
    hh = hh*ones(1,Ndim);
    dat=zeros(2*LEN,Ndim); 
    dat1=zeros(2*LEN,Ndim);
    dat(1:LEN,:)=indat(1:LEN,:).*hh(1:LEN,:); 
    dat1(1:LEN,:)=indat(1+Step:LEN+Step,:).*hh(1:LEN,:);
    C=fft(dat1).*conj(fft(dat));
    DenomFactor = Step;
end

if strcmp('windowed',varargin{1})
    LEN=varargin{3};
    Step=varargin{4};
    dat=zeros(2*LEN,Ndim); 
    dat1=zeros(2*LEN,Ndim);
    dat(1:LEN,:)=indat(1:LEN,:); 
    dat1(1:LEN,:)=varargin{2}(1:LEN,:);
    C=fft(dat1).*conj(fft(dat));
    DenomFactor = Step;
end

if strcmp('freq_transformed',varargin{1})
    LEN=varargin{3};
    Step=varargin{4};
%     dat=zeros(2*LEN,1); 
%     dat1=zeros(2*LEN,1);
%     dat(1:LEN)=indat(1:LEN); 
%     dat1(1:LEN)=varargin{2}(1:LEN);
%     C=fft(dat1).*conj(fft(dat));
    fdat=indat; 
    fdat1=varargin{2}(1:end,:);
    C=fdat1.*conj(fdat);
    DenomFactor = Step;
end


if strcmp('xcspe_freq_transformed',varargin{1})
    %Here we assume that the data comes from two different channels, and
    %that it has been multiplied by an analysis window, padded with zeros
    %to 2x the orig length and then been FFT'd.  LEN = length of original
    %data before padding to 2x length.
    LEN=varargin{3};
    Step=varargin{4};
    truefreq = varargin{5}(1:end,:);
        [truefreqLEN,truefreqDim]=size(truefreq);
        if (truefreqLEN ~= PaddedLEN)
            fprintf('ERROR: input freq vector of wrong length, so exiting.\n');
            return;
        end
        if (truefreqDim ~= Ndim)
            fprintf('WARNING: Input Dims of truefreq vector not equal to input data dim, so copying to make dimension the same.\n');
            if (truefreqDim < Ndim)
                truefreq_tmp = zeros(PaddedLEN,Ndim);
                truefreq_tmp(:,1:truefreqDim)=truefreq;
                truefreq_tmp(:,truefreqDim+1:Ndim) = truefreq(:,1)*ones(1,Ndim-truefreqDim);
                truefreq = truefreq_tmp;
            else
                truefreq = truefreq(:,1:Ndim);
            end
        end
    fdat=indat; 
    fdat1=varargin{2}(1:end,:);
    C=fdat1.*conj(fdat);
    DenomFactor = truefreq;
end

if strcmp('update',varargin{1})
    LEN=varargin{3};
    Step=varargin{4};
    rrange=varargin{5}(1:end,:);
    if length(rrange) ~= LEN
        fprintf('ERROR, input should be just the bins from the CSPE that should be updated.  LEN should equal length(rrange).\n');
        return;
    end
%     dat=zeros(2*LEN,1); 
%     dat1=zeros(2*LEN,1);
%     dat(1:LEN)=indat(1:LEN); 
%     dat1(1:LEN)=varargin{2}(1:LEN);
%     C=fft(dat1).*conj(fft(dat));
    fdat=indat; 
    fdat1=varargin{2}(1:end,:);
    C=fdat1.*conj(fdat);
    DenomFactor = Step;
end

binlevels=zeros(2*LEN,Ndim);
Jump=1.0;
if (~strcmp('xcspe_freq_transformed',varargin{1}))
%binlevels=fix(([0:LEN]+(Jump/2))/Jump);
%binlevels(2*LEN:-1:LEN+1)=-binlevels(1:LEN);
Jump=fix(2*LEN/Step);
%levels=fix(([0:LEN-1]+(Jump/2))/Jump); %It appears that the transition to
%the next level was off by a bin, so I am adjusting it 14 June 2019
levels=fix(([0:LEN]'+(Jump/2))/Jump);
%binlevels(2:LEN+1)=levels;
binlevels(1:LEN+1,1:Ndim) = levels*ones(1,Ndim);

%binlevels(2*LEN:-1:LEN+2)=-levels(1:LEN-1);
binlevels(2*LEN:-1:LEN+2,1:Ndim)=-levels(2:LEN)*ones(1,Ndim);


if strcmp('update',varargin{1})
    binlevels=binlevels(rrange,:);
end
end

Xout=(2*LEN)*angle(C)./((2*pi)*DenomFactor)+Jump*binlevels;

%The following section seems wrong, since if we have a large jump caused by
%a Step larger than 1, then we should just get added multiples of pi.  We
%wouldn't just set the result equal to kk.  I assume this code is a check
%on the binlevels correction above, and that it wouldn't normally come into
%play.
% for kk=1:LEN
%     if (abs(Xout(kk)-kk) > Jump/2)
%         Xout(kk)=kk;
%     end
% end
% 
% for kk=1:LEN-1
%     if (abs(Xout(2*LEN-kk+1)+kk) > Jump/2)
%         Xout(2*LEN-kk+1)=-kk;
%     end
% end
 
Ycps=sqrt(abs(C)); 

%The solution for the large Step problem seems to be that every spectral peak, the two nearest bins need to be examined to determine if the true freq is likely to
%lie to the left or to the right of the integer period bin number corresponding to the bin of the peak itself.  Hence, if the bin value to the left is greater than
%the bin value to the right, the peak is left of the integr bin number where the max appears.  Thus, the binlevel correction should be taken from the bin to the left.
%If the true peak is to the right (based on estimate derived from nearby bins), then the binlevel value should be taken from the same bin where the max occurs.
