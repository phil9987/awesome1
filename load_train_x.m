%% Import data from text file.
% Script for importing data from the following text file:
%
%    /home/timethy/eth/4-LIS/awesome1/project_data/train.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2015/03/05 23:27:34

%% Initialize variables.
%filename = '/home/timethy/eth/4-LIS/awesome1/project_data/train.csv';
filename = './project_data/train.csv';
delimiter = ',';

%% Format string for each line of text:
%   column1: text (%s)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN, 'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
Time = dataArray{:, 1};
W1 = dataArray{:, 2};
W2 = dataArray{:, 3};
W3 = dataArray{:, 4};
W4 = dataArray{:, 5};
W5 = dataArray{:, 6};
W6 = dataArray{:, 7};

%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans;

T = datevec(Time);
WD = weekday(Time);