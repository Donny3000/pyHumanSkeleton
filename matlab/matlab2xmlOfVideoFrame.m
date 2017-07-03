function [ ] = matlab2xmlOfVideoFrame( variable, fileName, frameID, flag )
%MATLAB2OPENCV Save `variable` to yml/xml file 
% fileName: filename where the variable is stored
% flag: `a` for append, `w` for writing.
%   Detailed explanation goes here

[rows, cols] = size(variable);

% Beware of Matlab's linear indexing
variable = variable';

% Write mode as default
if ( ~exist('flag','var') )
    flag = 'w'; 
end

if ( ~exist(fileName,'file') || flag == 'w' )
    % New file or write mode specified 
    file = fopen( fileName, 'w');
    fprintf( file, '<?xml version="1.0"?>\n');
    fprintf( file, '<root>\n');
else
    % Append mode
    file = fopen( fileName, 'a');
end
num=erase(fileName,'mov');
num=erase(num,'.aedat');
% Write variable header
%fprintf( file, '    <%s type_id="opencv-matrix">\n', fileName);
fprintf( file, '    <mov type_num="%s" type_id="opencv-matrix">\n', num);
fprintf( file, '        <timestamp>%s</timestamp>\n', frameID);
%%fprintf( file, '    %s: !!opencv-matrix\n', inputname(1));
%%fprintf( file, '        rows: %d\n', rows);
fprintf( file, '        <rows>%d</rows>\n', rows);
%%fprintf( file, '        cols: %d\n', cols);
fprintf( file, '        <cols>%d</cols>\n', cols);
%%fprintf( file, '        dt: f\n');
fprintf( file, '        <dt>d</dt>\n');
%%fprintf( file, '        data: [ ');
fprintf( file, '        <data>\n');

% Write variable data
for i=1:rows*cols
    fprintf( file, '%.6f', variable(i));
    if (i == rows*cols), break, end
    fprintf( file, ' ');
    if mod(i+1,4) == 0
        fprintf( file, '\n            ');
    end
end

%%fprintf( file, ']\n');
fprintf( file, '    </data>\n');
fprintf( file, '    </mov>\n');

fclose(file);
end