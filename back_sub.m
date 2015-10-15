%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUAL TRACKING
% ----------------------
% Background Subtraction
% ----------------
% Date: september 2015
% Authors: You !!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

%%%%%%%%%%%%%%%%%%%%%%% LOAD THE IMAGES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Give image directory and extension
%imPath = 'car';
imPath = 'highway';
imExt = 'jpg';

groundTruthImages = 'groundtruth';
inputImages = 'input';

% check if directory and files exist
if isdir(fullfile(imPath, inputImages)) == 0
    error('USER ERROR : The image directory does not exist');
end

if isdir(fullfile(imPath, groundTruthImages)) == 0
    error('USER ERROR : The image directory does not exist');
end

%%%%%%%%%%%% THIS IS TO LOAD THE ORIGINAL VIDEO SEQUENCE %%%%%%%%%%%%%

filearray = dir([imPath filesep inputImages filesep '*.' imExt]); % get all files in the directory
NumImages = size(filearray,1); % get the number of images
if NumImages < 0
    error('No image in the directory');
end

disp('Loading input image files from the video sequence, please be patient...');
imgname = [imPath filesep inputImages filesep filearray(1).name]; % get image name
I = imread(imgname);
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);

ImSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
for ii = 1 : NumImages
    imgname = [imPath filesep inputImages filesep filearray(ii).name]; % get image name
    ImSeq(:, :, ii) = rgb2gray(imread(imgname)); % load image
end

%%%%%%%%%% THIS IS TO LOAD THE GROUNDTRUTH VIDEO SEQUENCE %%%%%%%%%%%%

disp('Loading ground truth image files from the video sequence, please be patient...');
imExt = 'png';
filearray = dir([imPath filesep groundTruthImages filesep '*.' imExt]);
imgname = [imPath filesep groundTruthImages filesep filearray(1).name]; % get image name
ImSeq_GroundTruth = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);

for ii = 1 : NumImages
    imgname = [imPath filesep groundTruthImages filesep filearray(ii).name]; % get image name
    ImSeq_GroundTruth(:, :, ii) = imread(imgname); % load image
end

disp(' ... OK!');

%set(0, 'DefaultFigurePosition', [200 100 1024 768]);

%%%%%%%%%%%%%%%%%%% BACKGROUND SUBTRACTION 1 %%%%%%%%%%%%%%%%%%%

% n=10;
% threshold = 50;
% for i=n+1:NumImages
%     I = ImSeq(:,:,i-n:i-1);
%     Back = median(I, 3); %background image
%     
%     Current_Image = ImSeq(:,:,i);
%     D = abs(Current_Image-Back); % difference image
%     
%     Object = D > threshold;
%     
%     figure(1), subplot(1,2,1), imshow(Back, []);
%     figure(1), subplot(1,2,2), imshow(Object, []);
% end


%%%%%%%%%%%%%%%%%%%% BACKGROUND SUBTRACTION 2 %%%%%%%%%%%%%%%%%%%

N=470;
threshold = 40;
alpha = 0.1;
I = ImSeq(:,:,1:470);
figure('name', 'Background Subtraction', 'units', 'normalized', 'outerposition', [0 0.2 1 0.6]);

tic;
%because of in the question we have to use 470 images for background model, so let's do it
Background = median(I, 3);
toc;

Total_Precision=0;
Total_Recall=0;
Total_F=0;

tic;
%and then use image 471 to 1700 to detect the car on the highway
for i=N+1:NumImages
    Current_Image = ImSeq(:,:,i);
    Difference    = abs(Current_Image - Background);
    
    %We compare value of Difference to treshold
    %Object Matrix become binary image
    %because of the result of comparison is true(1) or false(0)
    Object = Difference > threshold;
     
    Object_new = bwareaopen(Object, 30);
    Object_new = imfill(Object_new, 'holes');
    Object_new = bwmorph(Object_new, 'bridge', 'Inf');
    Object_new = imfill(Object_new, 'holes');
    Object_new = bwmorph(Object_new, 'erode', 1);
    Object_new = bwmorph(Object_new, 'dilate', 1);
    Object_new = medfilt2(Object_new, [5 5]);
    Object_new = bwmorph(Object_new, 'dilate', 1);
    Object_new = bwmorph(Object_new, 'bridge', 'Inf');
    Object_new = imfill(Object_new, 'holes');
        
    stats = regionprops(bwlabel(Object), 'Centroid', 'BoundingBox', 'Area');

    figure(1), subplot(1,3,1), imshow(Current_Image, []); title('Original Images');
    [row, col] = size(stats);
    for j=1:row
        if(stats(j).Area>100)
            rectangle('Position', stats(j).BoundingBox, 'EdgeColor','y');

            hold on;
            plot(stats(j).Centroid(1), stats(j).Centroid(2), '.', 'Color', 'r', 'MarkerSize',10);
            hold off;
        end
    end
    
    figure(1), subplot(1,3,2), imshow(Object, []); title('Detected Object');
    figure(1), subplot(1,3,3), imshow(Object_new, []); title('After Morphology');
    %figure(1), subplot(2,2,4), imshow(Object_new, []);
    drawnow;
    
    Object_GroundTruth = uint8(im2bw(ImSeq_GroundTruth(:, :, i)));
    Object_GroundTruth(Object_GroundTruth == 1) = 2; 
    
    ScoreFrame = Object_GroundTruth + uint8(Object_new);
    
    True_Negative = size(find(ScoreFrame == 0), 1);
    False_Positive = size(find(ScoreFrame == 1), 1);
    False_Negative = size(find(ScoreFrame == 2), 1);
    True_Positive = size(find(ScoreFrame == 3), 1);
    
    Current_Precision = (True_Positive / (True_Positive + False_Positive));
    Total_Precision = Total_Precision + Current_Precision;
    
    Current_Recall = (True_Positive / (True_Positive + False_Negative));
    Total_Recall = Total_Recall + Current_Recall;
    
    Current_F = (2 * ((Current_Precision * Current_Recall) / (Current_Precision + Current_Recall)));
    Total_F = Total_F + Current_F;

end
toc;

display(strcat('Average Precision : ', num2str(Total_Precision/(NumImages-(N+1)))));
display(strcat('Average Recall : ', num2str(Total_Recall/(NumImages-(N+1)))));
display(strcat('Average F-Score : ', num2str(Total_F/(NumImages-(N+1)))));

%%%%%%%%%%%%%%%%%%% AVERAGE GAUSSIAN %%%%%%%%%%%%%%%%%%%

% N=470;
% Mean_Images = ImSeq(:,:,1);
% variance = ones(size(Mean_Images(:,:,1)));
% 
% alpha = 0.01;
% T = 2.5;
% 
% figure('name', 'Average Gaussian', 'units', 'normalized', 'outerposition', [0 0.2 1 0.6]);
% 
% %because of in the question we have to use 470 images for background model, so let's do it
% tic;
% for i=2:N
%     Current_Image = ImSeq(:,:,i);
%     Mean_Images = alpha * Current_Image + (1-alpha) * Mean_Images;
%     
%     distance = abs(Current_Image - Mean_Images);
%     variance = alpha * distance.^2 + (1-alpha) * variance;
% end
% toc;
% 
% Total_Precision=0;
% Total_Recall=0;
% Total_F=0;
% 
% %and then use image 471 to 1700 to detect the car on the highway
% tic;
% for i=N+1:NumImages
%     Current_Image = ImSeq(:,:,i);
%     distance = abs(Current_Image - Mean_Images);
%     
%     Object = distance > T * sqrt(variance);
%     
%     Mean_Images = alpha * Current_Image + (1-alpha) * Mean_Images;
%     variance = alpha * distance.^2 + (1-alpha) * variance;
% 
%     Object_new = bwareaopen(Object, 30, 8);
%     Object_new = bwmorph(Object_new, 'dilate', 1);
%     Object_new = bwmorph(Object_new, 'bridge', 'Inf');
%     Object_new = imfill(Object_new, 'holes');
%     Object_new = medfilt2(Object_new, [5 5]);
%     Object_new = bwmorph(Object_new, 'erode', 1);
%     Object_new = imfill(Object_new, 'holes');
%     Object_new = bwmorph(Object_new, 'dilate', 1);
%     
%     stats = regionprops(bwlabel(Object_new), 'Centroid', 'BoundingBox');
% 
%     subplot(1,3,1), imshow(Current_Image, []);  title('Original Images');
%     [row, col] = size(stats);
%     for j=1:row
%         rectangle('Position', stats(j).BoundingBox, 'EdgeColor','y');
%         
%         hold on;
%         plot(stats(j).Centroid(1), stats(j).Centroid(2), '.', 'Color', 'r', 'MarkerSize',10);
%         hold off;
%     end
%     
%     subplot(1,3,2), imshow(Object, []);  title('Detected Object');
%     subplot(1,3,3), imshow(Object_new, []);  title('After Morphology');
%     %figure(1), subplot(2,2,4), imshow(variance, []);
%     
%     drawnow;
%     
%     Object_GroundTruth = uint8(im2bw(ImSeq_GroundTruth(:, :, i)));
%     Object_GroundTruth(Object_GroundTruth == 1) = 2; 
%     
%     ScoreFrame = Object_GroundTruth + uint8(Object_new);
%     
%     True_Negative = size(find(ScoreFrame == 0), 1);
%     False_Positive = size(find(ScoreFrame == 1), 1);
%     False_Negative = size(find(ScoreFrame == 2), 1);
%     True_Positive = size(find(ScoreFrame == 3), 1);
%     
%     Current_Precision = (True_Positive / (True_Positive + False_Positive));
%     Total_Precision = Total_Precision + Current_Precision;
%     
%     Current_Recall = (True_Positive / (True_Positive + False_Negative));
%     Total_Recall = Total_Recall + Current_Recall;
%     
%     Current_F = (2 * ((Current_Precision * Current_Recall) / (Current_Precision + Current_Recall)));
%     Total_F = Total_F + Current_F;
%     
% end
% toc;
% 
% display(strcat('Average Precision : ', num2str(Total_Precision/(NumImages-(N+1)))));
% display(strcat('Average Recall : ', num2str(Total_Recall/(NumImages-(N+1)))));
% display(strcat('Average F-Score : ', num2str(Total_F/(NumImages-(N+1)))));

%%%%%%%%%%%%%%%%%%% EIGEN BACKGROUND %%%%%%%%%%%%%%%%%%%%%%

% tic;
% threshold = 40;
% N=470;
% 
% figure('name', 'Eigen Background', 'units', 'normalized', 'outerposition', [0 0.2 1 0.6]);
% 
% % code below was really really slow
% % tic;
% % currentImageVector = zeros(VIDEO_HEIGHT*VIDEO_WIDTH, 1);
% % for i=1:N
% %     currentImage = ImSeq(:,:,i);
% %     currentImageVector = currentImageVector + currentImage(:);
% % end
% % meanImageVector = currentImageVector / N;
% % 
% % X = zeros(VIDEO_HEIGHT*VIDEO_WIDTH, 0);
% % for i=1:N
% %     currentImage = ImSeq(:,:,i);
% %     X = horzcat(X, currentImage(:)-meanImageVector);
% % end
% % toc;
% 
% tic;
% X = zeros(VIDEO_HEIGHT*VIDEO_WIDTH, N);
% for i=1:N
%     Current_Image = ImSeq(:,:,i);
%     X(:,i) = Current_Image(:);
% end
% meanImageVector = sum(X,2) / N;
% X = X - repmat(meanImageVector, [1 N]);
% toc;
% 
% tic;
% [U,S,V] = svd(X,0);
% toc;
% 
% k=10;
% Uk = U(:,1:k);
% 
% tic;
% %because of in the question we have to use 470 images for background model, so let's do it
% for i=1:N
%     y=ImSeq(:,:,i);
%     p=transpose(Uk)*(y(:)-meanImageVector);
%     y_hat = Uk*p+meanImageVector;
% end
% toc;
% 
% Total_Precision=0;
% Total_Recall=0;
% Total_F=0;
% 
% %and then use image 471 to 1700 to detect the car on the highway
% tic;
% for i=N+1:NumImages
%     y=ImSeq(:,:,i);
%     D = abs(reshape(y_hat,[240,320])-y);
%     Object = D > threshold;
%     
% %     Object_new = bwareaopen(Object, 16, 8);
% %     Object_new = bwmorph(Object_new, 'bridge', 'Inf');
% %     se = strel('disk', 5);
% %     Object_new = imdilate(Object_new, se);
% %     Object_new = medfilt2(Object_new, [5 5]);
% %     Object_new = imfill(Object_new, 'holes');
% %     Object_new = bwmorph(Object_new, 'erode', 5);
%     
%     Object_new = bwareaopen(Object, 30);
%     Object_new = imfill(Object_new, 'holes');
%     Object_new = bwmorph(Object_new, 'bridge', 'Inf');
%     Object_new = imfill(Object_new, 'holes');
%     Object_new = bwmorph(Object_new, 'erode', 1);
%     Object_new = bwmorph(Object_new, 'dilate', 1);
%     Object_new = medfilt2(Object_new, [5 5]);
%     Object_new = bwmorph(Object_new, 'dilate', 1);
%     Object_new = bwmorph(Object_new, 'bridge', 'Inf');
%     Object_new = imfill(Object_new, 'holes');
%     
%     stats = regionprops(bwlabel(Object_new), 'Centroid', 'BoundingBox');
%     
%     subplot(1,3,1), imshow(y, []); title('Original Images');
%     [row, col] = size(stats);
%     for j=1:row
%         rectangle('Position', stats(j).BoundingBox, 'EdgeColor','y');
%         
%         hold on;
%         plot(stats(j).Centroid(1), stats(j).Centroid(2), '.', 'Color', 'r', 'MarkerSize',10);
%         hold off;
%     end
% 
%     subplot(1,3,2), imshow(Object, []);  title('Detected Object');
%     subplot(1,3,3), imshow(Object_new, []);  title('After Morphology');
%     %figure(1), subplot(2,2,4), imshow(ObjectOpening, []);
% 
%     drawnow;
% 
%     Object_GroundTruth = uint8(im2bw(ImSeq_GroundTruth(:, :, i)));
%     Object_GroundTruth(Object_GroundTruth == 1) = 2; 
%     
%     ScoreFrame = Object_GroundTruth + uint8(Object_new);
%     
%     True_Negative = size(find(ScoreFrame == 0), 1);
%     False_Positive = size(find(ScoreFrame == 1), 1);
%     False_Negative = size(find(ScoreFrame == 2), 1);
%     True_Positive = size(find(ScoreFrame == 3), 1);
%     
%     Current_Precision = (True_Positive / (True_Positive + False_Positive));
%     Total_Precision = Total_Precision + Current_Precision;
%     
%     Current_Recall = (True_Positive / (True_Positive + False_Negative));
%     Total_Recall = Total_Recall + Current_Recall;
%     
%     Current_F = (2 * ((Current_Precision * Current_Recall) / (Current_Precision + Current_Recall)));
%     Total_F = Total_F + Current_F;
% end
% toc;
% 
% display(strcat('Average Precision : ', num2str(Total_Precision/(NumImages-(N+1)))));
% display(strcat('Average Recall : ', num2str(Total_Recall/(NumImages-(N+1)))));
% display(strcat('Average F-Score : ', num2str(Total_F/(NumImages-(N+1)))));

