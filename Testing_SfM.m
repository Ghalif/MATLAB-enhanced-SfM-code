% Use |imageDatastore| to get a list of all image file names in a
% directory.
imageDir = "D:\Ismail Ghalif Shahid\Personal\Taylor's University\Sem 8\FYP2\Project\Test Data Set 7\*.jpg";
imds = imageDatastore(imageDir);

% Display the images.
figure
montage(imds.Files, 'Size', [6, 7], 'BorderSize', [2,2]);

% Convert the images to grayscale.
images = cell(1, numel(imds.Files));
for i = 1:numel(imds.Files)
I = readimage(imds, i);
images{i} = im2gray(I);
end
title('Input Image Sequence')


%Create a View Set Containing the First View
% Get intrinsic parameters of the camera
intrinsics1 = cameraParams.Intrinsics;
% Undistort the first image.
[I, intrinsics] = undistortFisheyeImage(images{1}, intrinsics1, 'ScaleFactor', 0.35);
% Detect features. Increasing 'NumOctaves' helps detect large-scale
% features in high-resolution images. Use an ROI to eliminate spurious
% features around the edges of the image.
border = 50;
roi = [border, border, size(I, 2)- 2*border, size(I, 1)- 2*border];
prevPoints = detectSURFFeatures(I, 'NumOctaves', 8, 'ROI', roi);
% Extract features. Using 'Upright' features improves matching, as long as
% the camera motion involves little or no in-plane rotation.
[prevFeatures, vptP] = extractFeatures(I, prevPoints, 'Upright', true);
% Create an empty imageviewset object to manage the data associated with each
% view.
vSet = imageviewset;
% Add the first view. Place the camera associated with the first view
% and the origin, oriented along the Z-axis.
viewId = 1;
vSet = addView(vSet, viewId, rigid3d, 'Points', vptP);
A=I;


%Add the Rest of the Views
for i = 2:numel(images)
    % Undistort the current image.
    [I, intrinsics] = undistortFisheyeImage(images{i}, intrinsics1, 'ScaleFactor', 0.35);
    % Detect, extract and match features.
    currPoints = detectSURFFeatures(I, 'NumOctaves', 8, 'ROI', roi);
    [currFeatures, vptC] = extractFeatures(I, currPoints, 'Upright', true);
    indexPairs = matchFeatures(prevFeatures, currFeatures, ...
    'MaxRatio', .2, 'Unique', true, 'MatchThreshold', 3);
    % Select matched points.
    matchedPoints1 = vptP(indexPairs(:, 1));
    matchedPoints2 = vptC(indexPairs(:, 2));

    %figure 
    %showMatchedFeatures(I,A,matchedPoints1,matchedPoints2)
    %title('Matched SIFT Points With Outliers '+i);

    % Estimate the camera pose of current view relative to the previous view.
    % The pose is computed up to scale, meaning that the distance between
    % the cameras in the previous view and the current view is set to 1.
    % This will be corrected by the bundle adjustment.
    [relativeOrient, relativeLoc, inlierIdx] = helperEstimateRelativePose(...
    matchedPoints1, matchedPoints2, intrinsics);

    % Get the table containing the previous camera pose.
    prevPose = poses(vSet, i-1).AbsolutePose;
    relPose = rigid3d(relativeOrient, relativeLoc);
    % Compute the current camera pose in the global coordinate system
    % relative to the first view.
    currPose = rigid3d(relPose.T * prevPose.T);
    % Add the current view to the view set.
    vSet = addView(vSet, i, currPose, 'Points', vptC);
    % Store the point matches between the previous and the current views.
    vSet = addConnection(vSet, i-1, i, relPose, 'Matches', indexPairs(inlierIdx,:));
    % Find point tracks across all views.
    tracks = findTracks(vSet);
    % Get the table containing camera poses for all views.
    camPoses = poses(vSet);
    % Triangulate initial locations for the 3-D world points.
    xyzPoints = triangulateMultiview(tracks, camPoses, intrinsics);
    % Refine the 3-D world points and camera poses.
    [xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
    tracks, camPoses, intrinsics, 'FixedViewId', 1, ...
    'PointsUndistorted', true);
    % Store the refined camera poses.
    vSet = updateView(vSet, camPoses);
    prevFeatures = currFeatures;
    vptP = vptC;
    A=I;
end

% Display camera poses.
camPoses = poses(vSet);
figure;
plotCamera(camPoses, 'Size', 0.2);
hold on
% Exclude noisy 3-D points.
goodIdx = (reprojectionErrors < 5);
xyzPoints = xyzPoints(goodIdx, :);
% Display the 3-D points.
pcshow(xyzPoints, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
'MarkerSize', 45);
grid on
hold off
% Specify the viewing volume.
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-5, loc1(1)+4]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);
camorbit(0, -30);
title('Refined Camera Poses');


%Compute Dense Reconstruction
% Read and undistort the first image
I = undistortImage(images{1}, intrinsics);
% Detect corners in the first image.
vptP = detectMinEigenFeatures(I, 'MinQuality', 0.001);
% Create the point tracker object to track the points across views.
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 6);
% Initialize the point tracker.
vptP = vptP.Location;
initialize(tracker, vptP, I);
% Store the dense points in the view set.
vSet = updateConnection(vSet, 1, 2, 'Matches', zeros(0, 2));
vSet = updateView(vSet, 1, 'Points', vptP);
% Track the points across all views.
for i = 2:numel(images)
    % Read and undistort the current image.
    I = undistortImage(images{i}, intrinsics);
    % Track the points.
    [vptC, validIdx] = step(tracker, I);
    % Clear the old matches between the points.
    if i < numel(images)
        vSet = updateConnection(vSet, i, i+1, 'Matches', zeros(0, 2));
    end
    vSet = updateView(vSet, i, 'Points', vptC);
    % Store the point matches in the view set.
    matches = repmat((1:size(vptP, 1))', [1, 2]);
    matches = matches(validIdx, :);
    vSet = updateConnection(vSet, i-1, i, 'Matches', matches);
end
% Find point tracks across all views.
tracks = findTracks(vSet);
% Find point tracks across all views.
camPoses = poses(vSet);
% Triangulate initial locations for the 3-D world points.
xyzPoints = triangulateMultiview(tracks, camPoses,...
intrinsics);
% Refine the 3-D world points and camera poses.
[xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(...
xyzPoints, tracks, camPoses, intrinsics, 'FixedViewId', 1, ...
'PointsUndistorted', true);


%Display Dense Reconstruction
% Display the refined camera poses.
figure;
plotCamera(camPoses, 'Size', 0.2);
hold on
% Exclude noisy 3-D world points.
goodIdx = (reprojectionErrors < 5);
% Display the dense 3-D world points.
pcshow(xyzPoints(goodIdx, :), 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
'MarkerSize', 45);
grid on
hold off
% Specify the viewing volume.
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-5, loc1(1)+4]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);
camorbit(0, -30);
title('Dense Reconstruction');