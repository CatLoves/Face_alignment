%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training a shape dictionary %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Variables
global MAIN_DIR;
global PREPROCESSING_DIR;
global TRAIN_CODE_DIR;
global DATA_SET_DIR;
global TRAIN_OR_TEST;
global NUM_LANDMARKS;
global Input_images_dir;
global Input_images_format;
global TRAINING_DIR;
global TESTING_DIR;
global NORMALIZED_DIR;
global LOGS_DIR;
global ROLL_THRESHOLD;              % for an image , if its ROLL angle <= ROLL_THRESHOLD, do not rotate it ! 
global NUM_INITIAL_SHAPES;          % for each training image, NUM_INITIAL_SHAPES initial shapes would be used for training procedure
global Dim_Feat ;
global Dim_Phi;
global GLOBAL_DEBUG_MODE ;     % Control mode : test or debug
global map_test;
global shape_constraint_lambda;
global marker;
global M;           % Num of models, 3 by default( left / frontal / right )
global DATA_SET;
global CAFFE_HOME;
global PREPROCESSING_RESULT_FILE;
global DICTIONARY_RESULT_FILE;
global sparse_error_lambda;
global ERROR_DIR;     % save the error pictures etc
global map;
global D;
global feat_para;
global model;

%% Load data
Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
if ~exist('arr_imgs_attri_train') || isempty( arr_imgs_attri_train )
    load( Normalization_file );
    arr_imgs_attri_train = arr_imgs_attri;
end
Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
if ~exist('arr_imgs_attri_test') || isempty( arr_imgs_attri_test )
    load( Normalization_file );
    arr_imgs_attri_test = arr_imgs_attri;
end
load( ['map_',DATA_SET,'.mat'], 'map');
load( ['map_test_',DATA_SET,'.mat'], 'map_test');
load( ['mean_shapes_',DATA_SET,'.mat'], 'mean_shapes' );
% Do not divide
map{2}=1:length(arr_imgs_attri_train);map{1}=[];map{3}=[];
map_test{2}=1:length(arr_imgs_attri_test);map_test{1}=[];map_test{3}=[];

for m = 1 : 3
    n = length( map{m} ); if n ==0; continue; end
    mean_shape = mean_shapes{m};
    Tr_samples = zeros( NUM_LANDMARKS*2, n);
    for i = 1 : n
        item = arr_imgs_attri_train( map{m}(i) );
        true_shape = double( item.true_coord_all_landmarks );
        [d,z,gama]=procrustes( mean_shape', true_shape');
        aligned_shape = z';
        % centerize
        aligned_shape = aligned_shape - repmat( mean(aligned_shape,2),1,NUM_LANDMARKS);
        % unit norm
        aligned_shape = aligned_shape ./ norm( reshape(aligned_shape,numel(aligned_shape),1));
        % collect
        Tr_samples(:,i) = reshape( aligned_shape, numel(aligned_shape),1);
        % debugging
%         close all; figure(i);
%         subplot(1,3,1); shapeGt('disp_shape',mean_shape,'shape');title('mean');
%         subplot(1,3,2); shapeGt('disp_shape',true_shape,'shape');title('true');
%         subplot(1,3,3); shapeGt('disp_shape',aligned_shape,'shape');title('aligned');
    end
    
    % Training the K-SVD Dictionary
    error_limit = 0.02;
    dict_size = 200;
    PARAMS = struct( ...
    'data' , Tr_samples , ...
    'Edata' , error_limit , ...
    'dictsize' , dict_size,  ...
    'iternum'  , 50 ...
    ); 
    [ D , GAMMA_shape ] = ksvd( PARAMS , 'tr' );
    D_shape{m} = D;
end

% %% Prepare training samples
% Train_samples = zeros( 2*NUM_LANDMARKS, length(arr_imgs_attri) );
% for i = 1 : length(arr_imgs_attri_train)
%     item = arr_imgs_attri_train(i);
%     shape = double( item.true_coord_all_landmarks );
%     % %%%%%%%%%%%%%% Centerize the shape at mean(shape,2) && To unit l2
%     % norm
%     center = mean( shape, 2 );
%     centered_shape = shape - repmat( center, 1, NUM_LANDMARKS );
%     norm_shape = reshape( centered_shape, numel(centered_shape), 1);
%     norm_shape = norm_shape / norm(norm_shape);
%     
%     Train_samples(:,i) = norm_shape;
%     de = 0;
% end
% 
% %% Train the K-SVD dictionary
% error_limit = 0.02;
% dict_size = 200;
% PARAMS = struct( ...
% 'data' , Train_samples , ...
% 'Edata' , error_limit , ...
% 'dictsize' , dict_size,  ...
% 'iternum'  , 50 ...
% ); 
% [ D_shape , GAMMA_shape ] = ksvd( PARAMS , 'tr' );

%% Evaluate the K-SVD dictionary
lambda = 0.001;
dis_per_img = zeros( length( arr_imgs_attri_test ), 1 );
for i = 1 : length( arr_imgs_attri_test )
    item = arr_imgs_attri_test(i);
    true_shape = double( item.true_coord_all_landmarks );
    row_true_shape = [ true_shape(1,:) true_shape(2,:) ];
    % Shape is the initial shape
    shape_row = double(item.init_shapes.(train_para.shape_initialization)(1,:) );
    shape = [shape_row(1:NUM_LANDMARKS); shape_row(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
    [ds,dsAll,ds_per_landmark] = shapeGt( 'dist_per_landmark', model, shape_row, row_true_shape );
    nme_before_ssc(i) = ds;
    if mod(i,100)==0
        fprintf('before ssc, NME = %.2f %%\n', ds*100 );
    end
    
%     if item.pose_angles(1) < -10
%         mean_shape = mean_shapes{1};
%         m = 1;
%     elseif item.pose_angles(2) > 10
%         mean_shape = mean_shapes{3};
%         m = 3;
%     else
%         mean_shape = mean_shapes{2};
%         m = 2;
%     end
    mean_shape = mean_shapes{2};
    
    % Align to mean shape
    [d,z,gama]=procrustes( mean_shape', shape');
    z = z';
    aligned_shape = z;
    
    % %%%%%%%%%%%%%% Centerize the shape at mean(shape,2) && To unit l2
    % norm
    center = mean( aligned_shape, 2 );
    centered_shape = aligned_shape - repmat( center, 1, NUM_LANDMARKS );
    norm_shape = reshape( centered_shape, numel(centered_shape), 1);
    shape_norm = norm(norm_shape);
    norm_shape = norm_shape / norm(norm_shape);
    
    %  Express the test shape over the learnt shape dictionary
    fit = glmnet( D_shape{2}, norm_shape, 'gaussian', struct('lambda', 0.0001,'intr',0) );
    estimate = D_shape{2} * fit.beta;

    % % % % Find most similar shapes and fit by least squares
    % Find the most similar shapes
%     for i = 1 : 10 : size( D_shape{2}, 2 )
%         dict_s = D_shape{2}(:,i);
%         cur_s = norm_shape;
%         
%         close all;
%         subplot(1,2,1); shapeGt('disp_shape',dict_s,'atom'); title('dict');
%         subplot(1,2,2); shapeGt('disp_shape',cur_s,'atom'); title('current');
%     end
    
    % Transform back to original form
    estimate = estimate * shape_norm;
    estimate = reshape( estimate, 2, NUM_LANDMARKS );
    estimate = estimate + repmat( center, 1, NUM_LANDMARKS );
    ori_shape = ( ( estimate' - gama.c ) / gama.b ) /(gama.T);
    ori_shape = ori_shape';
    ori_shape_row = [ ori_shape(1,:) ori_shape(2,:) ];
    
    % Calculate residual
    [ds,dsAll,ds_per_landmark] = shapeGt( 'dist_per_landmark', model, ori_shape_row, row_true_shape );
    nme_after_ssc(i) = ds;
    if mod(i,100)==0
        fprintf('after ssc, NME = %.2f %% DF = %d Yaw angle = %.2f \n', ds*100, fit.df , item.pose_angles(1));
    end
end

nme_ssc_dif = nme_after_ssc - nme_before_ssc;
fprintf('Before ssc, mean nme = %.2f %%\n', mean(nme_before_ssc)*100 );
fprintf('After  ssc, mean nme = %.2f %%\n', mean(nme_after_ssc)*100 );

%% Save && Show result
save( [marker, '_', DATA_SET, '_shape_dict.mat'], 'D_shape' );
close all; h = histogram( dis_per_img );
