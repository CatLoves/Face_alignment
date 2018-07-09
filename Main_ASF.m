% Implementation of Dual Sparse Constrained Cascade Regression for Robust Face Alignment 
%   By Fedor Ng

%% Declare global variables
% main directory of this program, for convenience of program shift
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
global R_home;
global NORMALIZED_DIR;
global LOGS_DIR;
global ROLL_THRESHOLD;              % for an image , if its ROLL angle <= ROLL_THRESHOLD, do not rotate it ! 
global NUM_INITIAL_SHAPES;          % for each training image, NUM_INITIAL_SHAPES initial shapes would be used for training procedure
global Dim_Feat ;
global Dim_Phi;
global occlusion_detection_lambda;
global map_test;
global shape_constraint_lambda;
% once the variances of all landmarks are computed && sortd, we use
% var_rank_para points to estimate Transformation Parameter gama.
global var_rank_para;

global feature_selection_lambda;
global Delaunay_triangles_frontal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data structure designing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global train_para;
% model controls which database to be used.
% details:
% model.num_landmarks=68;
% model.dimension=model.num_landmarks*2;
% model.name='pie';
global model;
% N = num of images for training or testing
% N1 = N * L 
global N;
global N1;
% %%%%%%%%%%%%%%%%%%%% 3D poses for left/frontal/right model 
% NOTE: Only ( Yaw , Pitch ) are considered, Roll can be ignored due to

% Wrapper: pose{1} =pose_left pose{2}=pose_frontal pose{3}=pose_right
global pose;            
% t is stage indicator, t = 0( initial state ) 1 2 3 ... T
global t;
global GLOBAL_IOD;      % In our experiments, IOD = 100 (pixels) for all model

% %%%%%%%%%%%%%%%%%%%% Data structure for different models %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Mapping from left/right/frontal to the entire set
% %%%%%%%%%%%%%%%%%%%% left(1)= 13 means that the first one of left model
% corresponds to the 13th of the entire set( eg. arr_imgs_attri(13) )
% Using this mapping , there is no need to store
% shape_model_Helen_train_left , shape_model_Helen_train_right etc !!
% Luckily, Cell structure is perfectly suitable for this goal.
% For example, Entire trainset consists of 10 images
% Map{1} = [ 1 3 7 9 ] : left model contains the [1 3 7 9]th of the entire.
% Map{2} = [ 2 4 ] : frontal model contains the [2 4]th of the entire.
% Map{2} = [ 5 6 8 10] :right model contains the [5 6 8 10]th of the entire.
% This structure can support 4 models , 5 models , which is flexible
global M;           % Num of models, 3 by default( left / frontal / right )
global DATA_SET;
global CAFFE_HOME;
global ERROR_DIR;     % save the error pictures etc
global marker;        % ID of a specific experiment
global map;
global D;
global feat_para;
%% Set Software Platform
PLATFORM = 'Windows';    % 'Windows' or Ubuntu
MACHINE = 'Desktop';    % Laptop or Desktop
marker = 'Original';
R_home = 'D:/Program Files/R/R-3.5.0'; % Path to R installation
if strcmp( PLATFORM , 'Windows' ) 
    if strcmp( MACHINE , 'Desktop' )
        MAIN_DIR = 'F:/face_alignment/Matlab';  
        CAFFE_HOME = 'F:/face_alignment/Matlab/Tools/MTCNN_face_detection' ;
    else
        MAIN_DIR = 'G:/face_alignment_code/Matlab';  
        CAFFE_HOME = 'G:/face_alignment_code/Matlab/Tools/MTCNN_face_detection' ;
    end
elseif strcmp( PLATFORM , 'Ubuntu' )
    MAIN_DIR = '~/Desktop/face_alignment/Matlab';
    CAFFE_HOME = '~/Desktop/face_alignment/Matlab/Tools/MTCNN_face_detection' ;
else
    error('Invalid platform');
end
load('delaunay_triangles_cofw.mat','tri_left','tri_frontal','tri_right');
Delaunay_triangles_frontal = tri_frontal;

%% Set some pathes
PREPROCESSING_DIR = strcat( MAIN_DIR , '/Preprocessing' );     % path for preprocessing
TRAIN_CODE_DIR = strcat( MAIN_DIR , '/Training' );             % path for training
LOGS_DIR = strcat( MAIN_DIR , '/Logs' );
ERROR_DIR = [ MAIN_DIR , '/errors' ];

%% Set Dataset and train/test mode
DATA_SET = 'cofw';
TRAIN_OR_TEST = 'TRAIN';   
if strcmp( DATA_SET , 'helen' )
    model = shapeGt( 'createModel' , 'pie' );           % #landmarks = 68
    if strcmp( MACHINE , 'Desktop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'F:/face_dataset/helen/trainset';
            TESTING_DIR  = 'F:/face_dataset/helen/testset';
            NORMALIZED_DIR = 'F:/face_dataset/helen/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/E/face_dataset/helen/trainset';
            TESTING_DIR  = '/mnt/E/face_dataset/helen/testset';
            NORMALIZED_DIR = '/mnt/E/face_dataset/helen/normalized';
        else
            error('Invalid option');
        end
    elseif strcmp( MACHINE , 'Laptop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'D:/face_data/helen/trainset';
            TESTING_DIR  = 'D:/face_data/helen/testset';
            NORMALIZED_DIR = 'D:/face_data/helen/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/Documents/face_data/helen/trainset';
            TESTING_DIR  = '/mnt/Documents/face_data/helen/testset';
            NORMALIZED_DIR = '/mnt/Documents/face_data/helen/normalized';
        else
            error('Invalid option');
        end
    else
        error('Invalid option');
    end
    Input_images_format = 'jpg';
elseif strcmp( DATA_SET , 'lfpw' )
    if strcmp( MACHINE , 'Desktop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'F:/face_dataset/LFPW/trainset';
            TESTING_DIR  = 'F:/face_dataset/LFPW/testset';
            NORMALIZED_DIR = 'F:/face_dataset/LFPW/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/E/face_dataset/LFPW/trainset';
            TESTING_DIR  = '/mnt/E/face_dataset/LFPW/testset';
            NORMALIZED_DIR = '/mnt/E/face_dataset/LFPW/normalized';
        else
            error('Invalid option');
        end
    elseif strcmp( MACHINE , 'Laptop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'D:/face_data/LFPW/trainset';
            TESTING_DIR  = 'D:/face_data/LFPW/testset';
            NORMALIZED_DIR = 'D:/face_data/LFPW/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/Documents/face_data/LFPW/trainset';
            TESTING_DIR  = '/mnt/Documents/face_data/LFPW/testset';
            NORMALIZED_DIR = '/mnt/Documents/face_data/LFPW/normalized';
        else
            error('Invalid option');
        end
    else
        error('Invalid option');
    end
    Input_images_format = 'png';
elseif strcmp( DATA_SET , 'afw' )
    if strcmp( MACHINE , 'Desktop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'F:/face_dataset/afw/trainset';
            TESTING_DIR  = 'F:/face_dataset/afw/testset';
            NORMALIZED_DIR = 'F:/face_dataset/afw/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/E/face_dataset/afw/trainset';
            TESTING_DIR  = '/mnt/E/face_dataset/afw/testset';
            NORMALIZED_DIR = '/mnt/E/face_dataset/afw/normalized';
        else
            error('Invalid option');
        end
    elseif strcmp( MACHINE , 'Laptop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'D:/face_data/afw/trainset';
            TESTING_DIR  = 'D:/face_data/afw/testset';
            NORMALIZED_DIR = 'D:/face_data/afw/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/Documents/face_data/afw/trainset';
            TESTING_DIR  = '/mnt/Documents/face_data/afw/testset';
            NORMALIZED_DIR = '/mnt/Documents/face_data/afw/normalized';
        else
            error('Invalid option');
        end
    else
        error('Invalid option');
    end
    Input_images_format = 'jpg'; 
elseif strcmp( DATA_SET , 'ibug' )
    if strcmp( MACHINE , 'Desktop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'F:/face_dataset/ibug/trainset';
            TESTING_DIR  = 'F:/face_dataset/ibug/testset';
            NORMALIZED_DIR = 'F:/face_dataset/ibug/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/E/face_dataset/ibug/trainset';
            TESTING_DIR  = '/mnt/E/face_dataset/ibug/testset';
            NORMALIZED_DIR = '/mnt/E/face_dataset/ibug/normalized';
        else
            error('Invalid option');
        end
    elseif strcmp( MACHINE , 'Laptop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'D:/face_data/ibug/trainset';
            TESTING_DIR  = 'D:/face_data/ibug/testset';
            NORMALIZED_DIR = 'D:/face_data/ibug/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/Documents/face_data/ibug/trainset';
            TESTING_DIR  = '/mnt/Documents/face_data/ibug/testset';
            NORMALIZED_DIR = '/mnt/Documents/face_data/ibug/normalized';
        else
            error('Invalid option');
        end
    else
        error('Invalid option');
    end
    Input_images_format = 'jpg'; 
elseif strcmp( DATA_SET , 'cofw' )
    % Create model first of all
    model = shapeGt( 'createModel' , 'cofw' );           % #landmarks = 29
    if strcmp( MACHINE , 'Desktop' )
        if strcmp( PLATFORM , 'Windows' )
            if strcmp(marker,'modified')
                TRAINING_DIR = 'F:\face_dataset\COFW\my_modified_version\train';
            elseif strcmp( marker, 'Original')
                TRAINING_DIR = 'F:\face_dataset\COFW\Original\train';
            else
                error('invalid');
            end
            TESTING_DIR  = 'F:/face_dataset/COFW/Original/test';
            DATA_SET_DIR = 'F:/face_dataset/COFW/Original';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/home/fedor/Desktop/face_dataset/COFW/Original/train';
            TESTING_DIR  = '/home/fedor/Desktop/face_dataset/COFW/Original/test';
            DATA_SET_DIR = '/home/fedor/Desktop/face_dataset/COFW/Original';
%             TRAINING_DIR = '/home/fedor/Desktop/face_dataset/COFW/my_modified_version/train';
%             TESTING_DIR  = '/home/fedor/Desktop/face_dataset/COFW/my_modified_version/test';
%             DATA_SET_DIR = '/home/fedor/Desktop/face_dataset/COFW/my_modified_version';
            fprintf('Original version is used, please note !\n');
        else
            error('Invalid option');
        end
    elseif strcmp( MACHINE , 'Laptop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'D:/face_data/COFW/trainset';
            TESTING_DIR  = 'D:/face_data/COFW/testset';
            NORMALIZED_DIR = 'D:/face_data/COFW/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/Documents/face_data/COFW/trainset';
            TESTING_DIR  = '/mnt/Documents/face_data/COFW/testset';
            NORMALIZED_DIR = '/mnt/Documents/face_data/COFW/normalized';
        else
            error('Invalid option');
        end
    else
        error('Invalid option');
    end
    Input_images_format = 'png'; 
elseif strcmp( DATA_SET , '300w' )
    % Create model first of all
    model = shapeGt( 'createModel' , 'pie' );           % #landmarks = 29
    if strcmp( MACHINE , 'Desktop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'F:/face_dataset/300w/trainset';
            TESTING_DIR  = 'F:/face_dataset/300w/testset/common_subset';
            NORMALIZED_DIR = 'F:/face_dataset/300w/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/home/fedor/Desktop/face_dataset/300w/trainset';
            TESTING_DIR  = '/home/fedor/Desktop/face_dataset/300w/testset/common_subset';
            NORMALIZED_DIR = '/home/fedor/Desktop/face_dataset/300w/normalized';
        else
            error('Invalid option');
        end
    elseif strcmp( MACHINE , 'Laptop' )
        if strcmp( PLATFORM , 'Windows' )
            TRAINING_DIR = 'D:/face_data/COFW/trainset';
            TESTING_DIR  = 'D:/face_data/COFW/testset';
            NORMALIZED_DIR = 'D:/face_data/COFW/normalized';
        elseif strcmp( PLATFORM , 'Ubuntu' )
            TRAINING_DIR = '/mnt/Documents/face_data/COFW/trainset';
            TESTING_DIR  = '/mnt/Documents/face_data/COFW/testset';
            NORMALIZED_DIR = '/mnt/Documents/face_data/COFW/normalized';
        else
            error('Invalid option');
        end
    else
        error('Invalid option');
    end
    
elseif strcmp( DATA_SET , 'ocfw' )
    model = shapeGt( 'createModel' , 'pie' );           % #landmarks = 68
    if strcmp( PLATFORM , 'Windows' )
        TRAINING_DIR = 'F:\face_dataset\MVOCFW\OCFW\train';
        TESTING_DIR  = 'F:\face_dataset\MVOCFW\OCFW\test';
        NORMALIZED_DIR = 'F:/face_dataset/helen/normalized';
    elseif strcmp( PLATFORM , 'Ubuntu' )
        TRAINING_DIR = '/mnt/E/face_dataset/helen/trainset';
        TESTING_DIR  = '/mnt/E/face_dataset/helen/testset';
        NORMALIZED_DIR = '/mnt/E/face_dataset/helen/normalized';
    else
        error('Invalid option');
    end
elseif strcmp( DATA_SET , 'ocfw_unoccluded' )
    model = shapeGt( 'createModel' , 'pie' );           % #landmarks = 68
    if strcmp( PLATFORM , 'Windows' )
        TRAINING_DIR = 'F:\face_dataset\MVOCFW\OCFW\Unoccluded_faces';
        TESTING_DIR  = 'Unknown';
        NORMALIZED_DIR = 'F:/face_dataset/helen/normalized';
    elseif strcmp( PLATFORM , 'Ubuntu' )
        TRAINING_DIR = '/mnt/E/face_dataset/helen/trainset';
        TESTING_DIR  = '/mnt/E/face_dataset/helen/testset';
        NORMALIZED_DIR = '/mnt/E/face_dataset/helen/normalized';
    else
        error('Invalid option');
    end    
else    
    error('invalid');
end
Input_images_format = { 'jpg', 'png' };
NUM_LANDMARKS = model.num_landmarks ;
if strcmp( TRAIN_OR_TEST , 'TRAIN' )
    Input_images_dir = TRAINING_DIR;
else
    Input_images_dir = TESTING_DIR;
end
% display info
fprintf('DATA_SET: %s\n', DATA_SET );
fprintf('Train_or_test: %s\n', TRAIN_OR_TEST );
fprintf('Training dir: %s\n', TRAINING_DIR );
fprintf('Testing dir: %s\n\n', TESTING_DIR );

%% Set training parameters
M = 3; map = cell( 1 , M ); map_test = cell( 1, M );
NUM_INITIAL_SHAPES = 10;
marker = 'Original';
train_para = struct( ...
                     'shape_initialization','mean_shape_with_Gaussian_sampling',...
                     'T_stages' , 7 , ...
                     'L_num_initial_shapes' , NUM_INITIAL_SHAPES , ...
                     'shape_center' , 'center' , ...                        % The center of shape, needed for standard shape representation, can be nose_tip , eye_center , center
                     'box_type'    , '1_28_gt_box' , ...                    % Very important for alignment performance.
                     'occlu_enabled' , 1, ...                               % Whether the occlusion info should be used.
                     'occlu_lambda', 0.0015, ...                            % sparse logistic regression matrix
                     'occlusion_level_threshold', 0.25, ...                 % if occlusion_level >= occlusion_level_threshold, consider it as occluded!
                     'face_parts' , [] , ...                                % Divide all NUM_LANDMARKS points into several( for example five ) parts
                     'dict_err_limit', 12, ...                              % error limit: | X - D*GAMMA |_2 <= dict_err_limit
                     'box_overlap_threshold', 0.5, ...                      % threshold for box overlap
                     'flip_the_img' , 1 , ...                               % flip the image to double the training images
                     'align_to_mean_shape' ,'all_points' , ...              % align to mean shape , to get rid of the effect of rotation
                     'lasso_method' , 'glmnet' ,...                         % which kind of Lasso solvers to use, matlab or glmnet
                     'head_pose_estimation_method' , 'PNP', ...             % Estimate head pose by PNP/CNN/...
                     'var_rank_para' , round(NUM_LANDMARKS*0.9) ,...        % variation rank para
                     'std_w', 70,...                                        % standard width of face region
                     'std_h', 70,...                                       % standard height of face region. height/width is about 1.2(mean of training samples)
                     'padding',8, ...                                      % Padding to [ std_h, std_w ]
                     'channels', 1 ...                                      % RGB or gray levels
                    );
train_para.mean_shape = cell( 1 , M );
ROLL_THRESHOLD = 4;
GLOBAL_IOD = 100;
% Special rules
if strcmp( DATA_SET , 'lfw' )
    train_para.flip_the_img = 0;
end

% Divide the face landmarks into 5 parts
if NUM_LANDMARKS == 68
    train_para.face_parts = cell( 1 , 5 );
    train_para.face_parts{1} = [1,18:22,37:42];
    train_para.face_parts{2} = [23:27,17,43:48];
    train_para.face_parts{3} = [28:36];
    train_para.face_parts{4} = [2:8,51,62,68,59,50,61,60,49];
    train_para.face_parts{5} = [10:16,53,64,66,57,54,65,56,55];
elseif NUM_LANDMARKS == 29
    % Divide 29 points into 4 parts , each time from 3 parts , 4 points
    % C4(3) = 4 , but we repeat for 12 times
    train_para.face_parts = cell( 1 , 4 );
    train_para.face_parts{1} = [1 3 5 6 9 11 13 14 17];
    train_para.face_parts{2} = [2 4 7 8 10 12 15 16 18];
    train_para.face_parts{3} = [19 20 21 22];
    train_para.face_parts{4} = [23:29];
else
    error('invalid option');
end

%% %%%%%%%%%%%%%%%%%%%%%%%%% Feature types & parameters %%%%%%%%%%%%%%%%%%%%%%%%%
feat_para = struct( ...
                    'CellSize' , [8 8] ,...
                    'BlockSize' , [2 2] ,...
                    'NumBins' , 9 , ...
                    'Size_Feat' , 155 , ...
                    'RGB_or_Gray' , 'RGB' ...    
                  );
% dimension of phi( Num_landmarks * 2 )
Dim_Feat = NUM_LANDMARKS * feat_para.Size_Feat;
Dim_Phi = NUM_LANDMARKS * 2;
%% %%%%%%%%%%%%%%%%%%%%%%%%% Data Augmentation %%%%%%%%%%%%%%%%%%%%%%%%%
cd( PREPROCESSING_DIR );
Data_Augmentation;

%% %%%%%%%%%%%%%%%%%%%%%%%%% Face Normalization: Detection & 5 landmarks localization & Scaling & Rotation %%%%%%%%%%%%%%%%%%%%%%%%%
Face_Normalization;
%Face_Normalization;

%% %%%%%%%%%%%%%%%%%%%%%%%%% Facial shape initialization %%%%%%%%%%%%%%%%%%%%%%%%%
train_para.shape_initialization = 'mean_shape_with_Gaussian_sampling';
Facial_shape_initialization;

%% %%%%%%%%%%%%%%%%%%%%%%%%% Dictionary Learning %%%%%%%%%%%%%%%%%%%%%%%%%
% % Dictionary 1: Shape dictionary
% save( [marker, '_', DATA_SET, '_shape_dict.mat'], 'D_shape' );
if exist([marker, '_', DATA_SET, '_shape_dict.mat'],'file') == 0
    Train_shape_dictionary
end

% % Dictionary 2: Appearance dictionary

% % Dictionary 3: Occlusion dictionary

%% Training of Dual Sparse Constrained Cascaded Shape Regeression
%  Parameter settings
feature_selection_lambda = 0.2;
shape_constraint_lambda  = 0.001;
var_rank_para = round( NUM_LANDMARKS * 0.8 );
train_para.T_stages = 7;

%% %%%%%%%%%%%%%%%%%%%%%%%%% Train K-SVD Dictionary %%%%%%%%%%%%%%%%%%%%%%%%%



%% %% %%%%%%%%%%%%%%%%%%%%%%%%% Training process %%%%%%%%%%%%%%%%%%%%%%%%%

%% Important notes:
% 1. use true IOD to normalize the size of the image.
% 2. The face is rotated around the ground truth nose.

