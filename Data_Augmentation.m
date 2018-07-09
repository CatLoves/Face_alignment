% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Augmentation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Data Augmentation by Horizontally Flipping and random
% % Rotations(Optional)
% % 
% % Key points:
% % 1. Controlled by train_para.flip_the_img
% % 2. Rotate the true_coord and box together
% % 3. 翻转之后的图片，还是放在当前图片文件夹中 
% % 4. 翻转之后，生成 $marker$_Input_imagelist.txt
%% Declaration
global Input_images_dir
global Input_images_format
global TRAIN_OR_TEST;
global PREPROCESSING_DIR;
% for an image , if its ROLL angle <= ROLL_THRESHOLD, do not rotate it ! 
global ROLL_THRESHOLD;
global DATA_SET;
global NUM_LANDMARKS;
global train_para;
global marker;        % ID of a specific experiment
cd( PREPROCESSING_DIR );
%% Generate Input_imagelist.txt
images = strcat( Input_images_dir , '/*.*g' ) ;
fprintf('Note that only PNG and JPG are supported ! \n');
imgs = dir( images );
num_img = length( imgs );
assert( num_img > 0 );
assert( NUM_LANDMARKS > 0 );

%% Note: Flipping all images, true coord, boxes( mask & occlu_label ) together
%% If rerun, please delete all '*_flip' flies !!!!!!!!!!!!!!!!!!!!!
flip_files = strcat( Input_images_dir , '/*flip*.*' ) ;
flip_files = dir( flip_files );
num_flip_files = length( flip_files );

% 对图片进行水平翻转需要满足三个条件：
% 1. 当前为 TRAIN 模式
% 2. 当前数据库需要进行 flip
% 3. 当前目录下，没有翻转过的图片
% 翻转的图片仍然放在当前目录下面
if strcmp( TRAIN_OR_TEST, 'TRAIN' ) && train_para.flip_the_img &&  num_flip_files == 0
    % display message
    fprintf('Note: Begin to flip the image...');
        
    for i = 1 : num_img
        % %%%% Get the image name
        img_name = [ Input_images_dir, '/', imgs(i).name ];
        [ path , name , ext ] = fileparts( img_name );
        
        % Flip the image
        img = imread( img_name );
        flip_img = flip( img , 2 );
        imwrite( flip_img, [ Input_images_dir, '/', name, '_flip', '.png' ] ); % png for quality, no compression !

        % Flip true coordinate
        ext = '.pts' ;
        true_pose_pts = [ path, '/' name , ext ];
        f_pose = fopen( true_pose_pts , 'r' );
        % ignore headers
        str1 = fgets( f_pose );
        str2 = fgets( f_pose );
        str3 = fgets( f_pose );
        % read the pose , a little tricky ~
        coordinates = zeros( NUM_LANDMARKS , 2 );
        coordinates = fscanf( f_pose , '%f' , NUM_LANDMARKS*2 );
        true_coords = reshape( coordinates , [ 2 , NUM_LANDMARKS ] );
        fclose( f_pose );
        % flipping
        img_size = size( img );
        flip_coord = shapeGt( 'flip_coords' , true_coords, img_size(1:2) );
        true_pose_pts = [ Input_images_dir, '/', name, '_flip', '.pts' ];
        f_out = fopen( true_pose_pts , 'w' );
        fprintf( f_out , 'version: 1\n' );
        fprintf( f_out , 'n_points:  29\n' );
        fprintf( f_out , '{\n' );
        for j = 1 : size(flip_coord,2)
            fprintf( f_out , '%.4f %.4f\n' , flip_coord(1,j) , flip_coord(2,j) );
        end
        fprintf( f_out , '}\n' );
        % close the file
        fclose( f_out );

        % Flip the box
        fp = fopen( [path,'/',name,'_box.pts'] , 'r'); 
        gt_box = fscanf( fp , '%f' );
        if size(gt_box,1) > 1
            gt_box = gt_box';
        end
        box    = shapeGt('flip_box', gt_box, img_size(1:2) );
        fclose( fp );
        fp = fopen( [ Input_images_dir, '/', name, '_flip', '_box.pts'] , 'w'); 
        fprintf( fp , '%d %d %d %d\n' , box(1) , box(2) , box(3) , box(4) );
        fclose( fp );
        
        % Flip the mask ( if mask exist )
        mask_file = [path,'/',name,'_mask.mat'];
        if exist( mask_file )
            load( mask_file, 'mask' );
            mask = flip( mask, 2 );
            mask_file = [path,'/',name,'_flip_mask.mat'];
            save( mask_file, 'mask' );
        end
        
        % Flip the occlu_label ( if exist )
        label_file = [path,'/',name,'_occlu_label.mat'];
        if strcmp( DATA_SET, 'COFW') || strcmp( DATA_SET, 'cofw')
            load( label_file, 'occlu_label' );
            flip_occlu_label = shapeGt('flip_labels', occlu_label );
            occlu_label = flip_occlu_label;
            new_label_file = [path,'/',name,'_flip_occlu_label.mat'];
            save( new_label_file, 'occlu_label' );
        end
        
        % 300w is a little more complicated
        % gt box
        if strcmp( DATA_SET, '300w' )
            fp = fopen( [path,'/',name,'_gt_box.pts'] , 'r'); 
            gt_box = fscanf( fp , '%f' );
            if size(gt_box,1) > 1
                gt_box = gt_box';
            end
            box    = shapeGt( 'flip_box', gt_box, img_size(1:2) );
            fclose( fp );
            fp = fopen( [ path, '/', name, '_flip_gt_box', '.pts'] , 'w'); 
            fprintf( fp , '%d %d %d %d\n' , box(1) , box(2) , box(3) , box(4) );
            fclose( fp );
        end 
    end
    
    % Report results
    fprintf('Data Augmentation: Horizontally flipping done.\n');
    images = strcat( Input_images_dir , '/*.*g' ) ;
    imgs   = dir(images);
    num_img = length(imgs);
    fprintf('After Augmentation, there are %d imgs.\n', num_img );
end

%% Generate Input_imagelist.txt
images = strcat( Input_images_dir , '/*.*g' ) ;
imgs = dir( images );
num_img = length( imgs );
assert( num_img > 0 );
f_Input_imagelist = fopen( [ marker, '_Input_imagelist.txt' ] , 'w' );

for i = 1 : num_img
    img_name = fullfile( Input_images_dir , imgs(i).name );
    fprintf( f_Input_imagelist , '%s\n' , img_name );
end
fclose( f_Input_imagelist );

% Display results
disp( 'Input_imagelist.txt Successfully Generated !' );

%% Unit test
% fprintf('***************************************\n');
% fprintf('***************************************\n');
% fprintf('***************************************\n');
% fprintf('Unit Test for the Data Augmentation\n');
% 
% % important case: 1240 and 363
% id = '181';
% img = imread( [Input_images_dir, '/', id, '.png'] );
% pts = shapeGt('read_coords', [Input_images_dir, '/', id, '.pts'], NUM_LANDMARKS );
% box = shapeGt('read_box', [Input_images_dir, '/', id, '_box.pts'] );
% % display
% tmp = disp_landmarks( img, pts, struct('box',box) );
% imshow(tmp);
% de = 0;







