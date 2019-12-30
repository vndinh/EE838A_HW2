clc; clear;

orig_dir = '../data/valid/HDR';
pred_dir = '../report/valid_result';

list_orig = dir(strcat(orig_dir, '/*.hdr'));
list_pred = dir(strcat(pred_dir, '/*.hdr'));

[num_valid, ~] = size(list_orig);

mpsnr = zeros(num_valid,1);

for i = 1:num_valid
  orig_name = getfield(list_orig, {i,1}, 'name');
  pred_name = getfield(list_pred, {i,1}, 'name');
  
  orig_path = strcat(orig_dir, '/', orig_name);
  pred_path = strcat(pred_dir, '/', pred_name);
  
  % Read HDR images
  orig_img = hdrread(orig_path);
  pred_img = hdrread(pred_path);
  
  max_orig = max(max(max(orig_img)));
  max_pred = max(max(max(pred_img)));
  
  orig_img = orig_img ./ max_orig;
  pred_img = pred_img ./ max_pred;
  
  [mpsnr(i), ~, ~] = mPSNR(orig_img, pred_img);
  log = sprintf(strcat(pred_name, ':', ' mPSNR = %f\n'), mpsnr(i));
  fprintf(log);
end
