clear; clc;

addpath('../hdrvdp/hdrvdp-2.2.1');

hdr_ref = hdrread('../data/valid/HDR/C45_HDR.hdr');
max_ref = max(max(max(hdr_ref)));
hdr_ref = hdr_ref ./ max_ref;

[h, w, d] = size(hdr_ref);
[Y_ref, Cb_ref, Cr_ref] = srgb2ycbcr_HDR(hdr_ref);

hdr_dist = hdrread('../report/valid_result/pred_C45_HDR.hdr');
max_dist = max(max(max(hdr_dist)));
hdr_dist = hdr_dist ./ max_dist;

[Y_dist, Cb_dist, Cr_dist] = srgb2ycbcr_HDR(hdr_dist);

res = hdrvdp(Y_ref, Y_dist, 'luminance', 30);
clf;
imshow(hdrvdp_visualize(res.P_map, Y_dist));


function [Y, Cb, Cr] = srgb2ycbcr_HDR(hdr)

m1=(2610/4096)*0.25;
m2=(2523/4096)*128;
c1=3424/4096;
c2=(2413/4096)*32;
c3=(2392/4096)*32;


hdri=double(hdr);
hdri = max(0,min(hdri,10000));
[r,c,~]=size(hdri);
if mod(r,2)==1
    hdri=hdri(1:r-1,:,:);
end
if mod(c,2)==1
    hdri=hdri(:,1:c-1,:);
end

%coding TF
Clip_hdri=max(0,min(hdri/10000,1));
PQTF_hdri=((c1+c2*(Clip_hdri.^m1))./(1+c3*(Clip_hdri.^m1))).^m2;

%R'G'B to Y'CbCr
Y=0.2627*PQTF_hdri(:,:,1)+0.6780*PQTF_hdri(:,:,2)+0.0593*PQTF_hdri(:,:,3);
Cb=(PQTF_hdri(:,:,3)-Y)/1.8814;
Cr=(PQTF_hdri(:,:,1)-Y)/1.4746;

end

