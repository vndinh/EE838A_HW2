clear; clc;

for i = 44:46
  if i == 46
    fprintf('%d \n', i);
    hdr_name = sprintf('../data/valid/HDR/C%02d_HDR.hdr', i);
    hdr = hdrread(hdr_name);
    
    R = hdr(:,:,1);
    G = hdr(:,:,2);
    B = hdr(:,:,3);
    
    R(~any(R,2),:) = [];
    G(~any(G,2),:) = [];
    B(~any(B,2),:) = [];
    
    [m, n] = size(R);
    new_hdr = zeros(m,n,3);
    new_hdr(:,:,1) = R;
    new_hdr(:,:,2) = G;
    new_hdr(:,:,3) = B;
    
    Hth = exposure(new_hdr);
    hdr = hdr./(Hth);

    sigma_mean = 0.6;
    sigma_std = 0.1;

    sigma = min(max(sigma_std*randn(1,1)+sigma_mean, 0.0), 5.0);

    n_mean = 0.9;
    n_std = 0.1;

    n = min(max(n_std*randn(1,1)+n_mean, 0.2), 2.5);

    ldr = (1+sigma).*(hdr.^(n))./(hdr.^(n)+sigma);

    ldr = ldr*255;
    ldr = min(max(ldr, 0), 255);
    ldr = uint8(ldr);

    ldr_name = sprintf('../data/valid/LDR/C%02d_LDR.png', i);
    imwrite(ldr, ldr_name);
  end
end

