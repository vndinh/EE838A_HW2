function Hth = exposure(hdr)
  hdr = double(hdr);
  hdr = max(10^(-5), min(hdr, 10^(5)));
  hdr_lum = mean(hdr, 3);
  sort_hdr_lum = sort(hdr_lum(:));
  th = min(max(randn(1,1)*0.05+0.9, 0.85), 0.95);
  th = floor(length(sort_hdr_lum)*th);
  Hth = sort_hdr_lum(th);
end