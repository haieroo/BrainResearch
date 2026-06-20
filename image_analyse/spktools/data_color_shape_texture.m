function [colordata, shapedata,texturedata] = data_color_shape_texture(bin_spike_after)
% picture1按照颜色形状纹理划分选择实验数据 
     bluedata=cat(2,bin_spike_after(:,:,:,1),bin_spike_after(:,:,:,5),bin_spike_after(:,:,:,9),bin_spike_after(:,:,:,13),bin_spike_after(:,:,:,17),bin_spike_after(:,:,:,21));
     greendata=cat(2,bin_spike_after(:,:,:,2),bin_spike_after(:,:,:,6),bin_spike_after(:,:,:,10),bin_spike_after(:,:,:,14),bin_spike_after(:,:,:,18),bin_spike_after(:,:,:,22));  
     reddata=cat(2,bin_spike_after(:,:,:,3),bin_spike_after(:,:,:,7),bin_spike_after(:,:,:,11),bin_spike_after(:,:,:,15),bin_spike_after(:,:,:,19),bin_spike_after(:,:,:,23));
     yellowdata=cat(2,bin_spike_after(:,:,:,4),bin_spike_after(:,:,:,8),bin_spike_after(:,:,:,12),bin_spike_after(:,:,:,16),bin_spike_after(:,:,:,20),bin_spike_after(:,:,:,24));
     colordata=cat(4,bluedata,greendata,reddata,yellowdata);
     % 形状
     S1_data=cat(2,bin_spike_after(:,:,:,1),bin_spike_after(:,:,:,2),bin_spike_after(:,:,:,3),bin_spike_after(:,:,:,4),bin_spike_after(:,:,:,5),bin_spike_after(:,:,:,6),bin_spike_after(:,:,:,7),bin_spike_after(:,:,:,8));
     S2_data=cat(2,bin_spike_after(:,:,:,9),bin_spike_after(:,:,:,10),bin_spike_after(:,:,:,11),bin_spike_after(:,:,:,12),bin_spike_after(:,:,:,13),bin_spike_after(:,:,:,14),bin_spike_after(:,:,:,15),bin_spike_after(:,:,:,16));  
     S3_data=cat(2,bin_spike_after(:,:,:,17),bin_spike_after(:,:,:,18),bin_spike_after(:,:,:,19),bin_spike_after(:,:,:,20),bin_spike_after(:,:,:,21),bin_spike_after(:,:,:,22),bin_spike_after(:,:,:,23),bin_spike_after(:,:,:,24));
     shapedata=cat(4,S1_data,S2_data,S3_data);
     % 纹理
     T1_data=cat(2,bin_spike_after(:,:,:,1),bin_spike_after(:,:,:,2),bin_spike_after(:,:,:,3),bin_spike_after(:,:,:,4));
     T2_data=cat(2,bin_spike_after(:,:,:,5),bin_spike_after(:,:,:,6),bin_spike_after(:,:,:,7),bin_spike_after(:,:,:,8));
     T3_data=cat(2,bin_spike_after(:,:,:,9),bin_spike_after(:,:,:,10),bin_spike_after(:,:,:,11),bin_spike_after(:,:,:,12));  
     T4_data=cat(2,bin_spike_after(:,:,:,13),bin_spike_after(:,:,:,14),bin_spike_after(:,:,:,15),bin_spike_after(:,:,:,16));  
     T5_data=cat(2,bin_spike_after(:,:,:,17),bin_spike_after(:,:,:,18),bin_spike_after(:,:,:,19),bin_spike_after(:,:,:,20));
     T6_data=cat(2,bin_spike_after(:,:,:,21),bin_spike_after(:,:,:,22),bin_spike_after(:,:,:,23),bin_spike_after(:,:,:,24));
     texturedata=cat(4,T1_data,T2_data,T3_data,T4_data,T5_data,T6_data);
end