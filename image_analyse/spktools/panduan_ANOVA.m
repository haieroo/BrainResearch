function [neuron_labels, class_id] = panduan_ANOVA3(final_feature_matrix)

nChannels = size(final_feature_matrix, 1);
neuron_labels = cell(nChannels, 1);

% 每个位点最多保存 3 个标签编号，不足补 0
% 1 = Shape, 2 = Color, 3 = Texture, 4 = Multi-dimensional modulation, 8 = Non-selective
class_id = zeros(nChannels, 3);

alpha = 0.05;
threshold_sub = alpha / 3;

for ch = 1:nChannels
    p = final_feature_matrix(ch, :);

    is_shape = p(1) < alpha;
    is_color = p(2) < alpha;
    is_texture = p(3) < alpha;

    is_shape_color_int = p(4) < alpha;
    is_texture_color_int = p(5) < threshold_sub;
    is_morph_color_int = p(6) < alpha;
    is_morph = p(7) < alpha;

    has_interaction = is_shape_color_int || ...
                      is_texture_color_int || ...
                      is_morph_color_int;

    has_shape_texture_conjunction = is_morph && ~is_texture;

    is_multidimensional_modulation = has_interaction || has_shape_texture_conjunction;

    labels = {};
    ids = [];

    if is_multidimensional_modulation
        ids = 4;
        labels{end + 1} = 'Multi-dimensional modulation';

    else
        if is_shape
            ids(end + 1) = 1;
            labels{end + 1} = 'Global shape main effect';
        end
        if is_color
            ids(end + 1) = 2;
            labels{end + 1} = 'Color main effect';
        end
        if is_texture
            ids(end + 1) = 3;
            labels{end + 1} = 'Local texture selectivity';
        end

        if isempty(ids)
            ids = 8;
            labels{end + 1} = 'Non-selective';
        end
    end

    class_id(ch, 1:numel(ids)) = ids;
    neuron_labels{ch} = strjoin(labels, '; ');
end

end