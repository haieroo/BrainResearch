% =========================================================================
% RDM within- vs between-category distance quantification and visualization
% Permutation-test version
% =========================================================================
% Usage:
%   stats = F_RDM_class_distance_EM_perm(rdm_ento, rdm_mvl);
%   stats = F_RDM_class_distance_EM_perm(rdm_ento, rdm_mvl, 5000);
%
% The permutation test uses:
%   deltaD = mean(between-category distance) - mean(within-category distance)
% For color and shape, feature labels are permuted across all 24 stimuli.
% For local texture, texture labels are permuted within each global shape category
% to respect the nested stimulus design.
% =========================================================================

function F_RDM_class_distance_EM_perm(rdm_ento, rdm_mvl, nPerm)

if nargin < 3 || isempty(nPerm)
    nPerm = 5000;
end

rng(1);  % for reproducibility

%% 1. Define stimulus labels
num_stim = 24;
shape_labels   = kron([1, 2, 3]', ones(8, 1));
texture_labels = repmat(kron([1, 2]', ones(4, 1)), 3, 1);
color_labels   = repmat([1, 2, 3, 4]', 6, 1);
upper_tri = triu(true(num_stim), 1);

%% 2. Extract within- and between-category distances

% Color
[ento_color_same, ento_color_diff] = extract_distances(rdm_ento, color_labels, upper_tri);
[mvl_color_same,  mvl_color_diff]  = extract_distances(rdm_mvl,  color_labels, upper_tri);

% Global shape
[ento_shape_same, ento_shape_diff] = extract_distances(rdm_ento, shape_labels, upper_tri);
[mvl_shape_same,  mvl_shape_diff]  = extract_distances(rdm_mvl,  shape_labels, upper_tri);

% Local texture, quantified within each global shape category
[ento_tex_same, ento_tex_diff] = extract_texture_distances(rdm_ento, shape_labels, texture_labels);
[mvl_tex_same,  mvl_tex_diff]  = extract_texture_distances(rdm_mvl,  shape_labels, texture_labels);

%% 3. Compute means and SEMs for plotting
calc_sem = @(x) std(x, [], 1) ./ sqrt(numel(x));

ento_means = [mean(ento_color_same), mean(ento_color_diff), ...
              mean(ento_shape_same), mean(ento_shape_diff), ...
              mean(ento_tex_same),   mean(ento_tex_diff)];

mvl_means  = [mean(mvl_color_same),  mean(mvl_color_diff), ...
              mean(mvl_shape_same),  mean(mvl_shape_diff), ...
              mean(mvl_tex_same),    mean(mvl_tex_diff)];

ento_sems = [calc_sem(ento_color_same), calc_sem(ento_color_diff), ...
             calc_sem(ento_shape_same), calc_sem(ento_shape_diff), ...
             calc_sem(ento_tex_same),   calc_sem(ento_tex_diff)];

mvl_sems  = [calc_sem(mvl_color_same),  calc_sem(mvl_color_diff), ...
             calc_sem(mvl_shape_same),  calc_sem(mvl_shape_diff), ...
             calc_sem(mvl_tex_same),    calc_sem(mvl_tex_diff)];

%% 4. Stimulus-label permutation tests
% One-tailed test: whether between-category distances are greater than
% within-category distances.

[p_ento_color, delta_ento_color, null_ento_color] = permutation_p_full( ...
    rdm_ento, color_labels, upper_tri, nPerm);

[p_mvl_color, delta_mvl_color, null_mvl_color] = permutation_p_full( ...
    rdm_mvl, color_labels, upper_tri, nPerm);

[p_ento_shape, delta_ento_shape, null_ento_shape] = permutation_p_full( ...
    rdm_ento, shape_labels, upper_tri, nPerm);

[p_mvl_shape, delta_mvl_shape, null_mvl_shape] = permutation_p_full( ...
    rdm_mvl, shape_labels, upper_tri, nPerm);

[p_ento_tex, delta_ento_tex, null_ento_tex] = permutation_p_texture( ...
    rdm_ento, shape_labels, texture_labels, nPerm);

[p_mvl_tex, delta_mvl_tex, null_mvl_tex] = permutation_p_texture( ...
    rdm_mvl, shape_labels, texture_labels, nPerm);

p_ento = [p_ento_color, p_ento_shape, p_ento_tex];
p_mvl  = [p_mvl_color,  p_mvl_shape,  p_mvl_tex];

%% 5. Organize source data
stats = struct();
stats.description = 'High-dimensional RDM feature clustering: within- vs between-category distances';
stats.nPerm = nPerm;
stats.test = 'one-tailed stimulus-label permutation test';
stats.delta_definition = 'mean(between-category distance) - mean(within-category distance)';
stats.labels.color = color_labels;
stats.labels.shape = shape_labels;
stats.labels.texture = texture_labels;

stats.ENTO.means = ento_means;
stats.ENTO.sems  = ento_sems;
stats.ENTO.p_values = p_ento;
stats.ENTO.delta = [delta_ento_color, delta_ento_shape, delta_ento_tex];
stats.ENTO.null.color = null_ento_color;
stats.ENTO.null.shape = null_ento_shape;
stats.ENTO.null.texture = null_ento_tex;
stats.ENTO.distances.color.within = ento_color_same;
stats.ENTO.distances.color.between = ento_color_diff;
stats.ENTO.distances.shape.within = ento_shape_same;
stats.ENTO.distances.shape.between = ento_shape_diff;
stats.ENTO.distances.texture.within = ento_tex_same;
stats.ENTO.distances.texture.between = ento_tex_diff;

stats.MVL.means = mvl_means;
stats.MVL.sems  = mvl_sems;
stats.MVL.p_values = p_mvl;
stats.MVL.delta = [delta_mvl_color, delta_mvl_shape, delta_mvl_tex];
stats.MVL.null.color = null_mvl_color;
stats.MVL.null.shape = null_mvl_shape;
stats.MVL.null.texture = null_mvl_tex;
stats.MVL.distances.color.within = mvl_color_same;
stats.MVL.distances.color.between = mvl_color_diff;
stats.MVL.distances.shape.within = mvl_shape_same;
stats.MVL.distances.shape.between = mvl_shape_diff;
stats.MVL.distances.texture.within = mvl_tex_same;
stats.MVL.distances.texture.between = mvl_tex_diff;

%% 6. Plot grouped bar charts
y_ento = [ento_means(1:2); ento_means(3:4); ento_means(5:6)];
y_mvl  = [mvl_means(1:2);  mvl_means(3:4);  mvl_means(5:6)];
err_ento = [ento_sems(1:2); ento_sems(3:4); ento_sems(5:6)];
err_mvl  = [mvl_sems(1:2);  mvl_sems(3:4);  mvl_sems(5:6)];

figure('Color', 'w', 'Position', [100, 100, 900, 450]);

% ENTO
subplot(1, 2, 1);
b1 = bar(y_ento, 'grouped', 'EdgeColor', 'none');
b1(1).FaceColor = [0.4 0.6 0.8];
b1(2).FaceColor = [0.1 0.3 0.6];
hold on;
for i = 1:2
    x_pos = b1(i).XEndPoints;
    errorbar(x_pos, y_ento(:, i), err_ento(:, i), ...
        'k', 'linestyle', 'none', 'LineWidth', 1.2);
end
draw_stars(b1, y_ento, err_ento, p_ento);
set(gca, 'XTickLabel', {'Color', 'Shape', 'Texture'}, ...
    'TickDir', 'out', 'Box', 'off', 'FontSize', 15);
ylabel('Dissimilarity (1 - r)', 'FontSize', 16);
title('ENTO', 'FontSize', 18);
legend({'Within-category', 'Between-category'}, ...
    'Location', 'northwest', 'box', 'off');

% MVL
subplot(1, 2, 2);
b2 = bar(y_mvl, 'grouped', 'EdgeColor', 'none');
b2(1).FaceColor = [0.8 0.4 0.4];
b2(2).FaceColor = [0.7 0.1 0.1];
hold on;
for i = 1:2
    x_pos = b2(i).XEndPoints;
    errorbar(x_pos, y_mvl(:, i), err_mvl(:, i), ...
        'k', 'linestyle', 'none', 'LineWidth', 1.2);
end
draw_stars(b2, y_mvl, err_mvl, p_mvl);
set(gca, 'XTickLabel', {'Color', 'Shape', 'Texture'}, ...
    'TickDir', 'out', 'Box', 'off', 'FontSize', 15);
ylabel('Dissimilarity (1 - r)', 'FontSize', 16);
title('MVL', 'FontSize', 18);

%% 7. Print results
fprintf('\n=== Stimulus-label permutation test results ===\n');
fprintf('Test statistic: deltaD = mean(between) - mean(within)\n');
fprintf('Permutation number: %d\n', nPerm);
fprintf('ENTO -> Color: delta=%.4f, p=%.4g; Shape: delta=%.4f, p=%.4g; Texture: delta=%.4f, p=%.4g\n', ...
    delta_ento_color, p_ento_color, delta_ento_shape, p_ento_shape, delta_ento_tex, p_ento_tex);
fprintf('MVL  -> Color: delta=%.4f, p=%.4g; Shape: delta=%.4f, p=%.4g; Texture: delta=%.4f, p=%.4g\n', ...
    delta_mvl_color, p_mvl_color, delta_mvl_shape, p_mvl_shape, delta_mvl_tex, p_mvl_tex);

end

%% ========================================================================
% Local helper functions
% ========================================================================

function [within_dist, between_dist] = extract_distances(rdm, labels, upper_tri)
    labels = labels(:);
    same_mask = (labels == labels') & upper_tri;
    diff_mask = (labels ~= labels') & upper_tri;
    within_dist  = rdm(same_mask);
    between_dist = rdm(diff_mask);
end

function [tex_within, tex_between] = extract_texture_distances(rdm, shape_labels, texture_labels)
    tex_within = [];
    tex_between = [];
    for s = unique(shape_labels(:))'
        idx = find(shape_labels == s);
        sub_rdm = rdm(idx, idx);
        sub_tex = texture_labels(idx);
        sub_upper = triu(true(numel(idx)), 1);
        tex_within  = [tex_within;  sub_rdm((sub_tex == sub_tex') & sub_upper)];
        tex_between = [tex_between; sub_rdm((sub_tex ~= sub_tex') & sub_upper)];
    end
end

function deltaD = compute_delta(within_dist, between_dist)
    deltaD = mean(between_dist) - mean(within_dist);
end

function [p_value, obs_delta, null_delta] = permutation_p_full(rdm, labels, upper_tri, nPerm)
    [within_obs, between_obs] = extract_distances(rdm, labels, upper_tri);
    obs_delta = compute_delta(within_obs, between_obs);

    null_delta = nan(nPerm, 1);
    labels = labels(:);
    for iperm = 1:nPerm
        perm_labels = labels(randperm(numel(labels)));
        [within_perm, between_perm] = extract_distances(rdm, perm_labels, upper_tri);
        null_delta(iperm) = compute_delta(within_perm, between_perm);
    end

    p_value = (sum(null_delta >= obs_delta) + 1) / (nPerm + 1);
end

function [p_value, obs_delta, null_delta] = permutation_p_texture(rdm, shape_labels, texture_labels, nPerm)
    [within_obs, between_obs] = extract_texture_distances(rdm, shape_labels, texture_labels);
    obs_delta = compute_delta(within_obs, between_obs);

    null_delta = nan(nPerm, 1);
    for iperm = 1:nPerm
        perm_tex = texture_labels(:);
        for s = unique(shape_labels(:))'
            idx = find(shape_labels == s);
            perm_tex(idx) = perm_tex(idx(randperm(numel(idx))));
        end
        [within_perm, between_perm] = extract_texture_distances(rdm, shape_labels, perm_tex);
        null_delta(iperm) = compute_delta(within_perm, between_perm);
    end

    p_value = (sum(null_delta >= obs_delta) + 1) / (nPerm + 1);
end

function draw_stars(b, y, err, p_vals)
    y_top = max(y(:) + err(:));
    if y_top <= 0
        y_top = 1;
    end
    step = 0.08 * y_top;
    line_height = 0.03 * y_top;

    hold on;
    for k = 1:numel(p_vals)
        if p_vals(k) < 0.05
            x1 = b(1).XEndPoints(k);
            x2 = b(2).XEndPoints(k);
            y_star = max(y(k, :) + err(k, :)) + step;

            plot([x1, x1, x2, x2], ...
                 [y_star, y_star + line_height, y_star + line_height, y_star], ...
                 'k', 'LineWidth', 1.2);

            text(mean([x1, x2]), y_star + line_height * 1.2, p_to_star(p_vals(k)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 16, 'FontWeight', 'bold');
        end
    end

    ylim_current = ylim;
    ylim([ylim_current(1), max(ylim_current(2), y_top + 3 * step)]);
end

function s = p_to_star(p)
    if p < 0.001
        s = '***';
    elseif p < 0.01
        s = '**';
    elseif p < 0.05
        s = '*';
    else
        s = 'n.s.';
    end
end
