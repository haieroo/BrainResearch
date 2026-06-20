function plot_all_eta2_sparseness_shared_axes(bin_spike_anova_ENTO, bin_spike_anova_MVL)

class_defs = {
    2, 'ColorSelective',         'Color',          'Color';
    1, 'ShapeSelective',         'Shape',          'Shape';
    3, 'TextureSensitive',       'Texture',        'TextureNestedMax';
    4, 'MultiFeatureModulation', 'Multi-feature',  'Stimulus24';
    8, 'NonSelective',           'Non-selective',  'Stimulus24'
};

metrics = {'Eta2', 'Sparseness'};
metric_labels = {'\eta^2', 'Sparseness'};

ento_color = [0.15 0.60 0.95];
mvl_color  = [0.85 0.33 0.10];

figure('Units', 'centimeters', ...
    'Position', [3, 3, 28, 14], ...
    'Color', 'w');

set(gcf, 'DefaultAxesFontName', 'Arial');
set(gcf, 'DefaultTextFontName', 'Arial');
set(gcf, 'Renderer', 'painters');

tiledlayout(2, 1, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

for m = 1:numel(metrics)
    nexttile;
    hold on;

    metric_name = metrics{m};
    metric_label = metric_labels{m};

    all_values = [];
    all_groups = [];
    all_positions = [];
    all_box_colors = [];

    p_values = nan(size(class_defs, 1), 1);
    max_each_class = nan(size(class_defs, 1), 1);

    for c = 1:size(class_defs, 1)
        class_id = class_defs{c, 1};
        class_name = class_defs{c, 2};
        grouping_name = class_defs{c, 4};

        y_ento = collect_metric(bin_spike_anova_ENTO, class_id, class_name, grouping_name, metric_name);
        y_mvl  = collect_metric(bin_spike_anova_MVL,  class_id, class_name, grouping_name, metric_name);

        y_ento = y_ento(~isnan(y_ento));
        y_mvl  = y_mvl(~isnan(y_mvl));

        pos_ento = c - 0.18;
        pos_mvl  = c + 0.18;

        g_ento = 2*c - 1;
        g_mvl  = 2*c;

        all_values = [all_values; y_ento(:); y_mvl(:)];
        all_groups = [all_groups; g_ento * ones(numel(y_ento), 1); g_mvl * ones(numel(y_mvl), 1)];
        all_positions = [all_positions, pos_ento, pos_mvl];
        all_box_colors = [all_box_colors; ento_color; mvl_color];

        rng(1 + c + m * 10);
        x_ento = pos_ento + (rand(numel(y_ento), 1) - 0.5) * 0.10;
        x_mvl  = pos_mvl  + (rand(numel(y_mvl), 1)  - 0.5) * 0.10;

        scatter(x_ento, y_ento, 20, ...
            'MarkerFaceColor', ento_color, ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', 0.58);

        scatter(x_mvl, y_mvl, 20, ...
            'MarkerFaceColor', mvl_color, ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', 0.58);

        if numel(y_ento) > 1 && numel(y_mvl) > 1
            p_values(c) = ranksum(y_ento, y_mvl);
        end

        if ~isempty([y_ento(:); y_mvl(:)])
            max_each_class(c) = max([y_ento(:); y_mvl(:)]);
        end
    end

    boxplot(all_values, all_groups, ...
        'Positions', all_positions, ...
        'Labels', repmat({''}, 1, numel(all_positions)), ...
        'Symbol', '', ...
        'Widths', 0.22, ...
        'Colors', [0.25 0.25 0.25]);

    set(findobj(gca, 'Tag', 'Box'), 'LineWidth', 1.0);
    set(findobj(gca, 'Tag', 'Median'), 'LineWidth', 1.2);

    y_min = min(all_values);
    y_max = max(all_values);
    y_range = max(y_max - y_min, eps);

    for c = 1:size(class_defs, 1)
        pos_ento = c - 0.18;
        pos_mvl  = c + 0.18;

        y_sig = max_each_class(c) + 0.10 * y_range;

        plot([pos_ento, pos_mvl], [y_sig, y_sig], ...
            'k-', 'LineWidth', 0.9);

        text(c, y_sig + 0.035 * y_range, p_to_star(p_values(c)), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 10, ...
            'FontName', 'Arial');
    end

    xlim([0.5, size(class_defs, 1) + 0.5]);
    ylim([max(0, y_min - 0.06 * y_range), y_max + 0.24 * y_range]);

    xticks(1:size(class_defs, 1));
    xticklabels(class_defs(:, 3));

    ylabel(metric_label, ...
        'FontSize', 13, ...
        'FontName', 'Arial');

    set(gca, ...
        'FontSize', 11, ...
        'LineWidth', 1.1, ...
        'Box', 'off', ...
        'TickDir', 'out', ...
        'FontName', 'Arial');

    if m == 1
        legend_handles = [
            scatter(nan, nan, 30, 'MarkerFaceColor', ento_color, 'MarkerEdgeColor', 'none')
            scatter(nan, nan, 30, 'MarkerFaceColor', mvl_color,  'MarkerEdgeColor', 'none')
        ];

        legend(legend_handles, {'ENTO', 'MVL'}, ...
            'Location', 'northeastoutside', ...
            'Box', 'off', ...
            'FontSize', 10);
    end

    hold off;
end


function y = collect_metric(bin_spike_anova, class_id, class_name, grouping_name, metric_name)

y = [];

for i = 1:numel(bin_spike_anova)
    if ~isfield(bin_spike_anova(i), 'per_neuron_table')
        continue;
    end

    T = bin_spike_anova(i).per_neuron_table;

    if isempty(T)
        continue;
    end

    if ismember('ClassName', T.Properties.VariableNames)
        idx_class = strcmp(T.ClassName, class_name);
    else
        idx_class = T.ClassID == class_id;
    end

    idx = idx_class & strcmp(T.StimulusGrouping, grouping_name);

    if any(idx)
        y = [y; T.(metric_name)(idx)];
    end
end

end

function s = p_to_star(p)

if isnan(p)
    s = 'n.s.';
elseif p < 0.001
    s = '***';
elseif p < 0.01
    s = '**';
elseif p < 0.05
    s = '*';
else
    s = 'n.s.';
end

end




end