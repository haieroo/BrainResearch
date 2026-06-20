% =========================================================================
% 顶刊级：核心特征分类分组柱状图 (Grouped Bar Chart - 纯净版)
% X轴：颜色 | 形状 | 纹理 | 交互 (去除视觉响应组，突出特征转化)
% =========================================================================

% 1. 你的绝对数量数据 (分类 1=形状, 2=颜色, 3=纹理, 4=组和)
counts_ENTO = [34, 36, 10, 29, 55];
counts_MVL  = [22, 26, 16, 39,39];

% =========================================================================
% ★ 策略选择：使用什么作为分母来计算百分比？ ★
% 策略 A：使用所有记录到的细胞作为分母 (保留你之前的比例)
total_ENTO = 148; % sum(counts_ENTO)
total_MVL  = 127;  % sum(counts_MVL)
% =========================================================================

% 2. 核心计算：算出 4 个核心类别的百分比 (%)
% (1) 纯颜色 (Color, 类 2)
color_ENTO = counts_ENTO(2) / total_ENTO * 100;
color_MVL  = counts_MVL(2)  / total_MVL  * 100;

% (2) 纯形状 (Shape, 类 1)
shape_ENTO = counts_ENTO(1) / total_ENTO * 100;
shape_MVL  = counts_MVL(1)  / total_MVL  * 100;

% (3) 纯纹理 (Texture, 类 3)
texture_ENTO = counts_ENTO(3) / total_ENTO * 100;
texture_MVL  = counts_MVL(3)  / total_MVL  * 100;

% (4) 交互 (Interactions: 4)
int_ENTO = sum(counts_ENTO(4)) / total_ENTO * 100;
int_MVL  = sum(counts_MVL(4))  / total_MVL  * 100;

% 组装绘图矩阵 (4行 x 2列)
data_matrix = [
    color_ENTO,    color_MVL;
    shape_ENTO,    shape_MVL;
    texture_ENTO,  texture_MVL;
    int_ENTO,      int_MVL
];

% =========================================================================
% 3. 绘图与美化
% =========================================================================
figure('Position', [150, 150, 700, 500], 'Color', 'w');
% ★ 强制将当前图窗的默认字体全部设为 Arial ★
set(gcf, 'DefaultAxesFontName', 'Arial');
set(gcf, 'DefaultTextFontName', 'Arial');
    % 将画布大小拓展 方便保存成PDF文件
set(gcf, 'Units', 'Inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);

hold on;

b = bar(data_matrix, 'grouped', 'EdgeColor', 'none', 'BarWidth', 0.85);

% 极简高级配色 (ENTO 蓝, MVL 绿)
b(1).FaceColor = [0.15, 0.60, 0.95]; 
b(2).FaceColor = [0.85, 0.33, 0.10]; % Standard MATLAB Orange/Red (looks better than 1 0 0)

% --- 坐标轴与标签 ---, 'FontWeight', 'bold', 'FontWeight', 'bold'
xticks(1:4);
xticklabels([]);
xtickangle(0);

x_labels = {
    {'Color', 'Selective'}
    {'Shape', 'Selective'}
    {'Texture', 'Sensitive'}
    {'Multi-feature', 'Modulation'}
};

yl = ylim;
y_text = yl(1) - 0.06 * range(yl);

for k = 1:4
    text(k, y_text, x_labels{k}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 18, ...
        'FontName', 'Arial', ...
        'Clipping', 'off');
end

set(gca, 'Position', [0.16, 0.20, 0.78, 0.72]);

ylabel('Percentage (%)', 'FontSize', 18);

set(gca, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.5, 'FontSize', 18);
ylim([0, max(data_matrix(:)) * 1.25]); 

% --- 添加图例 ---
lgd = legend({'ENTO', 'MVL'}, 'Location', 'northeast', 'FontSize', 14);
legend boxoff;

hold off;