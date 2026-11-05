import { app } from "../../scripts/app.js";

// Mirrors parse_tag() / parse_line() / line_weights() in __init__.py.
function parseTag(text, defaultMax) {
    const i = text.lastIndexOf("#");
    const num = text.slice(i + 1).trim();
    if (i > 0 && text.slice(0, i).trim() && num !== "" && Number.isFinite(Number(num))) {
        return [text.slice(0, i).trim(), Number(num)];
    }
    return [text, defaultMax];
}

function parseLine(line) {
    const parts = line.split("|").map((t) => t.trim()).filter((t) => t);
    const defaultMax = parts.length <= 2 ? 1.5 : 1.0;
    return parts.map((t) => parseTag(t, defaultMax));
}

function sliderRange(tags, mode) {
    // Only guidance can push away from a single tag; negative weights turn the image into noise.
    if (tags.length === 1) return [mode === "guidance" ? -tags[0][1] : 0, tags[0][1]];
    if (tags.length === 2) return [-tags[0][1], tags[1][1]];
    return [0, tags.length];
}

function parseLines(text) {
    return text.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"))
        .map((line) => ({ line, tags: parseLine(line) }));
}

function rebuildSliders(node) {
    const tagsWidget = node.widgets.find((w) => w.name === "tags");
    const valuesWidget = node.widgets.find((w) => w.name === "slider_values");
    const mode = node.widgets.find((w) => w.name === "mode").value;
    let values = {};
    try { values = JSON.parse(valuesWidget.value || "{}"); } catch { }

    for (const w of node.widgets.filter((w) => w.tagSlider)) node.removeWidget(w);

    const kept = {};
    for (const { line, tags } of parseLines(tagsWidget.value)) {
        if (line in kept) continue;
        const [min, max] = sliderRange(tags, mode);
        // Keep the stored value as is, so a negative survives a trip through attention mode.
        kept[line] = values[line] ?? 0;
        const slider = node.addWidget("slider", line, Math.min(max, Math.max(min, kept[line])), (v) => {
            kept[line] = v;
            valuesWidget.value = JSON.stringify(kept);
        }, { min, max, step2: 0.05, round: 0.05, precision: 2, serialize: false });
        slider.tagSlider = true;
        if (tags.length > 2) slider.label = `${tags[0][0]} → ${tags[tags.length - 1][0]} (${tags.length})`;
    }
    valuesWidget.value = JSON.stringify(kept);
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "Anima.TagSliders",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AnimaTagSliders") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.widgets.find((w) => w.name === "slider_values").hidden = true;
            const tagsWidget = this.widgets.find((w) => w.name === "tags");
            tagsWidget.element?.addEventListener("change", () => rebuildSliders(this));
            const modeWidget = this.widgets.find((w) => w.name === "mode");
            const modeCallback = modeWidget.callback;
            modeWidget.callback = (...args) => {
                const result = modeCallback?.apply(modeWidget, args);
                rebuildSliders(this);
                return result;
            };
            rebuildSliders(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            rebuildSliders(this);
        };
    },
});
