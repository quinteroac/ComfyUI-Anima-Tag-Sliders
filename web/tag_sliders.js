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

// Anima Tag Curves: one strength curve per tag line (up to 4) over the sampling steps, stored in the hidden `curves` JSON.
const MAX_CURVE_TAGS = 4;
const CURVE_COLORS = ["#4fa3ff", "#ff8a3d", "#5fd068", "#e05fd0"];
const PAD = { left: 34, right: 10, top: 26, bottom: 20 };
// Curve shapes for the right-click menu: strength at t in [0, 1] of the steps, as a fraction of max_strength.
// [before, after] is a jump at the first third of the steps.
const SMOOTH = (t) => t * t * (3 - 2 * t);
const BELL = (t) => Math.exp(-(((t - 0.5) / 0.18) ** 2) / 2);
const CURVE_SHAPES = [
    ["Flat (as written)", null],
    ["Ramp up", (t) => t],
    ["Ramp down", (t) => 1 - t],
    ["Ease in", (t) => t * t],
    ["Ease out", (t) => 1 - (1 - t) ** 2],
    ["S-curve up", SMOOTH],
    ["S-curve down", (t) => 1 - SMOOTH(t)],
    ["Cosine fade", (t) => Math.cos((t * Math.PI) / 2)],
    ["Bell (peak in the middle)", BELL],
    ["Valley (dip in the middle)", (t) => 1 - BELL(t)],
    ["Early only (first third)", [1, 0]],
    ["Late only (last two thirds)", [0, 1]],
];

class CurveEditor {
    constructor(node) {
        this.node = node;
        this.active = 0;
        this.drag = null;
        this.hover = null;
        this.root = document.createElement("div");
        this.root.style.cssText = "width:100%;height:100%;position:relative;";
        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = "width:100%;height:100%;display:block;cursor:crosshair;touch-action:none;";
        this.root.appendChild(this.canvas);
        new ResizeObserver(() => this.draw()).observe(this.root);
        this.canvas.addEventListener("pointerdown", (e) => this.onDown(e));
        this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
        this.canvas.addEventListener("pointerup", (e) => this.onUp(e));
        this.canvas.addEventListener("pointerleave", () => { this.hover = null; this.draw(); });
        this.canvas.addEventListener("dblclick", (e) => this.onDouble(e));
        this.canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); this.openMenu(e); });
    }

    widget(name) { return this.node.widgets.find((w) => w.name === name); }
    get steps() { return Math.max(1, this.widget("steps").value); }
    get max() { return this.widget("max_strength").value; }
    get low() { return this.widget("mode").value === "guidance" ? -this.max : 0; }
    get tags() {
        const lines = this.widget("tags").value.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
        return [...new Set(lines)].slice(0, MAX_CURVE_TAGS);
    }
    get extra() {
        const lines = this.widget("tags").value.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
        return new Set(lines).size - MAX_CURVE_TAGS;
    }

    load() {
        try { return JSON.parse(this.widget("curves").value || "{}"); } catch { return {}; }
    }

    save(curves) { this.widget("curves").value = JSON.stringify(curves); }

    // Keeps one curve per current tag. A renamed tag keeps the curve of the line it replaced, a new one starts flat at 1 (the tag as written).
    sync() {
        const stored = this.load();
        const before = Object.keys(stored);
        const tags = this.tags;
        const curves = {};
        tags.forEach((tag, i) => {
            const replaced = before[i] !== undefined && !tags.includes(before[i]) ? stored[before[i]] : undefined;
            curves[tag] = stored[tag] ?? replaced ?? [[0, 1], [this.steps - 1, 1]];
        });
        this.save(curves);
        this.active = Math.min(this.active, Math.max(0, this.tags.length - 1));
        this.draw();
    }

    // Moves every point to the same place on a new step axis.
    rescaleSteps(before) {
        const scale = (this.steps - 1) / Math.max(before - 1, 1);
        const curves = this.load();
        for (const tag in curves) curves[tag] = curves[tag].map(([x, y]) => [Math.round(x * scale), y]);
        this.save(curves);
        this.draw();
    }

    area() {
        const r = this.canvas.getBoundingClientRect();
        return { w: r.width, h: r.height, x0: PAD.left, y0: PAD.top, x1: r.width - PAD.right, y1: r.height - PAD.bottom };
    }

    toScreen([x, y], a) {
        const xs = a.x0 + (x / Math.max(this.steps - 1, 1)) * (a.x1 - a.x0);
        const ys = a.y1 - ((y - this.low) / (this.max - this.low)) * (a.y1 - a.y0);
        return [xs, ys];
    }

    fromScreen(px, py, a) {
        const x = Math.round(((px - a.x0) / (a.x1 - a.x0)) * Math.max(this.steps - 1, 1));
        const y = this.low + ((a.y1 - py) / (a.y1 - a.y0)) * (this.max - this.low);
        return [Math.min(this.steps - 1, Math.max(0, x)), Math.round(Math.min(this.max, Math.max(this.low, y)) * 100) / 100];
    }

    pos(e) {
        const r = this.canvas.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top];
    }

    legend(px, py) {
        if (py > PAD.top - 4) return -1;
        return this.legendBoxes.findIndex(([x0, x1]) => px >= x0 && px <= x1);
    }

    nearest(px, py, a) {
        const tag = this.tags[this.active];
        if (tag === undefined) return -1;
        const points = this.load()[tag];
        let best = -1, dist = 8;
        points.forEach((p, i) => {
            const [sx, sy] = this.toScreen(p, a);
            const d = Math.hypot(sx - px, sy - py);
            if (d < dist) { best = i; dist = d; }
        });
        return best;
    }

    onDown(e) {
        e.stopPropagation();
        const [px, py] = this.pos(e);
        const picked = this.legend(px, py);
        if (picked >= 0) { this.active = picked; this.draw(); return; }
        const tag = this.tags[this.active];
        if (tag === undefined || e.button !== 0) return;
        const a = this.area();
        let i = this.nearest(px, py, a);
        const curves = this.load();
        if (i < 0) {
            const point = this.fromScreen(px, py, a);
            const points = curves[tag].filter(([x]) => x !== point[0]);
            points.push(point);
            points.sort((p, q) => p[0] - q[0]);
            curves[tag] = points;
            this.save(curves);
            i = points.indexOf(point);
        }
        this.drag = i;
        this.canvas.setPointerCapture(e.pointerId);
        this.draw();
    }

    onMove(e) {
        const [px, py] = this.pos(e);
        const a = this.area();
        if (this.drag === null) {
            this.hover = this.fromScreen(px, py, a);
            this.draw();
            return;
        }
        e.stopPropagation();
        const tag = this.tags[this.active];
        const curves = this.load();
        const points = curves[tag];
        const point = this.fromScreen(px, py, a);
        // A point can't pass its neighbours on the step axis.
        const lo = this.drag > 0 ? points[this.drag - 1][0] + 1 : 0;
        const hi = this.drag < points.length - 1 ? points[this.drag + 1][0] - 1 : this.steps - 1;
        point[0] = Math.min(hi, Math.max(lo, point[0]));
        points[this.drag] = point;
        this.save(curves);
        this.hover = point;
        this.draw();
    }

    onUp(e) {
        if (this.drag === null) return;
        this.drag = null;
        this.canvas.releasePointerCapture(e.pointerId);
        this.draw();
    }

    // Points of a shape over the step axis: up to 11 evenly spaced for a smooth shape, the ends and both sides of the jump for a step.
    shapePoints(shape) {
        const last = this.steps - 1;
        const y = (v) => Math.round(v * this.max * 100) / 100;
        if (shape === null) return [[0, Math.min(1, this.max)], [last, Math.min(1, this.max)]];
        if (Array.isArray(shape)) {
            const jump = Math.min(Math.round(last / 3), last - 1);
            if (jump < 0) return [[0, y(shape[1])]];
            return [[0, y(shape[0])], [jump, y(shape[0])], [jump + 1, y(shape[1])], [last, y(shape[1])]].filter((p, i, a) => i === 0 || p[0] > a[i - 1][0]);
        }
        const n = Math.min(this.steps, 11);
        const xs = [...new Set(Array.from({ length: n }, (_, i) => Math.round((i * last) / Math.max(n - 1, 1))))];
        return xs.map((x) => [x, y(shape(last ? x / last : 0))]);
    }

    openMenu(e) {
        const tag = this.tags[this.active];
        if (tag === undefined) return;
        const [px, py] = this.pos(e);
        const point = this.nearest(px, py, this.area());
        document.querySelector(".curve-shape-menu")?.remove();
        const menu = document.createElement("div");
        menu.className = "curve-shape-menu";
        menu.style.cssText = `position:fixed;left:${e.clientX}px;top:${e.clientY}px;z-index:10000;background:#222;border:1px solid #555;border-radius:4px;padding:4px 0;font:12px sans-serif;color:#ddd;min-width:190px;box-shadow:0 4px 12px #0008;`;
        const item = (label, action, color) => {
            const row = document.createElement("div");
            row.textContent = label;
            row.style.cssText = `padding:4px 12px;cursor:${action ? "pointer" : "default"};color:${color ?? (action ? "#ddd" : "#888")};white-space:nowrap;`;
            if (action) {
                row.onmouseenter = () => { row.style.background = "#3a3a3a"; };
                row.onmouseleave = () => { row.style.background = ""; };
                row.onpointerdown = (ev) => { ev.stopPropagation(); close(); action(); };
            }
            menu.appendChild(row);
        };
        const outside = (ev) => { if (!menu.contains(ev.target)) close(); };
        const close = () => { menu.remove(); document.removeEventListener("pointerdown", outside, true); };
        item(`Curve shape for "${tag}"`, null, CURVE_COLORS[this.active]);
        for (const [label, shape] of CURVE_SHAPES) {
            item(label, () => {
                const curves = this.load();
                curves[tag] = this.shapePoints(shape);
                this.save(curves);
                this.draw();
            });
        }
        if (point >= 0 && this.load()[tag].length > 1) item("Delete point", () => this.onDouble(e), "#e88");
        document.body.appendChild(menu);
        // Keep the menu on screen.
        const r = menu.getBoundingClientRect();
        if (r.bottom > window.innerHeight) menu.style.top = `${Math.max(0, window.innerHeight - r.height)}px`;
        if (r.right > window.innerWidth) menu.style.left = `${Math.max(0, window.innerWidth - r.width)}px`;
        setTimeout(() => document.addEventListener("pointerdown", outside, true));
    }

    onDouble(e) {
        const [px, py] = this.pos(e);
        const tag = this.tags[this.active];
        const i = this.nearest(px, py, this.area());
        if (tag === undefined || i < 0) return;
        const curves = this.load();
        if (curves[tag].length <= 1) return;
        curves[tag].splice(i, 1);
        this.save(curves);
        this.draw();
    }

    draw() {
        const dpr = window.devicePixelRatio || 1;
        const a = this.area();
        if (a.w <= 0 || a.h <= 0) return;
        this.canvas.width = Math.round(a.w * dpr);
        this.canvas.height = Math.round(a.h * dpr);
        const ctx = this.canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, a.w, a.h);
        ctx.fillStyle = "#1b1b1b";
        ctx.fillRect(a.x0, a.y0, a.x1 - a.x0, a.y1 - a.y0);
        ctx.font = "10px sans-serif";
        ctx.lineWidth = 1;

        // Grid: every strength unit, 1 dashed (the tag as written), and the step ticks.
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        const unit = this.max - this.low > 8 ? Math.ceil((this.max - this.low) / 8) : 0.5;
        for (let y = Math.ceil(this.low / unit) * unit; y <= this.max + 1e-9; y += unit) {
            const [, sy] = this.toScreen([0, y], a);
            ctx.strokeStyle = y === 0 ? "#555" : "#2c2c2c";
            ctx.beginPath(); ctx.moveTo(a.x0, sy); ctx.lineTo(a.x1, sy); ctx.stroke();
            ctx.fillStyle = "#888";
            ctx.fillText(+y.toFixed(2), a.x0 - 4, sy);
        }
        if (1 >= this.low && 1 <= this.max) {
            const [, sy] = this.toScreen([0, 1], a);
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = "#777";
            ctx.beginPath(); ctx.moveTo(a.x0, sy); ctx.lineTo(a.x1, sy); ctx.stroke();
            ctx.setLineDash([]);
        }
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        const last = Math.max(this.steps - 1, 1);
        const every = Math.max(1, Math.ceil(this.steps / 10));
        for (let x = 0; x <= last; x += every) {
            const [sx] = this.toScreen([x, 0], a);
            ctx.fillStyle = "#888";
            ctx.fillText(x, sx, a.y1 + 4);
        }

        // Curves, the active one last and with its points.
        const curves = this.load();
        const tags = this.tags;
        const order = tags.map((_, i) => i).filter((i) => i !== this.active).concat(tags.length ? [this.active] : []);
        for (const i of order) {
            const points = curves[tags[i]] ?? [];
            if (!points.length) continue;
            const clamp = ([x, y]) => [x, Math.min(this.max, Math.max(this.low, y))];
            const screen = points.map((p) => this.toScreen(clamp(p), a));
            ctx.strokeStyle = CURVE_COLORS[i];
            ctx.globalAlpha = i === this.active ? 1 : 0.45;
            ctx.lineWidth = i === this.active ? 2 : 1.5;
            ctx.beginPath();
            ctx.moveTo(a.x0, screen[0][1]);
            for (const [sx, sy] of screen) ctx.lineTo(sx, sy);
            ctx.lineTo(a.x1, screen[screen.length - 1][1]);
            ctx.stroke();
            if (i === this.active) {
                ctx.fillStyle = CURVE_COLORS[i];
                for (const [sx, sy] of screen) { ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill(); }
            }
            ctx.globalAlpha = 1;
        }

        // Legend: click a tag to edit its curve.
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.font = "11px sans-serif";
        this.legendBoxes = [];
        let lx = a.x0;
        tags.forEach((tag, i) => {
            const label = tag.length > 18 ? tag.slice(0, 17) + "…" : tag;
            const w = ctx.measureText(label).width + 16;
            ctx.globalAlpha = i === this.active ? 1 : 0.6;
            ctx.fillStyle = CURVE_COLORS[i];
            ctx.fillRect(lx, 9, 8, 8);
            ctx.fillStyle = i === this.active ? "#ddd" : "#999";
            ctx.fillText(label, lx + 11, 13);
            this.legendBoxes.push([lx, lx + w]);
            lx += w + 6;
        });
        ctx.globalAlpha = 1;
        if (this.extra > 0) {
            ctx.fillStyle = "#e66";
            ctx.fillText(`+${this.extra} tags over ${MAX_CURVE_TAGS}`, lx, 13);
        }
        if (this.hover) {
            ctx.textAlign = "right";
            ctx.fillStyle = "#aaa";
            ctx.fillText(`step ${this.hover[0]}  ×${this.hover[1].toFixed(2)}`, a.x1, 13);
        }
    }
}

function onWidgetChange(node, name, update) {
    const widget = node.widgets.find((w) => w.name === name);
    const callback = widget.callback;
    widget.callback = function () {
        const before = widget.lastValue;
        const r = callback?.apply(this, arguments);
        update(before);
        widget.lastValue = widget.value;
        return r;
    };
    widget.lastValue = widget.value;
}

app.registerExtension({
    name: "Anima.TagCurves",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AnimaTagCurves") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.widgets.find((w) => w.name === "curves").hidden = true;
            const editor = new CurveEditor(this);
            this.curveEditor = editor;
            const widget = this.addDOMWidget("curve_editor", "ANIMA_TAG_CURVES", editor.root, { serialize: false, getMinHeight: () => 220 });
            widget.serialize = false;
            this.widgets.find((w) => w.name === "tags").element?.addEventListener("change", () => editor.sync());
            onWidgetChange(this, "steps", (before) => editor.rescaleSteps(before));
            onWidgetChange(this, "max_strength", () => editor.draw());
            onWidgetChange(this, "mode", () => editor.draw());
            editor.sync();
            if (this.size[0] < 380) this.setSize([380, this.size[1]]);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            this.widgets.find((w) => w.name === "steps").lastValue = this.widgets.find((w) => w.name === "steps").value;
            this.curveEditor.sync();
        };
    },
});
