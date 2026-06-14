"""
Local OKF (Open Knowledge Format) bundle visualizer.

Reads an OKF bundle directory and writes a single **self-contained** HTML file
that renders the bundle as an interactive knowledge graph — modeled on Google's
``knowledge-catalog`` OKF viewer (``okf/.../viewer/static/viz.js``), but
re-implemented natively for SkillMind so it needs **no server and no vendored
google-adk code**. Double-click the produced file to open it in a browser.

What it does:

1. Parse every concept file (reusing the OKF importer's ``discover_concept_files``
   / ``parse_concept_file`` so the parsing stays in one place).
2. Resolve each ``[text](../folder/concept.md)`` relative markdown link to a
   concept ID → these become the directed graph edges.
3. Embed the whole graph as ``window.BUNDLE`` inside the HTML (no ``fetch`` — the
   file works from ``file://``). Cytoscape.js + marked.js are loaded from a CDN
   on first open; the graph data itself is fully local.

The HTML layout mirrors the OKF viewer: force-directed graph on the left, a
detail panel on the right (type chip, description, resource link, rendered
markdown body, backlinks), plus a search box, type filter and layout selector.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..importers.okf import (
    _strip_leading_title,
    discover_concept_files,
    parse_concept_file,
)

# Color per SkillMind / OKF concept type (CI-aligned where it makes sense).
TYPE_COLORS: dict[str, str] = {
    "user": "#2f8ae5",       # blue
    "feedback": "#f6571e",   # primary orange
    "project": "#1abc9c",    # teal
    "reference": "#8b5cf6",  # violet
    "skill": "#f59e0b",      # amber
    "general": "#64748b",    # slate
}
DEFAULT_COLOR = "#64748b"

# A markdown link whose target is another concept file: `](../folder/x.md)`.
# OKF filenames keep spaces unencoded ("SEO Freelancer.md"), so the path may
# contain spaces — match lazily up to the `.md`, then tolerate an optional
# #anchor, an optional "title" and trailing whitespace before the closing paren.
_LINK_RE = re.compile(r"\]\(\s*([^)\n]+?\.md)(?:#[^)\s]*)?(?:\s+\"[^\"]*\")?\s*\)")


class OKFVisualizer:
    """Render an OKF bundle as a self-contained interactive HTML graph."""

    def __init__(self, bundle_path: str | Path, title: str | None = None):
        self.bundle_path = Path(bundle_path)
        self.title = title or self.bundle_path.name

    # ── Public API ────────────────────────────────────────────────

    def build(self, output_name: str = "okf-graph.html") -> Path:
        """Parse the bundle and write the HTML file. Returns the output path."""
        graph = self.build_graph()
        html_doc = self._render_html(graph)
        out_path = self.bundle_path / output_name
        out_path.write_text(html_doc, encoding="utf-8")
        return out_path

    def build_graph(self) -> dict[str, Any]:
        """
        Parse the bundle into the ``window.BUNDLE`` graph contract.

        Returns a dict with keys: ``types`` (sorted unique node types),
        ``nodes`` (Cytoscape node defs), ``edges`` (directed), ``bodies``
        (concept-id → markdown body), ``stats``.
        """
        if not self.bundle_path.is_dir():
            raise NotADirectoryError(f"OKF bundle not found: {self.bundle_path}")

        base = self.bundle_path.resolve()
        files = discover_concept_files(self.bundle_path)

        concepts: dict[str, dict[str, Any]] = {}
        links: list[tuple[str, str]] = []  # (source_id, raw_link_target)

        for path in files:
            parsed = parse_concept_file(path)
            if not parsed:
                continue
            cid = self._concept_id(path, base)
            fm = parsed["frontmatter"]
            body = parsed["body"]

            concepts[cid] = {
                "id": cid,
                "dir": str(Path(cid).parent).replace("\\", "/"),
                "label": parsed["title"],
                "type": self._concept_type(fm, path),
                "description": self._str_field(fm.get("description")),
                "resource": self._str_field(fm.get("resource")),
                "tags": self._tag_list(fm.get("tags")),
                "body": _strip_leading_title(body, parsed["title"]),
            }

            for raw in _LINK_RE.findall(body):
                links.append((cid, raw))

        # Resolve relative links → concept IDs → directed edges (deduplicated).
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str]] = set()
        degree: dict[str, int] = {cid: 0 for cid in concepts}

        for src, raw in links:
            tgt = self._resolve_link(concepts[src]["dir"], raw)
            if tgt is None or tgt not in concepts or tgt == src:
                continue
            key = (src, tgt)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"data": {"id": f"e{len(edges)}", "source": src, "target": tgt}})
            degree[src] += 1
            degree[tgt] += 1

        # Build Cytoscape node defs (size scales with connectivity).
        nodes: list[dict[str, Any]] = []
        for cid, c in concepts.items():
            color = TYPE_COLORS.get(c["type"], DEFAULT_COLOR)
            size = 26 + min(degree.get(cid, 0) * 4, 44)
            nodes.append({
                "data": {
                    "id": cid,
                    "label": c["label"],
                    "type": c["type"],
                    "color": color,
                    "size": size,
                    "description": c["description"],
                    "resource": c["resource"],
                    "tags": c["tags"],
                }
            })

        types = sorted({c["type"] for c in concepts.values()})
        bodies = {cid: c["body"] for cid, c in concepts.items()}

        return {
            "types": types,
            "nodes": nodes,
            "edges": edges,
            "bodies": bodies,
            "stats": {
                "concepts": len(concepts),
                "edges": len(edges),
                "types": len(types),
            },
        }

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _concept_id(path: Path, base: Path) -> str:
        """Bundle-relative path minus ``.md`` (POSIX) — the canonical concept ID."""
        try:
            rel = path.resolve().relative_to(base)
        except ValueError:
            rel = Path(path.name)
        return rel.with_suffix("").as_posix()

    @staticmethod
    def _concept_type(fm: dict[str, Any], path: Path) -> str:
        """Prefer the skillmind/OKF ``type``; fall back to the parent folder name."""
        for key in ("skillmind_type", "skillmind-type", "type"):
            val = fm.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().lower()
        folder = path.parent.name.lower().rstrip("s")  # references → reference
        return folder or "general"

    @staticmethod
    def _str_field(value: Any) -> str:
        return value.strip() if isinstance(value, str) and value.strip() else ""

    @staticmethod
    def _tag_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(t).strip() for t in value if str(t).strip()]
        if isinstance(value, str) and value.strip():
            return [t.strip() for t in re.split(r"[,;]", value) if t.strip()]
        return []

    @staticmethod
    def _resolve_link(from_dir: str, raw: str) -> str | None:
        """Resolve a relative link target (from a concept's folder) to a concept ID."""
        target = raw.strip().split("#", 1)[0]
        if "://" in target or not target.endswith(".md"):
            return None
        target = target[:-3]  # drop .md
        base_parts = [p for p in from_dir.split("/") if p and p != "."]
        stack = base_parts[:]
        for part in target.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(part)
        return "/".join(stack) if stack else None

    # ── HTML rendering ────────────────────────────────────────────

    def _render_html(self, graph: dict[str, Any]) -> str:
        # Embed as JSON; neutralize any "</script>" sequence inside bodies so it
        # cannot break out of the <script> tag.
        bundle_json = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
        name_json = json.dumps(self.title, ensure_ascii=False).replace("</", "<\\/")
        return _HTML_TEMPLATE.replace("/*__BUNDLE__*/", bundle_json).replace(
            "/*__NAME__*/", name_json
        )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OKF Knowledge Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<style>
  :root { --bg:#0c1115; --panel:#11161c; --line:#1f2933; --text:#e5edf5; --muted:#94a3b8; --accent:#f6571e; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); }
  #app { display:flex; flex-direction:column; height:100vh; }
  header { display:flex; align-items:center; gap:.6rem; padding:.55rem .9rem; background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1 { font-size:1rem; margin:0 .6rem 0 0; font-weight:700; }
  header h1 .accent { color:var(--accent); }
  header .stat { color:var(--muted); font-size:.78rem; }
  header input, header select { background:#0c1115; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:.35rem .5rem; font-size:.82rem; }
  header input { min-width:200px; }
  header button { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:.4rem .7rem; font-size:.82rem; cursor:pointer; }
  header button.ghost { background:#1c252f; color:var(--text); }
  main { flex:1; display:flex; min-height:0; }
  #cy { flex:1; min-width:0; background:radial-gradient(circle at 50% 40%, #131b24 0%, #0c1115 75%); }
  #detail { width:380px; max-width:42vw; background:var(--panel); border-left:1px solid var(--line); padding:1rem 1.1rem; overflow-y:auto; }
  #detail .placeholder { color:var(--muted); font-size:.9rem; line-height:1.5; }
  #detail h2 { font-size:1.15rem; margin:.2rem 0 .5rem; }
  .chip { display:inline-block; font-size:.7rem; text-transform:uppercase; letter-spacing:.04em; padding:.15rem .5rem; border-radius:999px; color:#0c1115; font-weight:700; }
  .cid { color:var(--muted); font-size:.72rem; word-break:break-all; margin:.35rem 0 .6rem; }
  .desc { color:#cbd5e1; font-size:.9rem; margin:.4rem 0 .7rem; }
  .resource a { color:var(--accent); font-size:.82rem; word-break:break-all; }
  .tags { margin:.5rem 0; }
  .tags span { display:inline-block; background:#1c252f; color:#cbd5e1; font-size:.72rem; padding:.12rem .45rem; border-radius:6px; margin:0 .25rem .25rem 0; }
  .sec { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; margin:1rem 0 .35rem; border-top:1px solid var(--line); padding-top:.7rem; }
  .body { font-size:.88rem; line-height:1.55; }
  .body h1,.body h2,.body h3 { font-size:1rem; margin:.9rem 0 .35rem; }
  .body a { color:var(--accent); }
  .body code { background:#0c1115; padding:.1rem .3rem; border-radius:4px; font-size:.82em; }
  .body pre { background:#0c1115; padding:.6rem; border-radius:8px; overflow-x:auto; }
  .links a { display:block; color:#9cc4ff; font-size:.84rem; padding:.18rem 0; cursor:pointer; }
  .links a:hover { color:var(--accent); }
  #fallback { display:none; padding:2rem; color:#fca5a5; font-size:.95rem; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1><span class="accent">OKF</span> Graph</h1>
    <span class="stat" id="stat"></span>
    <input id="search" type="search" placeholder="Search concepts…" autocomplete="off">
    <select id="typeFilter"><option value="">All types</option></select>
    <select id="layout">
      <option value="cose">Layout: force</option>
      <option value="concentric">Layout: concentric</option>
      <option value="breadthfirst">Layout: tree</option>
      <option value="circle">Layout: circle</option>
      <option value="grid">Layout: grid</option>
    </select>
    <button class="ghost" id="fit">Fit</button>
    <button id="reset">Reset</button>
  </header>
  <main>
    <div id="cy"></div>
    <aside id="detail"><p class="placeholder">Tap a node to inspect a concept. Use search and the type filter to focus the graph.</p></aside>
  </main>
  <div id="fallback">Could not load cytoscape.js / marked.js from the CDN. An internet connection is required the first time you open this file; afterwards the graph data is fully local.</div>
</div>
<script>
window.BUNDLE_NAME = /*__NAME__*/;
window.BUNDLE = /*__BUNDLE__*/;
</script>
<script>
(function () {
  var B = window.BUNDLE || { nodes: [], edges: [], bodies: {}, types: [] };
  document.title = (window.BUNDLE_NAME || "OKF") + " — Knowledge Graph";
  document.querySelector("header h1").innerHTML =
    '<span class="accent">' + escapeHtml(window.BUNDLE_NAME || "OKF") + '</span> Graph';
  var st = B.stats || {};
  document.getElementById("stat").textContent =
    (st.concepts || B.nodes.length) + " concepts · " + (st.edges || B.edges.length) + " links";

  if (typeof cytoscape === "undefined") {
    document.querySelector("main").style.display = "none";
    document.getElementById("fallback").style.display = "block";
    return;
  }

  // Type filter options.
  var sel = document.getElementById("typeFilter");
  (B.types || []).forEach(function (t) {
    var o = document.createElement("option"); o.value = t; o.textContent = t; sel.appendChild(o);
  });

  // Lookups.
  var labelById = {}, nodeData = {};
  B.nodes.forEach(function (n) { labelById[n.data.id] = n.data.label; nodeData[n.data.id] = n.data; });
  var backlinks = {};
  B.edges.forEach(function (e) {
    (backlinks[e.data.target] = backlinks[e.data.target] || []).push(e.data.source);
  });

  var cy = cytoscape({
    container: document.getElementById("cy"),
    elements: B.nodes.concat(B.edges),
    style: [
      { selector: "node", style: {
        "background-color": "data(color)", "width": "data(size)", "height": "data(size)",
        "label": "data(label)", "font-size": 9, "color": "#dce6f0",
        "text-wrap": "wrap", "text-max-width": 110, "text-valign": "bottom",
        "text-margin-y": 3, "border-width": 2, "border-color": "#0f172a" } },
      { selector: "node:selected", style: { "border-width": 4, "border-color": "#f59e0b" } },
      { selector: "edge", style: {
        "width": 1.2, "line-color": "#33414f", "target-arrow-color": "#33414f",
        "target-arrow-shape": "triangle", "arrow-scale": .8, "curve-style": "bezier" } },
      { selector: ".dim", style: { "opacity": 0.12 } },
      { selector: ".hl", style: { "line-color": "#f6571e", "target-arrow-color": "#f6571e", "width": 2.2 } }
    ],
    layout: layoutOpts("cose"),
    wheelSensitivity: 0.25
  });

  function layoutOpts(name) {
    var o = { name: name, animate: false, fit: true, padding: 40 };
    if (name === "cose") { o.nodeRepulsion = 9000; o.idealEdgeLength = 90; o.animate = true; o.numIter = 800; }
    if (name === "concentric") { o.concentric = function (n) { return n.degree(); }; o.levelWidth = function () { return 2; }; }
    return o;
  }

  cy.on("tap", "node", function (evt) { showDetail(evt.target.id()); highlight(evt.target); });
  cy.on("tap", function (evt) { if (evt.target === cy) { clearHighlight(); } });

  function highlight(node) {
    clearHighlight();
    var nb = node.closedNeighborhood();
    cy.elements().difference(nb).addClass("dim");
    node.connectedEdges().addClass("hl");
  }
  function clearHighlight() { cy.elements().removeClass("dim hl"); applyFilters(); }

  // Search + type filter → dim non-matching nodes (and their lone edges).
  var searchEl = document.getElementById("search");
  searchEl.addEventListener("input", applyFilters);
  sel.addEventListener("change", applyFilters);

  function applyFilters() {
    var q = searchEl.value.trim().toLowerCase();
    var type = sel.value;
    if (!q && !type) { cy.nodes().removeClass("dim"); cy.edges().removeClass("dim"); return; }
    cy.nodes().forEach(function (n) {
      var d = n.data();
      var hay = (d.label + " " + d.id + " " + (d.tags || []).join(" ")).toLowerCase();
      var match = (!q || hay.indexOf(q) !== -1) && (!type || d.type === type);
      n.toggleClass("dim", !match);
    });
    cy.edges().forEach(function (e) {
      var hidden = e.source().hasClass("dim") || e.target().hasClass("dim");
      e.toggleClass("dim", hidden);
    });
  }

  document.getElementById("layout").addEventListener("change", function () {
    cy.layout(layoutOpts(this.value)).run();
  });
  document.getElementById("fit").addEventListener("click", function () { cy.fit(null, 40); });
  document.getElementById("reset").addEventListener("click", function () {
    searchEl.value = ""; sel.value = ""; clearHighlight();
    cy.layout(layoutOpts(document.getElementById("layout").value)).run();
  });

  function showDetail(id) {
    var d = nodeData[id]; if (!d) return;
    var color = d.color || "#64748b";
    var html = "";
    html += '<span class="chip" style="background:' + color + '">' + escapeHtml(d.type) + "</span>";
    html += "<h2>" + escapeHtml(d.label) + "</h2>";
    html += '<div class="cid">' + escapeHtml(id) + "</div>";
    if (d.description) html += '<div class="desc">' + escapeHtml(d.description) + "</div>";
    if (d.resource) html += '<div class="resource">🔗 <a href="' + escapeHtml(d.resource) +
      '" target="_blank" rel="noopener">' + escapeHtml(d.resource) + "</a></div>";
    if (d.tags && d.tags.length) {
      html += '<div class="tags">' + d.tags.map(function (t) {
        return "<span>" + escapeHtml(t) + "</span>"; }).join("") + "</div>";
    }
    var body = (B.bodies || {})[id];
    if (body) {
      html += '<div class="sec">Concept</div><div class="body">' + renderMd(body) + "</div>";
    }
    var back = backlinks[id] || [];
    if (back.length) {
      html += '<div class="sec">Referenced by (' + back.length + ')</div><div class="links">';
      back.forEach(function (s) {
        html += '<a data-goto="' + escapeHtml(s) + '">← ' + escapeHtml(labelById[s] || s) + "</a>";
      });
      html += "</div>";
    }
    var det = document.getElementById("detail");
    det.innerHTML = html; det.scrollTop = 0;
    wireInternalLinks(det, id);
  }

  // Rewrite in-app navigation: backlink chips + relative .md links in the body.
  function wireInternalLinks(container, currentId) {
    container.querySelectorAll("a[data-goto]").forEach(function (a) {
      a.addEventListener("click", function () { goTo(a.getAttribute("data-goto")); });
    });
    container.querySelectorAll(".body a[href$='.md']").forEach(function (a) {
      var tid = resolveLink(currentId, a.getAttribute("href"));
      a.addEventListener("click", function (ev) {
        ev.preventDefault(); if (tid && nodeData[tid]) goTo(tid);
      });
    });
  }

  function goTo(id) {
    var node = cy.getElementById(id);
    if (node && node.length) { cy.elements().unselect(); node.select(); cy.center(node); highlight(node); }
    showDetail(id);
  }

  // Resolve a relative `[..](../folder/x.md)` href against the current concept's folder.
  function resolveLink(currentId, href) {
    if (!href) return null;
    href = href.split("#")[0].replace(/^\.\//, "");
    if (!/\.md$/.test(href)) return null;
    var dir = currentId.split("/").slice(0, -1);
    var parts = href.replace(/\.md$/, "").split("/");
    var stack = dir.slice();
    parts.forEach(function (p) {
      if (p === "" || p === ".") return;
      if (p === "..") { stack.pop(); } else { stack.push(p); }
    });
    return stack.join("/");
  }

  function renderMd(md) {
    try { return (window.marked ? marked.parse(md) : "<pre>" + escapeHtml(md) + "</pre>"); }
    catch (e) { return "<pre>" + escapeHtml(md) + "</pre>"; }
  }
  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
})();
</script>
</body>
</html>
"""
