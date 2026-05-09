(function () {
  function renderMermaidBlocks() {
    if (!window.mermaid) {
      return;
    }
    window.mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });
    document.querySelectorAll("pre code.language-mermaid").forEach(function (code) {
      var pre = code.parentElement;
      var container = document.createElement("div");
      container.className = "mermaid";
      container.textContent = code.textContent;
      pre.replaceWith(container);
    });
    window.mermaid.run({ querySelector: ".mermaid" });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMermaidBlocks);
  } else {
    renderMermaidBlocks();
  }
})();




