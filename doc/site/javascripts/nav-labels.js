(function () {
  function normalizeUrl(href) {
    try {
      var url = new URL(href, document.baseURI);
      url.hash = "";
      return url.href.replace(/\/index\.html$/, "/").replace(/\/$/, "");
    } catch (error) {
      return href;
    }
  }

  function sectionForHref(href) {
    var target = normalizeUrl(href);
    var navLinks = document.querySelectorAll(".wy-menu-vertical a[href]");
    for (var i = 0; i < navLinks.length; i += 1) {
      if (normalizeUrl(navLinks[i].getAttribute("href")) !== target) {
        continue;
      }
      var group = navLinks[i].closest("ul");
      var caption = group ? group.previousElementSibling : null;
      if (caption && caption.classList.contains("caption")) {
        return caption.textContent.replace(/\s+/g, " ").trim();
      }
      return "Inicio";
    }
    return "Documentación";
  }

  function removePageAnchorsFromSidebar() {
    document.querySelectorAll(".wy-menu-vertical a[href]").forEach(function (link) {
      var href = link.getAttribute("href") || "";
      if (href !== "#" && href.indexOf("#") !== -1) {
        var item = link.closest("li");
        if (item) {
          item.remove();
        }
      }
    });
    document.querySelectorAll(".wy-menu-vertical ul").forEach(function (list) {
      if (!list.querySelector("li")) {
        list.remove();
      }
    });
  }

  function relabelNavigationButtons() {
    document.querySelectorAll(".rst-footer-buttons a").forEach(function (link) {
      var text = link.textContent.replace(/\s+/g, " ").trim();
      var title = link.getAttribute("title") || text
        .replace(/^Previous\s*/i, "")
        .replace(/^Anterior\s*/i, "")
        .replace(/^Next\s*/i, "")
        .replace(/^Siguiente\s*/i, "")
        .trim();
      var section = sectionForHref(link.getAttribute("href") || "");
      if (/^Previous/i.test(text) || /^Anterior/i.test(text)) {
        link.innerHTML = '<span class="doc-nav-kicker">Página anterior - ' + section + '</span><span class="doc-nav-title">' + title + "</span>";
      }
      if (/^Next/i.test(text) || /^Siguiente/i.test(text)) {
        link.innerHTML = '<span class="doc-nav-kicker">Página siguiente - ' + section + '</span><span class="doc-nav-title">' + title + "</span>";
      }
    });
  }

  function addTopNavigationButtons() {
    if (document.querySelector(".doc-top-navigation")) {
      return;
    }
    var footerButtons = document.querySelector(".rst-footer-buttons:not(.doc-top-navigation)");
    var breadcrumbs = document.querySelector(".wy-breadcrumbs");
    if (!footerButtons || !breadcrumbs) {
      return;
    }
    var clone = footerButtons.cloneNode(true);
    clone.classList.add("doc-top-navigation");
    breadcrumbs.parentElement.insertAdjacentElement("afterend", clone);
  }

  function enhanceNavigation() {
    removePageAnchorsFromSidebar();
    relabelNavigationButtons();
    addTopNavigationButtons();
    relabelNavigationButtons();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhanceNavigation);
  } else {
    enhanceNavigation();
  }
})();



