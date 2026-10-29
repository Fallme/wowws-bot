/* Theme switcher: swaps the theme stylesheet and remembers the choice. */
(function () {
  var THEMES = {
    classic: { color: "#0c171c" },
    endfield: { color: "#f5f6f7" },
    arknights: { color: "#0c0e10" },
    mono: { color: "#ffffff" }
  };
  var STORAGE_KEY = "wowsConsoleTheme";
  var VERSION = "20261029_1";
  var root = document.documentElement;
  var meta = document.getElementById("themeColorMeta");
  var buttons = Array.prototype.slice.call(
    document.querySelectorAll("[data-theme-option]")
  );

  function themeLink() {
    var link = document.getElementById("themeStylesheet");
    if (link) return link;
    link = document.createElement("link");
    link.id = "themeStylesheet";
    link.rel = "stylesheet";
    document.head.appendChild(link);
    return link;
  }

  function apply(name, persist) {
    var theme = Object.prototype.hasOwnProperty.call(THEMES, name)
      ? name
      : "endfield";
    root.dataset.theme = theme;
    themeLink().href = "/themes/" + theme + ".css?v=" + VERSION;
    if (meta) meta.setAttribute("content", THEMES[theme].color);
    buttons.forEach(function (button) {
      var active = button.getAttribute("data-theme-option") === theme;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, theme);
        var url = new URL(location.href);
        url.searchParams.set("theme", theme);
        history.replaceState(null, "", url);
      } catch (error) {}
    }
  }

  apply(root.dataset.theme || "endfield", false);
  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      apply(button.getAttribute("data-theme-option"), true);
    });
  });
})();
