/* Theme picker: swaps the palette stylesheet and remembers the choice. */
(function () {
  var THEMES = {
    classic: { label: "黑金", color: "#0c171c" },
    "light-gold": { label: "浅金", color: "#f5f6f7" },
    "mid-grey": { label: "中灰", color: "#6d7276" },
    mono: { label: "灰白", color: "#ffffff" },
    navy: { label: "深蓝", color: "#0e1622" },
    amber: { label: "暖褐", color: "#1c1511" },
    teal: { label: "青灰", color: "#e7efee" },
    violet: { label: "紫灰", color: "#efedf4" }
  };
  var ALIASES = { endfield: "light-gold", arknights: "mid-grey" };
  var STORAGE_KEY = "wowsConsoleTheme";
  var VERSION = "20261029_2";
  var root = document.documentElement;
  var meta = document.getElementById("themeColorMeta");
  var trigger = document.getElementById("themeTrigger");
  var menu = document.getElementById("themeMenu");
  var currentLabel = document.getElementById("themeCurrentLabel");
  var options = Array.prototype.slice.call(
    document.querySelectorAll("[data-theme-option]")
  );

  function resolve(name) {
    var value = ALIASES[name] || name;
    return Object.prototype.hasOwnProperty.call(THEMES, value) ? value : "";
  }

  function themeLink() {
    var link = document.getElementById("themeStylesheet");
    if (link) return link;
    link = document.createElement("link");
    link.id = "themeStylesheet";
    link.rel = "stylesheet";
    document.head.appendChild(link);
    return link;
  }

  function openMenu(open) {
    if (!menu || !trigger) return;
    menu.hidden = !open;
    trigger.setAttribute("aria-expanded", String(open));
    if (open && options.length) {
      options[0].focus();
    }
  }

  function apply(name, persist) {
    var theme = resolve(name) || "light-gold";
    root.dataset.theme = theme;
    themeLink().href = "/themes/" + theme + ".css?v=" + VERSION;
    if (meta) meta.setAttribute("content", THEMES[theme].color);
    if (trigger) trigger.dataset.theme = theme;
    if (currentLabel) currentLabel.textContent = THEMES[theme].label;
    options.forEach(function (button) {
      var active = button.getAttribute("data-theme-option") === theme;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    if (menu) menu.hidden = true;
    if (trigger) trigger.setAttribute("aria-expanded", "false");
    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, theme);
        var url = new URL(location.href);
        url.searchParams.set("theme", theme);
        history.replaceState(null, "", url);
      } catch (error) {}
    }
  }

  apply(root.dataset.theme || "light-gold", false);

  if (trigger && menu) {
    trigger.addEventListener("click", function (event) {
      event.stopPropagation();
      openMenu(menu.hidden);
    });
    menu.addEventListener("click", function (event) {
      var button = event.target.closest("[data-theme-option]");
      if (!button) return;
      apply(button.getAttribute("data-theme-option"), true);
      trigger.focus();
    });
    document.addEventListener("click", function (event) {
      if (!menu.hidden && event.target !== trigger) {
        openMenu(false);
      }
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && !menu.hidden) {
        openMenu(false);
        trigger.focus();
      }
    });
  }
})();
