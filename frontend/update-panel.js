/* Update chip: reads /api/update, shows repo version and active stats. */
(function () {
  var chip = document.getElementById("updateChip");
  var popover = document.getElementById("updatePopover");
  var closeButton = document.getElementById("updateClose");
  var checkButton = document.getElementById("updateCheck");
  var link = document.getElementById("updateLink");
  if (!chip || !popover || !checkButton || !link) return;

  function text(id, value) {
    var node = document.getElementById(id);
    if (node) node.textContent = value;
  }

  function formatDate(value) {
    if (!value) return "";
    var date = new Date(value);
    if (isNaN(date.getTime())) return value;
    return date.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit"
    });
  }

  function render(data) {
    var current = data.current_version || "unknown";
    var latest = data.latest_version || "unknown";
    var update = data.update_available === true;
    var commit = data.commit || {};
    var stats = data.stats || {};
    var github = data.github || {};
    chip.classList.toggle("has-update", update);
    chip.title = update
      ? "发现新版本 " + latest
      : "当前版本 " + current + "，点击查看更新";
    text("updateVersion", update ? "可更新" : "v" + current);
    text("updateCurrent", current === "unknown" ? "—" : "v" + current);
    text("updateLatest", latest === "unknown" ? "—" : "v" + latest);
    text(
      "updateCommit",
      commit.message || "—"
    );
    var commitNode = document.getElementById("updateCommit");
    if (commitNode) {
      commitNode.title = [formatDate(commit.date), commit.sha]
        .filter(Boolean)
        .join(" · ");
    }
    if (Number.isFinite(stats.active_30d)) {
      text(
        "updateActive",
        "7天 " + (stats.active_7d || 0) + " · 30天 " + stats.active_30d
      );
    } else if (Number.isFinite(github.clones_unique_14d)) {
      text(
        "updateActive",
        "克隆 " + github.clones_unique_14d + " · 访问 " +
          (github.views_unique_14d || 0) + "（14天）"
      );
    } else {
      text("updateActive", "未配置统计");
    }
    var note = "已是最新版本。";
    if (data.error) {
      note = "更新检查失败：" + String(data.error).slice(0, 140);
    } else if (update) {
      note = "发现新版本 " + latest + "，建议更新后再运行任务。";
    } else if (data.local_ahead) {
      note = "本地版本高于仓库主分支。";
    }
    text("updateNote", note);
    link.href = commit.url || data.repo_url || link.href;
    link.textContent = commit.url ? "打开提交" : "打开仓库";
  }

  function fail(error) {
    chip.classList.remove("has-update");
    chip.title = "更新检查不可用";
    text("updateVersion", "更新不可用");
    text("updateNote", "无法读取更新信息：" + String(error.message || error));
  }

  function setBusy(busy) {
    checkButton.disabled = busy;
    checkButton.textContent = busy ? "检查中…" : "重新检查";
  }

  function load(force) {
    setBusy(true);
    if (!force) text("updateVersion", "检查中…");
    return fetch("/api/update" + (force ? "?refresh=1" : ""), {
      cache: "no-store"
    })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data || (data.error && !data.current_version)) {
          throw new Error((data && data.error) || "更新服务未就绪");
        }
        render(data);
      })
      .catch(fail)
      .finally(function () {
        setBusy(false);
      });
  }

  function open(show) {
    popover.hidden = !show;
    chip.setAttribute("aria-expanded", String(show));
  }

  chip.addEventListener("click", function (event) {
    event.stopPropagation();
    open(popover.hidden);
  });
  if (closeButton) {
    closeButton.addEventListener("click", function () {
      open(false);
    });
  }
  checkButton.addEventListener("click", function () {
    load(true);
  });
  document.addEventListener("click", function (event) {
    if (!popover.hidden && event.target !== chip) {
      open(false);
    }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !popover.hidden) open(false);
  });

  load(false);
})();
