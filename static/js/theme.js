(function () {
  const root = document.documentElement;
  const toggleBtn = document.getElementById("themeToggle");

  function applyTheme(theme) {
    if (theme === "dark") {
      root.setAttribute("data-theme", "dark");
      if (toggleBtn) toggleBtn.textContent = "☀️ Light";
    } else {
      root.removeAttribute("data-theme");
      if (toggleBtn) toggleBtn.textContent = "🌙 Dark";
    }
  }

  const saved = localStorage.getItem("theme") || "light";
  applyTheme(saved);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", function () {
      const current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
      const next = current === "dark" ? "light" : "dark";
      localStorage.setItem("theme", next);
      applyTheme(next);
    });
  }
})();