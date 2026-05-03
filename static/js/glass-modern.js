(function () {
  var root = document.documentElement;
  var storedTheme = window.localStorage.getItem("portfolio-theme");
  var defaultTheme = storedTheme || "dark";
  var sidebar = document.querySelector("[data-sidebar]");
  var toggleButton = document.querySelector("[data-theme-toggle]");
  var mobileButton = document.querySelector("[data-mobile-nav]");

  function setTheme(theme) {
    root.setAttribute("data-theme", theme);
    window.localStorage.setItem("portfolio-theme", theme);
    if (toggleButton) {
      toggleButton.textContent = theme === "dark" ? "Switch to Light" : "Switch to Dark";
      toggleButton.setAttribute("aria-label", toggleButton.textContent);
    }
  }

  setTheme(defaultTheme);

  if (toggleButton) {
    toggleButton.addEventListener("click", function () {
      var currentTheme = root.getAttribute("data-theme") || "dark";
      setTheme(currentTheme === "dark" ? "light" : "dark");
    });
  }

  if (mobileButton && sidebar) {
    mobileButton.addEventListener("click", function () {
      var isOpen = sidebar.classList.toggle("open");
      mobileButton.classList.toggle("is-open", isOpen);
      mobileButton.setAttribute("aria-expanded", String(isOpen));
    });

    sidebar.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        sidebar.classList.remove("open");
        mobileButton.classList.remove("is-open");
        mobileButton.setAttribute("aria-expanded", "false");
      });
    });
  }
})();
