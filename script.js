document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.querySelector(".sidebar"),
    backdrop = document.querySelector(".menu-backdrop"),
    toggle = () => {
      sidebar?.classList.toggle("open");
      backdrop?.classList.toggle("show");
    };
  document.querySelector(".menu-toggle")?.addEventListener("click", toggle);
  document.querySelector(".close-menu")?.addEventListener("click", toggle);
  backdrop?.addEventListener("click", toggle);
  const targets = document.querySelectorAll(
    ".feature-card,.about-box,.history-card,.result-card,.page-heading",
  );
  targets.forEach((item, index) => {
    item.classList.add("reveal");
    item.style.transitionDelay = `${Math.min(index % 4, 3) * 75}ms`;
  });
  const observer = new IntersectionObserver(
    (entries) =>
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      }),
    { threshold: 0.12 },
  );
  targets.forEach((item) => observer.observe(item));
});
