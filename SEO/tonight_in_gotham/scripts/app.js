const dropdownMenu = document.querySelector(".dropdown");
const dropdownLink = document.querySelector(".header-dropdown-link");

function toggleNavbar() {
  if (
    !dropdownMenu.getAttribute("style") ||
    dropdownMenu.getAttribute("style") === "display: none;"
  ) {
    dropdownMenu.style.display = "block";
    dropdownLink.setAttribute("aria-expanded", "true");
  } else {
    dropdownMenu.style.display = "none";
    dropdownLink.setAttribute("aria-expanded", "false");
  }
}

dropdownLink.addEventListener("click", (e) => {
  e.preventDefault();
  toggleNavbar();
});
