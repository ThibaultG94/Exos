setInterval(() => {
  let firstColor = Math.floor(Math.random() * 255);
  let secondColor = Math.floor(Math.random() * 255);
  let thirdColor = Math.floor(Math.random() * 255);

  const rgb = [firstColor, secondColor, thirdColor];
  const body = document.querySelector("body");
  const rbgText = document.querySelector(".text");

  body.style.background = "rgb(" + rgb[0] + ", " + rgb[1] + ", " + rgb[2] + ")";
  rbgText.textContent = "rgb(" + rgb[0] + ", " + rgb[1] + ", " + rgb[2] + ")";
}, 1000);
