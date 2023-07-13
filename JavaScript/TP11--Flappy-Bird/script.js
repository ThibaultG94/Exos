const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const img = new Image();
img.src = "./media/flappy-bird-set.png";

// general settings
let gamePlaying = false;
let scoreMode = false;
let levelMode = false;
const gravity = 0.5;
const speed = 6.2;
const size = [51, 36];
const jump = -11.5;
const cTenth = canvas.width / 10;

let index = 0,
  flight,
  flyHeight,
  bestScore,
  bestLevelScore,
  currentScore,
  lastScore = 0,
  sessionScore = 0,
  pipe;

if (localStorage.score) {
  bestScore = localStorage.score;
} else {
  bestScore = 0;
}

if (localStorage.levelScore) {
  bestLevelScore = localStorage.levelScore;
} else {
  bestLevelScore = 0;
}

// pipe settings
const pipeWidth = 78;
const pipeGap = 270;
const pipeLoc = () =>
  Math.random() * (canvas.height - (pipeGap + pipeWidth) - pipeWidth) +
  pipeWidth;

const setup = () => {
  currentScore = 0;
  flight = jump;

  // set initial flyHeight (middle of screen - size of the bird)
  flyHeight = canvas.height / 2 - size[1] / 2;

  // setup first 3 pipes
  pipes = Array(3)
    .fill()
    .map((a, i) => [canvas.width + i * (pipeGap + pipeWidth), pipeLoc()]);
};

const render = () => {
  //make the pipe and bird moving
  index++;

  // background first part
  ctx.drawImage(
    img,
    0,
    0,
    canvas.width,
    canvas.height,
    -((index * (speed / 2)) % canvas.width) + canvas.width,
    0,
    canvas.width,
    canvas.height
  );
  // background second part
  ctx.drawImage(
    img,
    0,
    0,
    canvas.width,
    canvas.height,
    -((index * (speed / 2)) % canvas.width),
    0,
    canvas.width,
    canvas.height
  );

  // pipe display
  if (gamePlaying) {
    pipes.map((pipe) => {
      // pipe moving
      pipe[0] -= speed;

      // top pipe
      ctx.drawImage(
        img,
        432,
        588 - pipe[1],
        pipeWidth,
        pipe[1],
        pipe[0],
        0,
        pipeWidth,
        pipe[1]
      );
      // bottom pipe
      ctx.drawImage(
        img,
        432 + pipeWidth,
        108,
        pipeWidth,
        canvas.height - pipe[1] + pipeGap,
        pipe[0],
        pipe[1] + pipeGap,
        pipeWidth,
        canvas.height - pipe[1] + pipeGap
      );

      // take one point + create new pipe
      if (pipe[0] <= -pipeWidth) {
        currentScore++;
        // check if it's the best score
        sessionScore = Math.max(sessionScore, currentScore);
        if (scoreMode) {
          bestScore = Math.max(bestScore, currentScore);
          localStorage.score = bestScore;
        } else if (levelMode) {
          bestLevelScore = Math.max(bestLevelScore, currentScore);
          localStorage.levelScore = bestLevelScore;
        }

        // remove and create new pipe
        pipes = [
          ...pipes.slice(1),
          [pipes[pipes.length - 1][0] + pipeGap + pipeWidth, pipeLoc()],
        ];
      }

      // if hit the pipe, end
      if (
        [
          pipe[0] <= cTenth + size[0],
          pipe[0] + pipeWidth >= cTenth,
          pipe[1] > flyHeight || pipe[1] + pipeGap < flyHeight + size[1],
        ].every((elem) => elem)
      ) {
        gamePlaying = false;
        lastScore = currentScore;
        setup();
      }
    });
  }

  // draw bird
  if (gamePlaying) {
    ctx.drawImage(
      img,
      432,
      Math.floor((index % 9) / 3) * size[1],
      ...size,
      cTenth,
      flyHeight,
      ...size
    );
    flight += gravity;
    flyHeight = Math.min(flyHeight + flight, canvas.height - size[1]);
  } else if (scoreMode) {
    ctx.drawImage(
      img,
      432,
      Math.floor((index % 9) / 3) * size[1],
      ...size,
      canvas.width / 2 - size[0] / 2,
      flyHeight,
      ...size
    );
    flyHeight = canvas.height / 2 - size[1] / 2;
    ctx.fillText(`Meilleur Score : ${bestScore}`, 55, 120);
    ctx.fillText(`Score Session : ${sessionScore}`, 74, 210);
    ctx.fillText(`Dernier Score : ${lastScore}`, 74, 300);
    ctx.fillText(`Cliquez pour jouer`, 48, 535);
    ctx.font = "bold 30px courier";
  } else if (levelMode) {
    ctx.drawImage(
      img,
      432,
      Math.floor((index % 9) / 3) * size[1],
      ...size,
      canvas.width / 2 - size[0] / 2,
      flyHeight,
      ...size
    );
    flyHeight = canvas.height / 2 - size[1] / 2;
    ctx.fillText(`Meilleur Score : ${bestLevelScore}`, 55, 120);
    ctx.fillText(`Score Session : ${sessionScore}`, 74, 210);
    ctx.fillText(`Dernier Score : ${lastScore}`, 74, 300);
    ctx.fillText(`Cliquez pour jouer`, 48, 535);
    ctx.font = "bold 30px courier";
  }

  if (scoreMode) {
    document.getElementById("bestScore").innerHTML = `Meilleur : ${bestScore}`;
    document.getElementById(
      "currentScore"
    ).innerHTML = `Actuel : ${currentScore}`;
  } else if (levelMode) {
    document.getElementById(
      "bestScore"
    ).innerHTML = `Meilleur : ${bestLevelScore}`;
    document.getElementById(
      "currentScore"
    ).innerHTML = `Actuel : ${currentScore}`;
  }

  // tell the browser to perform anim
  window.requestAnimationFrame(render);
};

// Launch setup
setup();
img.onload = render;

// start game
document.addEventListener("click", (e) => {
  if (scoreMode || levelMode) {
    gamePlaying = true;
  } else if (e.target.id == "btn1") {
    scoreMode = true;
    document.querySelector(".on-canvas").innerHTML = ``;
  } else if (e.target.id == "btn2") {
    levelMode = true;
    document.querySelector(".on-canvas").innerHTML = ``;
  }
});
window.onclick = () => (flight = jump);
