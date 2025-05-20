// Bubble sort

// for (let i = 0; i < numberBubble.length - 1; i++) {
//   for (let j = 0; j < numberBubble.length - i; j++) {
//     if (numberBubble[j] > numberBubble[j + 1]) {
//       [numberBubble[j], numberBubble[j + 1]] = [
//         numberBubble[j + 1],
//         numberBubble[j],
//       ];
//     }
//   }
// }

btn.addEventListener("click", () => {
  let num2 = document.querySelectorAll("#sortBubble > td");
  let j = 0;
  let k = 0;

  if (btn.value == "Reset") {
    window.location.reload();
  } else {
    btn.style.visibility = "hidden";
    function numberAppearBubble() {
      if (j < numberBubble.length - 1) {
        btn.style.visibility = "hidden";
        if (k < 10) {
          num2[k].style.background = "#a9a9ff";
        }
        setTimeout(() => {
          if (k < numberBubble.length - j) {
            if (numberBubble[k] > numberBubble[k + 1]) {
              [numberBubble[k], numberBubble[k + 1]] = [
                numberBubble[k + 1],
                numberBubble[k],
              ];
              for (let i = 0; i < numberBubble.length; i++) {
                num2.forEach((e) => {
                  if (e.id == i) {
                    e.textContent = numberBubble[i];
                  }
                });
              }
              num2[k].style.background = "rgb(255, 255, 169)";
              k++;
              numberAppearBubble();
            } else {
              num2[k].style.background = "rgb(255, 255, 169)";
              k++;
              numberAppearBubble();
            }
          } else {
            k = 0;
            j++;
            numberAppearBubble();
          }
        }, 500);
      } else {
        setTimeout(() => {
          num2[k].style.background = "#a9a9ff";
        }, 0);
        setTimeout(() => {
          num2[k + 1].style.background = "#a9a9ff";
          num2[k].style.background = "rgb(255, 255, 169)";
        }, 500);
        setTimeout(() => {
          num2[k].style.background = "#a9a9ff";
        }, 1000);
        setTimeout(() => {
          btn.style.visibility = "visible";
          btn.value = "Reset";
        }, 1500);
      }
    }
    numberAppearBubble();
  }
});
