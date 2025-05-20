// function selectionSort() {
//   for (let i = 0; i < numberSelection.length - 1; i++) {
//     let minIndex = i;
//     for (let j = i + 1; j < numberSelection.length; j++) {
//       if (numberSelection[j] < numberSelection[minIndex]) {
//         minIndex = j;
//       }
//     }
//     [numberSelection[i], numberSelection[minIndex]] = [
//       numberSelection[minIndex],
//       numberSelection[i],
//     ];
//     console.log(numberSelection);
//   }
//   console.log(numberSelection);
// }

// selectionSort();

btn.addEventListener("click", () => {
  let num3 = document.querySelectorAll("#sortSelection > td");
  let l = 0;
  let m = l + 1;

  if (btn.value == "Reset") {
    window.location.reload();
  } else {
    btn.style.opacity = 0;
    function numberAppearSelection() {
      if (l < numberSelection.length - 1) {
        btn.style.opacity = 0;
        let minIndex = l;
        if (m < 10) {
          num3[m].style.background = "#a9a9ff";
        }
        setTimeout(() => {
          if (m < numberSelection.length) {
            if (numberSelection[m] < numberSelection[minIndex]) {
              minIndex = m;
            }
            [numberSelection[l], numberSelection[minIndex]] = [
              numberSelection[minIndex],
              numberSelection[l],
            ];
            for (let i = 0; i < numberSelection.length; i++) {
              num3.forEach((e) => {
                if (e.id == i) {
                  e.textContent = numberSelection[i];
                }
              });
            }
            num3[m].style.background = "rgb(255, 255, 169)";
            if (m == numberSelection.length - 1) {
              num3[l].style.background = "#a9a9ff";
            }
            m++;
            numberAppearSelection();
          } else {
            l++;
            m = l + 1;
            numberAppearSelection();
          }
        }, 500);
      } else {
        num3[l].style.background = "#a9a9ff";
        setTimeout(() => {
          btn.style.opacity = 1;
          btn.value = "Reset";
        }, 500);
      }
    }
    numberAppearSelection();
  }
});
