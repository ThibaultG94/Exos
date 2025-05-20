// function quickSort(array) {
//   if (array.length === 1) {
//     return array;
//   }

//   const pivot = array[array.length - 1];
//   let arrLeft = [];
//   let arrRight = [];

//   for (let i = 0; i < array.length - 1; i++) {
//     array[i] < pivot ? arrLeft.push(array[i]) : arrRight.push(array[i]);
//   }

//   if (arrLeft.length > 0 && arrRight.length > 0) {
//     console.log(arrLeft, pivot, arrRight);
//     return [...quickSort(arrLeft), pivot, ...quickSort(arrRight)];
//   } else if (arrLeft.length > 0) {
//     console.log(arrLeft);
//     return [...quickSort(arrLeft), pivot];
//   } else {
//     console.log(arrRight);
//     return [pivot, ...quickSort(arrRight)];
//   }
// }
// console.log(quickSort(numberQuick));

btn.addEventListener("click", () => {
  let num4 = document.querySelectorAll("#sortQuick > td");
  let n = 0;
  let o = 0;
  let arrLeft = [];
  let arrRight = [];

  if (btn.value == "Reset") {
    window.location.reload();
  } else {
    btn.style.transform = "translateY(500px)";
  }

  function quickSort(array) {
    setTimeout(() => {
      if (array.length === 1) {
        return array;
      }

      const pivot = array[array.length - 1];

      for (let i = 0; i < num4.length; i++) {
        if (num4[i].textContent == pivot) {
          num4[i].style.background = "#a9a9ff";
        }
      }

      if (n === 0) {
        let newTr = document.createElement("tr");
        let newTd = document.createElement("td");
        addTr.append(newTr);
        newTr.classList.add("addTd");
        addTd = document.querySelector(".addTd");
        addTd.append(newTd);
        newTd.id = n;
        newTd.style.visibility = "visible";
      } else {
        let newTd = document.createElement("td");
        addTd = document.querySelector(".addTd");
        addTd.append(newTd);
        newTd.id = n;
        newTd.style.visibility = "visible";
      }

      if (n < array.length - 1) {
        if (array[n] < pivot) {
          arrLeft.push(array[n]);
          num4.forEach((e) => {
            if (e.id == n) {
              e.style.visibility = "hidden";
            }
          });
          document.querySelectorAll(".addTd > td").forEach((e) => {
            if (e.id == arrLeft.length - 1) {
              e.textContent = arrLeft[arrLeft.length - 1];
            }
          });

          n++;
          quickSort(numberQuick);
        } else {
          arrRight.push(array[n]);
          console.log(arrRight);
          n++;
          quickSort(numberQuick);
        }
      } else {
        document.querySelectorAll(".addTd > td").forEach((e) => {
          if ((e.id = arrLeft.length)) {
            e.textContent = pivot;
            console.log(e.id, arrLeft.length);
          }
        });
        let p = 0;

        // document.querySelectorAll(".addTd > td").forEach((e) => {
        //   setTimeout(() => {
        //     if (e.textContent == "") {
        //       if (p < arrRight.length) {
        //         e.textContent = arrRight[p];
        //         p++;
        //       } else {
        //         e.textContent = pivot;
        //         num4.forEach((e) => {
        //           e.style.visibility = "hidden";
        //         });
        //       }
        //     }
        //   }, 500);
        // });

        // n = 0;
      }
    }, 500);
  }
  quickSort(numberQuick);
});
