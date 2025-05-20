// Génération de nombres dans le tableau

let numberRandom = () => {
  let number = Math.floor(Math.random() * 100);
  return number;
};

let numberOne = numberRandom();
let numberTwo = numberRandom();
let numberThree = numberRandom();
let numberFour = numberRandom();
let numberFive = numberRandom();
let numberSix = numberRandom();
let numberSeven = numberRandom();
let numberEight = numberRandom();
let numberNine = numberRandom();
let numberTen = numberRandom();

const numberArray = [
  numberOne,
  numberTwo,
  numberThree,
  numberFour,
  numberFive,
  numberSix,
  numberSeven,
  numberEight,
  numberNine,
  numberTen,
];

const numberBubble = [
  numberOne,
  numberTwo,
  numberThree,
  numberFour,
  numberFive,
  numberSix,
  numberSeven,
  numberEight,
  numberNine,
  numberTen,
];

const numberSelection = [
  numberOne,
  numberTwo,
  numberThree,
  numberFour,
  numberFive,
  numberSix,
  numberSeven,
  numberEight,
  numberNine,
  numberTen,
];

const numberQuick = [
  numberOne,
  numberTwo,
  numberThree,
  numberFour,
  numberFive,
  numberSix,
  numberSeven,
  numberEight,
  numberNine,
  numberTen,
];

for (let i = 0; i < numberArray.length; i++) {
  let num1 = document.querySelectorAll("#sortArray > td");
  let num2 = document.querySelectorAll("#sortBubble > td");
  let num3 = document.querySelectorAll("#sortSelection > td");
  let num4 = document.querySelectorAll("#sortQuick > td");
  num1.forEach((e) => {
    if (e.id == i) {
      e.textContent = numberArray[i];
    }
  });
  num2.forEach((e) => {
    if (e.id == i) {
      e.textContent = numberBubble[i];
    }
  });
  num3.forEach((e) => {
    if (e.id == i) {
      e.textContent = numberSelection[i];
    }
  });
  // num4.forEach((e) => {
  //   if (e.id == i) {
  //     e.textContent = numberQuick[i];
  //   }
  // });
}
