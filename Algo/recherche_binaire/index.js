let myArray = [];
let i = 0;

function randomArray() {
  if (i < 100) {
    let randomNumber = Math.floor(Math.random() * `${i}0`);
    myArray.push(randomNumber);
    i++;
    randomArray();
  }
}

const findElement = (array, thingToFind) => {
  for (let element of array) {
    if (element === thingToFind) {
      return true;
    }
  }
  return false;
};

const binarySearch = (array, thingToFind, start, end) => {
  if (start > end) {
    console.log("false");
    return false;
  }
  let mid = Math.floor((start + end) / 2);
  if (array[mid] === thingToFind) {
    console.log("true");
    return true;
  }

  if (thingToFind < array[mid]) {
    return binarySearch(array, thingToFind, start, mid - 1);
  } else {
    return binarySearch(array, thingToFind, mid + 1, end);
  }
  console.log(start, mid, end);
};

randomArray();
myArray.sort((a, b) => a - b);
console.log(myArray);
findElement(myArray, 20);
binarySearch(myArray, 20, 0, myArray.length);
