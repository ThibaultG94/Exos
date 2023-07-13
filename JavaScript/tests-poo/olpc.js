/************ OBJET LITTERAL **********/
const Me = {
	firstName: 'Thibault',
	lastName: 'Guilhem',
	getFullName: () => `${Me.firstName} ${Me.lastName}`,
	sayHello: () => console.log('Hello'),
};

console.log(Me.getFullName());

Me.sayHello();

/************ PROTOTYPE **********/
function ReMe(firstNameAgain, lastNameAgain) {
	this._firstNameAgain = firstNameAgain;
	this._lastNameAgain = lastNameAgain;
}

ReMe.prototype.getFullNameAgain = function () {
	return `${this._firstNameAgain} ${this._lastNameAgain}`;
};

ReMe.prototype.sayHelloAgain = function () {
	console.log('Hello There !');
};

const Thibault = new ReMe('Thibault', 'Guilhem');

console.log(Thibault.getFullNameAgain());

Thibault.sayHelloAgain();

/************ CLASS **********/
class ThisIsMe {
	constructor(myFirstName, myLastName) {
		this._myFirstName = myFirstName;
		this._myLastName = myLastName;
	}

	getMyFullName() {
		return `${this._myFirstName} ${this._myLastName}`;
	}

	sayHelloToMe() {
		console.log(`Hello ${this._myFirstName} !`);
	}
}

const ThisIsThibault = new ThisIsMe('Thibault', 'Guilhem');

console.log(ThisIsThibault.getMyFullName());

ThisIsThibault.sayHelloToMe();
