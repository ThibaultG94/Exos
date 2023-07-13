const $signInButton = document.querySelector('.sign-in-button');

$signInButton.addEventListener('click', function () {
	console.log(this);
});

$signInButton.addEventListener('click', () => {
	console.log(this);
});
