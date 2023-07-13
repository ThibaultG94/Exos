import React, { useRef } from 'react';
import emailjs from '@emailjs/browser';
import { init } from '@emailjs/browser';
init(process.env.ID);

const ContactForm = () => {
	const form = useRef();

	const sendEmail = (e) => {
		e.preventDefault();
		const formMessage = document.querySelector('.form-message');

		emailjs
			.sendForm(
				'service_adfkhc8',
				'template_96ais68',
				form.current,
				process.env.REACT_APP_ID
			)
			.then(
				(result) => {
					console.log(result.text);
					form.current.reset();
					formMessage.innerHTML =
						"<p class='success'>Message envoyé !</p>";

					setTimeout(() => {
						formMessage.innerHTML = '';
					}, 2500);
				},
				(error) => {
					console.log(error.text);
					formMessage.innerHTML =
						"<p class='error'>Une erreur s'est produite, veuillez réessayez plus tard</p>";

					setTimeout(() => {
						formMessage.innerHTML = '';
					}, 2500);
				}
			);
	};

	return (
		<div className="form-container">
			<h2>contactez-nous</h2>
			<form ref={form} onSubmit={sendEmail} className="form-content">
				<label>Name</label>
				<input
					type="text"
					name="name"
					required
					autoComplete="off"
					id="name"
				/>
				<label>Email</label>
				<input
					type="email"
					name="email"
					required
					autoComplete="off"
					id="mail"
				/>
				<label>Message</label>
				<textarea name="message" required />
				<input
					type="submit"
					value="Envoyer"
					id="mess"
					className="hover button"
				/>
			</form>
			<div className="form-message"></div>
		</div>
	);
};

export default ContactForm;
