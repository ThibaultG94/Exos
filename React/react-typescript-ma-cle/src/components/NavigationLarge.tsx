import React from 'react';

const NavigationLarge = () => {
	return (
		<nav>
			<ul className="header-navbar">
				<li className="navbar_item">
					<a href="/">Ma Clé</a>
					<div className="trick"></div>
					<ul className="navbar_under-item">
						<li>Ma Clé Expertise</li>
						<li>La Location Meublée</li>
					</ul>
				</li>
				<li className="navbar_item">
					<a href="/">Conciergerie Locative</a>
				</li>
				<li className="navbar_item">
					<a href="/">Bail Mobilité</a>
				</li>
				<li className="navbar_item">
					<a href="/">Nos tarifs</a>
				</li>
				<li className="navbar_item">
					<a href="/">Proposez votre bien</a>
				</li>
				<li className="navbar_item">
					<a href="/">Contact</a>
				</li>
			</ul>
		</nav>
	);
};

export default NavigationLarge;
