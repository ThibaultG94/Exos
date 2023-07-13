import React from 'react';
import MediaQuery from 'react-responsive';
import NavigationLarge from './NavigationLarge';
import NavigationMiddle from './NavigationMiddle';

const Header = () => {
	return (
		<header className="header">
			<div className="header-container">
				<div className="header-top">
					<h1>Spécialiste gestion locative de courte durée</h1>
				</div>
				<div className="header-primary">
					<div className="header-nav">
						<div className="header-logo">
							<a href="/" className="logo">
								<img src="./img/headerlogo.webp" alt="logo" />
							</a>
						</div>
						<MediaQuery minWidth={1025}>
							<NavigationLarge />
						</MediaQuery>
						<MediaQuery maxWidth={1024}>
							<NavigationMiddle />
						</MediaQuery>
					</div>
					<div className="btn-container header-btn-container">
						<a href="" className="btn header-btn">
							<span>Confiez-nous votre bien</span>
						</a>
					</div>
				</div>
			</div>
		</header>
	);
};

export default Header;
