import React from 'react';

const NavigationMiddle = () => {
	return (
		<nav className="nav-responsive">
			<button className="toggle-menu">
				<span className="toggle-menu-content">
					<span className="toggle-menu-lines">
						<span className="toggle-menu-line"></span>
						<span className="toggle-menu-line"></span>
						<span className="toggle-menu-line"></span>
					</span>
					<span className="toggle-menu-label">Menu</span>
				</span>
			</button>
		</nav>
	);
};

export default NavigationMiddle;
