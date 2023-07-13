import React from 'react';

const Card = ({ meal }) => {
	return (
		<li className="card">
			<div className="infos">
				<div className="meal-title">
					<h2>{meal.strMeal}</h2>
				</div>
				<p>{'Origin : ' + meal.strArea}</p>
			</div>
			<img src={meal.strMealThumb} alt={'recette de ' + meal.strMeal} />
			<div className="recette">
				<p>{meal.strInstructions}</p>
			</div>
		</li>
	);
};

export default Card;
