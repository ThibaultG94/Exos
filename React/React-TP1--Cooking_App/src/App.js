import React from 'react';
import { useEffect, useState } from 'react';
import axios from 'axios';
import Card from './components/Card';

const App = () => {
	const [data, setData] = useState([]);
	const [textInput, setTextInput] = useState('');

	useEffect(() => {
		axios
			.get(
				'https://www.themealdb.com/api/json/v1/1/search.php?s=' +
					textInput
			)
			.then((res) => setData(res.data.meals));
	}, [textInput]);

	return (
		<div className="app">
			<h1>React Cook</h1>
			<input
				type="text"
				placeholder="Tapez le nom d'un aliment (en anglais)"
				onChange={(e) => setTextInput(e.target.value)}
			/>
			<ul className="meals">
				{data &&
					data
						.slice(0, 24)
						.map((meal) => <Card key={meal.idMeal} meal={meal} />)}
			</ul>
		</div>
	);
};

export default App;
