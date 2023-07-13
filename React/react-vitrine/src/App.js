import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './page/Home';
import Project1 from './page/Project1';
import Project2 from './page/Project2';
import Project3 from './page/Project3';
import Project4 from './page/Project4';
import Contact from './page/Contact';
import NotFound from './page/NotFound';
import { AnimatePresence } from 'framer-motion';

const App = () => {
	return (
		<BrowserRouter>
			<AnimatePresence>
				<Routes>
					<Route path="/" element={<Home />} />
					<Route path="/project-1" element={<Project1 />} />
					<Route path="/project-2" element={<Project2 />} />
					<Route path="/project-3" element={<Project3 />} />
					<Route path="/project-4" element={<Project4 />} />
					<Route path="/contact" element={<Contact />} />
					<Route path="*" element={<NotFound />} />
				</Routes>
			</AnimatePresence>
		</BrowserRouter>
	);
};

export default App;
