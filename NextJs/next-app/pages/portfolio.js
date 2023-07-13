import Link from 'next/link';
import Meta from '../components/Meta';
import { projects } from '../projectsData';

const portfolio = () => {
	console.log(projects);
	return (
		<div>
			<Meta
				title={'Porfolio dev, les tous derniers projets réalisés'}
				description={'suite de 70mots'}
			/>
			<h2>PORTFOLIO</h2>
			<ul>
				{projects.map((project) => (
					<li key={project.id}>
						<Link
							href="/projects/[id]"
							as={'projects/' + project.id}>
							{project.title} <span>➡️</span>
						</Link>
					</li>
				))}
			</ul>
		</div>
	);
};

export default portfolio;
