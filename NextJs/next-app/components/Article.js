import Link from 'next/link';

const Article = ({ article }) => {
	return (
		<Link href="/article/[id]" as={`/article/${article.id}`}>
			<div className="article">
				<h3>{article.title} ➡️</h3>
				<p>{article.body}</p>
			</div>
		</Link>
	);
};

export default Article;
