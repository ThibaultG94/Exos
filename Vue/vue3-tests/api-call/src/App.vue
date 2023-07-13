<template>
	<div id="app" class="container">
		<h1 class="mt-5">
			{{ title.toUpperCase() }}
		</h1>
		<p class="lead">
			{{ body }}
		</p>
		<p>POST ID : {{ id }} de user {{ userId }}</p>
	</div>
</template>

<script>
import axios from 'axios';

export default {
	name: 'App',
	data() {
		return {
			apiUrl: 'https://jsonplaceholder.typicode.com/posts/1',
			id: '',
			title: '',
			body: '',
			userId: '',
		};
	},
	mounted() {
		axios.get(this.apiUrl).then((response) => {
			console.log(response);
			this.title = response.data.title;
			this.body = response.data.body;
			this.id = response.data.id;
			this.userId = response.data.userId;
		});
	},
};
</script>

<style>
#app {
	font-family: Avenir, Helvetica, Arial, sans-serif;
	-webkit-font-smoothing: antialiased;
	-moz-osx-font-smoothing: grayscale;
	text-align: center;
	color: #2c3e50;
	margin-top: 60px;
}
</style>
