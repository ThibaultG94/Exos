<template>
	<div class="row d-inline-flex">
		<h1 class="mt-5">
			{{ title.toUpperCase() }}
		</h1>
		<p class="lead">
			{{ body }}
		</p>
		<p>POST ID : {{ id }} de user {{ userId }}</p>

		<form @submit.prevent="onSubmit">
			<input type="hidden" name="id" v-model="this.id" ref="delete_id" />
			<button class="btn btn-danger">Supprimer</button>
		</form>
	</div>
</template>

<script>
export default {
	name: 'PostDelete',

	data() {
		return {
			apiUrl: 'https://jsonplaceholder.typicode.com/posts/1',
			id: '',
			title: '',
			body: '',
			userId: '',
		};
	},

	methods: {
		onSubmit() {
			const id = this.$refs.delete_id.value;

			this.axios
				.delete('https://jsonplaceholder.typicode.com/posts/' + id)
				.then((response) => {
					console.log(response);
					console.log(response.status);
					if (response.status === 200) {
						alert('Post supprimé');
					}
				});
		},
	},

	mounted() {
		this.axios.get(this.apiUrl).then((response) => {
			console.log(response);
			this.title = response.data.title;
			this.body = response.data.body;
			this.id = response.data.id;
			this.userId = response.data.userId;
		});
	},
};
</script>

<style scoped></style>
