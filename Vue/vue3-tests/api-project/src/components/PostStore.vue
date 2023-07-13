<template>
	<div class="row d-linline-flex">
		<form @submit.prevent="onSubmit">
			<div class="form-group">
				<input
					type="text"
					name="post.title"
					v-model="post.title"
					class="form-control"
					placeholder="Titre" />
			</div>
			<br />
			<div class="form-group">
				<textarea
					name="post.body"
					v-model="post.body"
					class="form-control"
					placeholder="Body..."></textarea>
			</div>
			<br /><br />
			<div class="form-group">
				<button class="btn btn-primary">Envoyer</button>
			</div>
		</form>
	</div>
</template>

<script>
export default {
	name: 'PostStore',

	data() {
		return {
			post: {},
			responseData: null,
		};
	},
	methods: {
		onSubmit() {
			if (this.post.title && this.post.body) {
				console.log('');
			}
			this.responseData = null;
			let body = JSON.stringify({
				title: this.post.title,
				body: this.post.body,
				userId: 2,
			});
			this.axios
				.post('https://jsonplaceholder.typicode.com/posts', body)
				.then((res) => {
					console.log(res);
					console.log(res.status);
					this.responseData = res.data;
					if (res.status === 201) {
						console.log(
							'Post ajouté !' + JSON.stringify(this.responseData)
						);
						this.post = {};
					}
				});
		},
	},
};
</script>

<style scoped></style>
